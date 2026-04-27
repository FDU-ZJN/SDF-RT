import chisel3._
import chisel3.util._
import raytrace_utils._
import raytrace_utils.fudian._
import SDF.SdfMemWriteIO

import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Paths}
import scala.io.Source

class FpgaTop(
               width:           Int = GlobalConfig.frameWidth,
               height:          Int = GlobalConfig.frameHeight,
               pixelQueueDepth: Int = GlobalConfig.pixelQueueDepth
             ) extends Module {
  val c = TriPeConfig(cfg = FloatConfig.FP32.copy())

  val io = IO(new Bundle {
    val setup_valid      = Input(Bool())
    val setup_origin     = Input(new Vec3(c.cfg))
    val setup_grid_min   = Input(new Vec3(c.cfg))
    val setup_grid_max   = Input(new Vec3(c.cfg))
    val setup_ready      = Output(Bool())

    val frame_start      = Input(Bool())

    val pixel_valid      = Output(Bool())
    val pixel_ready      = Input(Bool())
    val pixel_rgb8       = Output(UInt(24.W))
    val pixel_hit_id     = Output(UInt(c.addrWidth.W))

    val frame_done       = Output(Bool())
    val busy             = Output(Bool())
    val frame_count      = Output(UInt(32.W))
    val validation_error = Output(Bool())
    val stall_detected   = Output(Bool())

    // SDF memory write port for PS initialization
    val sdf_mem_wr = Flipped(new SdfMemWriteIO)
  })

  val simTop     = Module(new SimTop)
  val rayDirCalc = Module(new RayDirCalc(c.cfg))

  val setupReg        = RegInit(false.B)
  val setupOriginReg  = RegInit(0.U.asTypeOf(new Vec3(c.cfg)))
  val setupGridMinReg = RegInit(0.U.asTypeOf(new Vec3(c.cfg)))
  val setupGridMaxReg = RegInit(0.U.asTypeOf(new Vec3(c.cfg)))

  when(io.setup_valid && !setupReg) {
    setupOriginReg  := io.setup_origin
    setupGridMinReg := io.setup_grid_min
    setupGridMaxReg := io.setup_grid_max
    setupReg        := true.B
  }.elsewhen(simTop.io.setup_finish) {
    setupReg := false.B
  }

  simTop.io.setup_valid    := setupReg
  simTop.io.setup_origin   := setupOriginReg
  simTop.io.setup_grid_min := setupGridMinReg
  simTop.io.setup_grid_max := setupGridMaxReg
  io.setup_ready           := !setupReg
  
  // Connect SDF memory write port
  simTop.io.sdf_mem_wr <> io.sdf_mem_wr

  val idle :: rendering :: frameComplete :: Nil = Enum(3)
  val state = RegInit(idle)

  val pixelCountW   = log2Ceil(width * height + 1)
  val totalPixels   = (width * height).U(pixelCountW.W)
  val totalPixels1  = (width * height - 1).U(pixelCountW.W)
  val frameCountReg = RegInit(0.U(32.W))
  io.frame_count := frameCountReg

  val frameStartReg   = RegNext(io.frame_start, false.B)
  val frameStartPulse = io.frame_start && !frameStartReg

  val frameDonePulse = WireInit(false.B)
  io.frame_done := frameDonePulse

  val rayDirFifoDepth = GlobalConfig.rayDirFifoDepth
  val issuedCount  = RegInit(0.U(pixelCountW.W))
  val retiredCount = RegInit(0.U(pixelCountW.W))
  val rayIssueFire = WireDefault(false.B)
  val bootstrapLaunches = RegInit(0.U(log2Ceil(rayDirFifoDepth + 1).W))

  val validationError = RegInit(false.B)   // Fix 8
  io.validation_error := validationError

  val enqFire = WireDefault(false.B)

  // Optimize: terminal flag avoids retiredCount+1 on the state-transition path
  val retireFireReg = RegInit(false.B)
  when(state === idle) {
    retireFireReg := false.B
  }.otherwise {
    retireFireReg := enqFire
  }

  val retiredCountWillFinish = retireFireReg && (retiredCount === totalPixels1)

  switch(state) {
    is(idle) {
      issuedCount     := 0.U
      retiredCount    := 0.U
      bootstrapLaunches := 0.U
      validationError := false.B   // Fix 8
      when(frameStartPulse && !setupReg) {
        bootstrapLaunches := Mux(totalPixels < rayDirFifoDepth.U, totalPixels, rayDirFifoDepth.U)
        state := rendering
      }
    }
    is(rendering) {
      when(rayIssueFire) {
        issuedCount := issuedCount + 1.U
      }
      when(retiredCountWillFinish) {
        state := frameComplete
      }
    }
    is(frameComplete) {
      frameDonePulse := true.B
      frameCountReg  := frameCountReg + 1.U
      state          := idle
    }
  }

  io.busy := (state === rendering)

  class RayDirBundle extends Bundle {
    val dir_x   = UInt(c.cfg.totalWidth.W)
    val dir_y   = UInt(c.cfg.totalWidth.W)
    val dir_z   = UInt(c.cfg.totalWidth.W)
  }

  val rayDirFifo = Module(new Queue(new RayDirBundle, entries = rayDirFifoDepth))
  val simTopFire = rayDirFifo.io.deq.valid && simTop.io.out_ready
  val shouldLaunchRay =
    (state === rendering) &&
      rayDirCalc.io.in_ready &&
      (issuedCount < totalPixels) &&
      ((bootstrapLaunches =/= 0.U) || simTopFire)

  rayIssueFire := shouldLaunchRay

  when(state === rendering && rayIssueFire && (bootstrapLaunches =/= 0.U)) {
    bootstrapLaunches := bootstrapLaunches - 1.U
  }

  rayDirCalc.io.clear := (state === idle)
  rayDirCalc.io.in_valid := rayIssueFire

  // RayDirCalc → rayDirFifo
  rayDirCalc.io.out_ready         := rayDirFifo.io.enq.ready
  rayDirFifo.io.enq.valid         := rayDirCalc.io.out_valid
  rayDirFifo.io.enq.bits.dir_x   := rayDirCalc.io.dir_x
  rayDirFifo.io.enq.bits.dir_y   := rayDirCalc.io.dir_y
  rayDirFifo.io.enq.bits.dir_z   := rayDirCalc.io.dir_z

  assert(
    !(rayDirCalc.io.out_valid && !rayDirFifo.io.enq.ready),
    "[FpgaTop] BUG: rayDirFifo overflow!"
  )

  rayDirFifo.io.deq.ready := simTopFire

  simTop.io.rd_valid := simTopFire
  simTop.io.rd_in.x  := rayDirFifo.io.deq.bits.dir_x
  simTop.io.rd_in.y  := rayDirFifo.io.deq.bits.dir_y
  simTop.io.rd_in.z  := rayDirFifo.io.deq.bits.dir_z


  class PixelBundle extends Bundle {
    val rgb8   = UInt(24.W)
    val hit_id = UInt(c.addrWidth.W)
  }

  val pixelQueue = Module(new Queue(new PixelBundle, pixelQueueDepth))

  // Fix 3: enqFire = 真实握手成功（valid && ready 同时为真）
  enqFire := simTop.io.out_valid && pixelQueue.io.enq.ready

  when(retireFireReg) {
    retiredCount := retiredCount + 1.U   // Fix 3
  }

  pixelQueue.io.enq.valid        := simTop.io.out_valid
  pixelQueue.io.enq.bits.rgb8   := simTop.io.out_rgb8
  pixelQueue.io.enq.bits.hit_id := simTop.io.out_id
  pixelQueue.io.deq.ready       := io.pixel_ready

  assert(
    !(simTop.io.out_valid && !pixelQueue.io.enq.ready),
    "[FpgaTop] BUG: pixelQueue full! Increase pixelQueueDepth or simTopMaxInflight"
  )

  io.pixel_valid  := pixelQueue.io.deq.valid
  io.pixel_rgb8   := pixelQueue.io.deq.bits.rgb8
  io.pixel_hit_id := pixelQueue.io.deq.bits.hit_id


  val underflow = retiredCount > issuedCount
  val overflow  = issuedCount > totalPixels

  when(underflow || overflow) {
    validationError := true.B
  }

  val stallCounter = RegInit(0.U(16.W))
  val progressMade = rayIssueFire || enqFire

  when(state === idle) {
    stallCounter := 0.U           // Fix 7
  }.elsewhen(progressMade) {
    stallCounter := 0.U
  }.elsewhen(state === rendering) {
    stallCounter := stallCounter + 1.U
  }

  val stallTimeout = stallCounter > 10000.U
  io.stall_detected := stallTimeout

  when(stallTimeout && (state === rendering)) {
    validationError := true.B
  }
}

// =============================================================================
// Verilog 生成入口
// =============================================================================
object FpgaTopGen {
  private val bbFileListMarker =
    "// ----- 8< ----- FILE \"firrtl_black_box_resource_files.f\" ----- 8< -----"

  private def stripFirrtlBbFileList(targetDir: String): Unit = {
    val outPath = Paths.get(targetDir, "FpgaTop.sv")
    if (!Files.exists(outPath)) return
    val src  = Source.fromFile(outPath.toFile, "UTF-8")
    val text = try src.mkString finally src.close()
    val idx  = text.indexOf(bbFileListMarker)
    if (idx >= 0) {
      val trimmed = text.substring(0, idx).trim + "\n"
      Files.write(outPath, trimmed.getBytes(StandardCharsets.UTF_8))
    }
  }

  def generateFpgaTopVerilog(targetDir: String = "build/fpga",withMemImplMode :Int=0,withUseFloatIP:Boolean=true): Unit = {
    println("Generating FPGA_TOP Verilog...")
    GlobalConfig.withMemImplMode(withMemImplMode) {
      GlobalConfig.withUseFloatIP(withUseFloatIP) {
        emitVerilog(
          new FpgaTop(
            width = GlobalConfig.frameWidth,
            height = GlobalConfig.frameHeight,
            pixelQueueDepth = GlobalConfig.pixelQueueDepth
          ),
          Array("--target-dir", targetDir)
        )
      }
    }

    stripFirrtlBbFileList(targetDir)
    println(s"FPGA_TOP generated in $targetDir")
  }

  def main(args: Array[String]): Unit = {
    generateFpgaTopVerilog(targetDir = "build/vivado", withMemImplMode = 2)
    generateFpgaTopVerilog(withUseFloatIP = false)
  }
}
