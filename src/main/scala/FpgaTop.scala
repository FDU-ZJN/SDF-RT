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
    val pixel_x          = Output(UInt(16.W))
    val pixel_y          = Output(UInt(16.W))
    val pixel_rgb        = Output(new Vec3(c.cfg))
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

  val issuedCount  = RegInit(0.U(pixelCountW.W))
  val retiredCount = RegInit(0.U(pixelCountW.W))
  val rayIssueFire = WireDefault(false.B)

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
      validationError := false.B   // Fix 8
      when(frameStartPulse && !setupReg) {
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

  val rayDirFifoDepth   = GlobalConfig.rayDirFifoDepth
  val rayDirCalcLatency =
    4 * c.cfg.fmulLatency + 3 * c.cfg.faddLatency + c.cfg.fsqrtLatency
  val rayReserveW = log2Ceil(rayDirFifoDepth + rayDirCalcLatency + 2)

  class RayDirBundle extends Bundle {
    val dir_x   = UInt(c.cfg.totalWidth.W)
    val dir_y   = UInt(c.cfg.totalWidth.W)
    val dir_z   = UInt(c.cfg.totalWidth.W)
    val pixel_x = UInt(16.W)
    val pixel_y = UInt(16.W)
  }

  val rayDirFifo = Module(new Queue(new RayDirBundle, entries = rayDirFifoDepth))

  val rayDirPipeInflight = RegInit(0.U(rayReserveW.W))
  val pipeIn  = rayIssueFire
  val pipeOut = rayDirCalc.io.out_valid
  when(pipeIn && !pipeOut) {
    rayDirPipeInflight := rayDirPipeInflight + 1.U
  }.elsewhen(!pipeIn && pipeOut) {
    rayDirPipeInflight := rayDirPipeInflight - 1.U
  }


  val rayFifoCountExt = Wire(UInt(rayReserveW.W))
  rayFifoCountExt := rayDirFifo.io.count
  val rayFifoFree         = rayDirFifoDepth.U(rayReserveW.W) - rayFifoCountExt
  val canReserveRayOutput = rayFifoFree > rayDirPipeInflight

  // Optimize: Direct register maintenance for issuePixelX/Y to avoid modulo/division ops
  val issuePixelX = RegInit(0.U(16.W))
  val issuePixelY = RegInit(0.U(16.W))

  val canFireRay =
    (state === rendering) &&
      rayDirCalc.io.in_ready &&
      (issuedCount < totalPixels) &&
      canReserveRayOutput

  when(state === idle) {
    issuePixelX := 0.U
    issuePixelY := 0.U
  }.elsewhen(canFireRay) {
    when(issuePixelX === (width - 1).U) {
      issuePixelX := 0.U
      issuePixelY := issuePixelY + 1.U
    }.otherwise {
      issuePixelX := issuePixelX + 1.U
    }
  }

  rayIssueFire           := canFireRay

  val rayDirInValidReg = RegInit(false.B)
  val rayDirPixelXReg   = RegInit(0.U(16.W))
  val rayDirPixelYReg   = RegInit(0.U(16.W))

  when(state === idle) {
    rayDirInValidReg := false.B
    rayDirPixelXReg   := 0.U
    rayDirPixelYReg   := 0.U
  }.otherwise {
    rayDirInValidReg := canFireRay
    when(canFireRay) {
      rayDirPixelXReg := issuePixelX
      rayDirPixelYReg := issuePixelY
    }
  }

  rayDirCalc.io.in_valid := rayDirInValidReg
  rayDirCalc.io.pixel_x  := rayDirPixelXReg
  rayDirCalc.io.pixel_y  := rayDirPixelYReg

  // RayDirCalc → rayDirFifo
  rayDirCalc.io.out_ready         := rayDirFifo.io.enq.ready
  rayDirFifo.io.enq.valid         := rayDirCalc.io.out_valid
  rayDirFifo.io.enq.bits.dir_x   := rayDirCalc.io.dir_x
  rayDirFifo.io.enq.bits.dir_y   := rayDirCalc.io.dir_y
  rayDirFifo.io.enq.bits.dir_z   := rayDirCalc.io.dir_z
  rayDirFifo.io.enq.bits.pixel_x := rayDirCalc.io.out_pixel_x
  rayDirFifo.io.enq.bits.pixel_y := rayDirCalc.io.out_pixel_y

  assert(
    !(rayDirCalc.io.out_valid && !rayDirFifo.io.enq.ready),
    "[FpgaTop] BUG: rayDirFifo full despite reservation!"
  )

  val canIssueRay = rayDirFifo.io.deq.valid && simTop.io.out_ready
  rayDirFifo.io.deq.ready := canIssueRay

  simTop.io.rd_valid := canIssueRay
  simTop.io.rd_in.x  := rayDirFifo.io.deq.bits.dir_x
  simTop.io.rd_in.y  := rayDirFifo.io.deq.bits.dir_y
  simTop.io.rd_in.z  := rayDirFifo.io.deq.bits.dir_z


  class PixelBundle extends Bundle {
    val x      = UInt(16.W)
    val y      = UInt(16.W)
    val rgb    = new Vec3(c.cfg)
    val hit_id = UInt(c.addrWidth.W)
  }

  val pixelQueue = Module(new Queue(new PixelBundle, pixelQueueDepth))

  // Optimize: direct result X/Y registers to avoid modulo/division on critical path
  val resultX = RegInit(0.U(16.W))
  val resultY = RegInit(0.U(16.W))

  when(state === idle) {
    resultX := 0.U
    resultY := 0.U
  }.elsewhen(enqFire) {
    when(resultX === (width - 1).U) {
      resultX := 0.U
      resultY := resultY + 1.U
    }.otherwise {
      resultX := resultX + 1.U
    }
  }

  // Fix 3: enqFire = 真实握手成功（valid && ready 同时为真）
  enqFire := simTop.io.out_valid && pixelQueue.io.enq.ready

  when(retireFireReg) {
    retiredCount := retiredCount + 1.U   // Fix 3
  }

  pixelQueue.io.enq.valid        := simTop.io.out_valid
  pixelQueue.io.enq.bits.x      := resultX
  pixelQueue.io.enq.bits.y      := resultY
  pixelQueue.io.enq.bits.rgb    := simTop.io.out_rgb
  pixelQueue.io.enq.bits.hit_id := simTop.io.out_id
  pixelQueue.io.deq.ready       := io.pixel_ready

  assert(
    !(simTop.io.out_valid && !pixelQueue.io.enq.ready),
    "[FpgaTop] BUG: pixelQueue full! Increase pixelQueueDepth or simTopMaxInflight"
  )

  io.pixel_valid  := pixelQueue.io.deq.valid
  io.pixel_x      := pixelQueue.io.deq.bits.x
  io.pixel_y      := pixelQueue.io.deq.bits.y
  io.pixel_rgb    := pixelQueue.io.deq.bits.rgb
  io.pixel_hit_id := pixelQueue.io.deq.bits.hit_id

  // =========================================================================
  // 合法性检查
  // =========================================================================
  val underflow = retiredCount > issuedCount
  val overflow  = issuedCount > totalPixels

  when(underflow || overflow) {
    validationError := true.B
  }

  val stallCounter = RegInit(0.U(16.W))
  val progressMade = canFireRay || enqFire

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
object FpgaTopGen extends App {
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

  def generateFpgaTopVerilog(targetDir: String = "build/fpga",withUseFloatIP:Boolean=true): Unit = {
    println("Generating FPGA_TOP Verilog...")
    GlobalConfig.withMemImplMode(2) {
      GlobalConfig.withUseFloatIP(withUseFloatIP) {
        emitVerilog(
          new FpgaTop(
            width = GlobalConfig.frameWidth,
            height = GlobalConfig.frameWidth,
            pixelQueueDepth = GlobalConfig.pixelQueueDepth
          ),
          Array("--target-dir", targetDir)
        )
      }
    }

    stripFirrtlBbFileList(targetDir)
    println(s"FPGA_TOP generated in $targetDir")
  }

  generateFpgaTopVerilog(targetDir = "build/vivado")
  generateFpgaTopVerilog(withUseFloatIP = false)
}