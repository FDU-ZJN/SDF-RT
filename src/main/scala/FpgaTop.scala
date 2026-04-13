import chisel3._
import chisel3.util._
import raytrace_utils._
import raytrace_utils.fudian._

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

  val idle :: rendering :: frameComplete :: Nil = Enum(3)
  val state = RegInit(idle)

  val totalPixels   = (width * height).U(32.W)
  val frameCountReg = RegInit(0.U(32.W))
  io.frame_count := frameCountReg

  val frameStartReg   = RegNext(io.frame_start, false.B)
  val frameStartPulse = io.frame_start && !frameStartReg

  val frameDonePulse = WireInit(false.B)
  io.frame_done := frameDonePulse

  val issuedCount  = RegInit(0.U(32.W))
  val retiredCount = RegInit(0.U(32.W))
  val rayIssueFire = WireDefault(false.B)

  val validationError = RegInit(false.B)   // Fix 8
  io.validation_error := validationError

  val enqFire = Wire(Bool())

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
      // retiredCount 在 pixelQueue 节自增（Fix 3）
      when(retiredCount >= totalPixels) {
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

  val issuePixelCounter = RegInit(0.U(32.W))
  when(state === idle) {
    issuePixelCounter := 0.U
  }
  val issuePixelX = issuePixelCounter % width.U
  val issuePixelY = issuePixelCounter / width.U

  val canFireRay =
    (state === rendering) &&
      rayDirCalc.io.in_ready &&
      (issuedCount < totalPixels) &&
      canReserveRayOutput

  rayIssueFire           := canFireRay
  rayDirCalc.io.in_valid := canFireRay
  rayDirCalc.io.pixel_x  := issuePixelX
  rayDirCalc.io.pixel_y  := issuePixelY

  when(canFireRay) {
    issuePixelCounter := issuePixelCounter + 1.U
  }

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

  // Fix 6: idle 时重置结果坐标计数器
  val resultCounter = RegInit(0.U(32.W))
  when(state === idle) {
    resultCounter := 0.U
  }
  val resultX = resultCounter % width.U
  val resultY = resultCounter / width.U

  // Fix 3: enqFire = 真实握手成功（valid && ready 同时为真）
  enqFire := simTop.io.out_valid && pixelQueue.io.enq.ready

  when(enqFire) {
    resultCounter := resultCounter + 1.U
    retiredCount  := retiredCount + 1.U   // Fix 3
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

  // =========================================================================
  // 停滞检测（Fix 7: idle 时重置）
  // =========================================================================
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
    GlobalConfig.withUseBlackBox(true) {
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