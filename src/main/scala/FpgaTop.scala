import chisel3._
import chisel3.util._
import raytrace_utils._
import raytrace_utils.fudian._
import DDA.DDA
import SDF.{InitStage, SdfMemWriteIO, SdfStage, SetupUnit}
import Trace.TraceController

import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Paths}
import scala.io.Source

class FpgaTop(
               maxWidth:        Int = GlobalConfig.frameWidth,
               maxHeight:       Int = GlobalConfig.frameHeight,
               traceRespQueueDepth: Int = GlobalConfig.pixelQueueDepth
             ) extends Module {
  val c = TriPeConfig(cfg = FloatConfig.FP32.copy())
  private val sdfCfg = SdfPeConfig(cfg = c.cfg)
  private val postHitQueueDepth = 2
  private val initOutLatency = 4 + c.cfg.fdivLatency + c.cfg.fmulLatency + (4 * c.cfg.fcmpLatency) + 1
  private val rayDirLatency = 4 + c.cfg.fsqrtLatency + c.cfg.fmulLatency
  private val postInitQueueDepth = rayDirLatency + initOutLatency + 32
  require(postInitQueueDepth > rayDirLatency + initOutLatency + 2,
    "postInitQueueDepth must absorb the non-backpressurable RayDir+Init pipeline tail")

  val io = IO(new Bundle {
    val setup_valid      = Input(Bool())
    val setup_origin     = Input(new Vec3(c.cfg))
    val setup_grid_min   = Input(new Vec3(c.cfg))
    val setup_grid_max   = Input(new Vec3(c.cfg))
    val setup_res_x      = Input(UInt(16.W))
    val setup_res_y      = Input(UInt(16.W))
    val setup_ready      = Output(Bool())

    val frame_start      = Input(Bool())

    val trace_resp_valid = Output(Bool())
    val trace_resp_slotId = Output(Vec(2, UInt(GlobalConfig.slotBits.W)))
    val trace_resp_hit    = Output(Vec(2, Bool()))
    val trace_resp_hitId  = Output(Vec(2, UInt(c.addrWidth.W)))

    val trace_resp_ready = Input(Bool())

    // SDF memory write port for PS initialization
    val sdf_mem_wr = Flipped(new SdfMemWriteIO)
  })

  val rayDirCalcs = Seq.tabulate(2)(lane => Module(new RayDirCalc(c.cfg, maxWidth, maxHeight, laneId = lane, numLanes = 2)))
  val initStages = Seq.fill(2)(Module(new InitStage(c.cfg, c.addrWidth)))
  val setupUnit = Module(new SetupUnit(c.cfg, sdfCfg))
  val sdfStage = Module(new SdfStage(c.cfg, c.addrWidth))
  val ddaStage = Module(new DDA(c.cfg, c.addrWidth, globalRes = sdfCfg.DDAGlobalRes, subRes = sdfCfg.SubRes, maxTraversalSteps = sdfCfg.DDAMaxSteps))
  val traceController = Module(new TraceController(c, sdfCfg.DDAMaxSteps))
  val traceJobQueue = Module(new Queue(new DdaTraceJobDesc(c.cfg, c.addrWidth, sdfCfg.DDAMaxSteps), GlobalConfig.triBatchQueueDepth))
  val commitQueue = Module(new CommitQueue(c.cfg))
  val postHitQs = Seq.fill(2)(Module(new Queue(new RayIssue(c.cfg, c.addrWidth), postHitQueueDepth)))
  val postInitQs = Seq.fill(2)(Module(new Queue(new InitStageResp(c.cfg, c.addrWidth), postInitQueueDepth)))

  val setupReg        = RegInit(false.B)
  val setupOriginReg  = Reg(new Vec3(c.cfg))
  val setupGridMinReg = Reg(new Vec3(c.cfg))
  val setupGridMaxReg = Reg(new Vec3(c.cfg))
  val resXReg         = RegInit(0.U(16.W))
  val resYReg         = RegInit(0.U(16.W))
  val zScaledMagReg   = RegInit(0.U(18.W))
  val zScaledSqReg    = RegInit(0.U(36.W))
  val zScaledFpReg    = RegInit(0.U(32.W))

  when(io.setup_valid && !setupReg) {
    setupOriginReg  := io.setup_origin
    setupGridMinReg := io.setup_grid_min
    setupGridMaxReg := io.setup_grid_max
    resXReg         := io.setup_res_x
    resYReg         := io.setup_res_y
    setupReg        := true.B
  }.elsewhen(setupUnit.io.setup_finish) {
    setupReg := false.B
  }

  private def unsignedToFp32(value: UInt): UInt = {
    val width = value.getWidth
    val isZero = value === 0.U
    val msbOH = PriorityEncoderOH(Reverse(value))
    val msbFromTop = OHToUInt(msbOH)
    val msbIdx = (width - 1).U - msbFromTop
    val exp = (127.U(8.W) + msbIdx)(7, 0)
    val aligned = (value << ((width - 1).U - msbIdx))(width - 1, 0)
    val frac = if (width >= 24) aligned(width - 2, width - 24) else Cat(aligned(width - 2, 0), 0.U((24 - width).W))
    Mux(isZero, 0.U(32.W), Cat(0.U(1.W), exp, frac))
  }

  val zScaledMag = (9.U * resYReg) / 5.U
  val zScaledFpBase = unsignedToFp32(zScaledMag)
  val zScaledFpNeg = Cat(1.U(1.W), zScaledFpBase(30, 0))
  zScaledMagReg := zScaledMag
  zScaledSqReg := zScaledMag * zScaledMag
  zScaledFpReg := zScaledFpNeg

  setupUnit.io.setup_valid := setupReg
  setupUnit.io.setup_origin := setupOriginReg
  setupUnit.io.setup_grid_min := setupGridMinReg
  setupUnit.io.setup_grid_max := setupGridMaxReg
  io.setup_ready := !setupReg
  
  for (lane <- 0 until 2) {
    initStages(lane).io.setup_origin := setupUnit.io.origin
    initStages(lane).io.setup_grid_min := setupUnit.io.gridMin
    initStages(lane).io.setup_grid_max := setupUnit.io.gridMax
  }

  sdfStage.io.grid_min := setupUnit.io.gridMin
  sdfStage.io.inv_voxel := setupUnit.io.invVoxel
  sdfStage.io.sdf_mem_wr <> io.sdf_mem_wr

  ddaStage.io.grid_min := setupUnit.io.gridMin
  ddaStage.io.inv_sub_voxel := setupUnit.io.invSubVoxel
  ddaStage.io.slot_release := traceController.io.slot_release
  for (lane <- 0 until 2) {
    traceController.io.cmd_write(lane).valid := ddaStage.io.cmd_write(lane).valid
    traceController.io.cmd_write(lane).bits := ddaStage.io.cmd_write(lane).bits
  }
  traceJobQueue.io.enq <> ddaStage.io.trace_job_out
  when(traceJobQueue.io.enq.valid) {
    assert(traceJobQueue.io.enq.ready, "FpgaTop traceJobQueue overflow")
  }
  traceController.io.job_in <> traceJobQueue.io.deq

  val traceWbQ = Module(new Queue(new RenderResult(c.cfg, c.addrWidth), 16))
  traceWbQ.io.enq.valid := traceController.io.result_out.valid
  traceWbQ.io.enq.bits.meta := traceController.io.result_out.bits.meta
  traceWbQ.io.enq.bits.hit := traceController.io.result_out.bits.hit
  traceWbQ.io.enq.bits.hitId := traceController.io.result_out.bits.hitId
  traceWbQ.io.enq.bits.rgb8 := 0.U
  traceController.io.result_out.ready := traceWbQ.io.enq.ready

  commitQueue.io.writeback <> traceWbQ.io.deq
  commitQueue.io.writeback6.valid := false.B
  commitQueue.io.writeback6.bits  := 0.U.asTypeOf(new RenderResult(c.cfg, c.addrWidth))
  commitQueue.io.traceDone(0).valid := false.B
  commitQueue.io.traceDone(0).bits  := 0.U
  commitQueue.io.traceDone(1).valid := false.B
  commitQueue.io.traceDone(1).bits  := 0.U

  val idle :: rendering :: frameComplete :: Nil = Enum(3)
  val state = RegInit(idle)

  val pixelCountW   = log2Ceil(maxWidth * maxHeight + 1)
  val totalPixelsRuntime = resXReg * resYReg
  val totalPairsRuntime = totalPixelsRuntime >> 1
  val frameCountReg = RegInit(0.U(32.W))

  val frameStartReg   = RegNext(io.frame_start, false.B)
  val frameStartPulse = io.frame_start && !frameStartReg

  val frameDonePulse = WireInit(false.B)

  val setupReady = setupUnit.io.setup_finish
  val inputPair = WireDefault(false.B)

  for (lane <- 0 until 2) {
    initStages(lane).io.in.valid := inputPair
    initStages(lane).io.in.bits.rd := 0.U.asTypeOf(new Vec3(c.cfg))
    initStages(lane).io.in.bits.meta.slotId := 0.U
    initStages(lane).io.in.bits.meta.pixelX := 0.U
    initStages(lane).io.in.bits.meta.pixelY := 0.U
    postInitQs(lane).io.enq <> initStages(lane).io.out
  }

  val postInitPairValid = postInitQs(0).io.deq.valid && postInitQs(1).io.deq.valid
  val postInitHit0 = postInitQs(0).io.deq.bits.hit
  val postInitHit1 = postInitQs(1).io.deq.bits.hit
  val initMissWb2ValidReg = RegInit(false.B)
  val initMissWb2BitsReg = Reg(new RenderResult(c.cfg, c.addrWidth))
  val initMissWb4ValidReg = RegInit(false.B)
  val initMissWb4BitsReg = Reg(new RenderResult(c.cfg, c.addrWidth))
  val initMissWb2CanAccept = !initMissWb2ValidReg || commitQueue.io.writeback2.ready
  val initMissWb4CanAccept = !initMissWb4ValidReg || commitQueue.io.writeback4.ready
  val postInitCanPushHits =
    (!postInitHit0 || postHitQs(0).io.enq.ready) &&
      (!postInitHit1 || postHitQs(1).io.enq.ready)
  val postInitCanPushMisses =
    (postInitHit0 || initMissWb2CanAccept) &&
      (postInitHit1 || initMissWb4CanAccept)
  val postInitCanAlloc = commitQueue.io.alloc(0).ready && commitQueue.io.alloc(1).ready
  val postInitDispatch = postInitPairValid && postInitCanPushHits && postInitCanPushMisses && postInitCanAlloc

  commitQueue.io.alloc(0).valid := postInitDispatch
  commitQueue.io.alloc(0).bits := 0.U
  commitQueue.io.alloc(1).valid := postInitDispatch
  commitQueue.io.alloc(1).bits := 0.U
  when(postInitDispatch) {
    assert(commitQueue.io.alloc(0).ready, "FpgaTop postInit commit alloc0 overflow")
    assert(commitQueue.io.alloc(1).ready, "FpgaTop postInit commit alloc1 overflow")
  }

  for (lane <- 0 until 2) {
    val hitPush = postInitDispatch && postInitQs(lane).io.deq.bits.hit
    postHitQs(lane).io.enq.valid := hitPush
    postHitQs(lane).io.enq.bits.ray := postInitQs(lane).io.deq.bits.ray
    postHitQs(lane).io.enq.bits.meta := postInitQs(lane).io.deq.bits.meta
    postHitQs(lane).io.enq.bits.meta.slotId := commitQueue.io.allocSlot(lane)
    postInitQs(lane).io.deq.ready := postInitDispatch
    when(hitPush) {
      assert(postHitQs(lane).io.enq.ready, s"FpgaTop postHitQ lane $lane overflow")
    }
  }

  val sdfMissWbQs = Seq.fill(2)(Module(new Queue(new RenderResult(c.cfg, c.addrWidth), GlobalConfig.sdfMissWritebackQueueDepth)))

  for (lane <- 0 until 2) {
    sdfStage.io.issue_in(lane) <> postHitQs(lane).io.deq

    val sdfOut = sdfStage.io.out(lane).bits
    val sdfOutValid = sdfStage.io.out(lane).valid
    val sdfOutHit = sdfOut.hit

    val wb3 = Wire(new RenderResult(c.cfg, c.addrWidth))
    wb3.meta := sdfOut.meta
    wb3.hit := false.B
    wb3.hitId := 0.U
    wb3.rgb8 := 0.U

    ddaStage.io.in(lane).valid := sdfOutValid && sdfOutHit
    ddaStage.io.in(lane).bits.ray := sdfOut.ray
    ddaStage.io.in(lane).bits.meta := sdfOut.meta
    ddaStage.io.in(lane).bits.traceSlot := 0.U

    sdfMissWbQs(lane).io.enq.valid := sdfOutValid && !sdfOutHit
    sdfMissWbQs(lane).io.enq.bits := wb3

    sdfStage.io.out(lane).ready := Mux(sdfOutHit, ddaStage.io.in(lane).ready, sdfMissWbQs(lane).io.enq.ready)
  }

  val wb2 = Wire(new RenderResult(c.cfg, c.addrWidth))
  wb2.meta := postInitQs(0).io.deq.bits.meta
  wb2.meta.slotId := commitQueue.io.allocSlot(0)
  wb2.hit := false.B
  wb2.hitId := 0.U
  wb2.rgb8 := 0.U

  val wb4 = Wire(new RenderResult(c.cfg, c.addrWidth))
  wb4.meta := postInitQs(1).io.deq.bits.meta
  wb4.meta.slotId := commitQueue.io.allocSlot(1)
  wb4.hit := false.B
  wb4.hitId := 0.U
  wb4.rgb8 := 0.U

  when(commitQueue.io.writeback2.ready) {
    initMissWb2ValidReg := false.B
  }
  when(commitQueue.io.writeback4.ready) {
    initMissWb4ValidReg := false.B
  }
  when(postInitDispatch && !postInitQs(0).io.deq.bits.hit) {
    assert(initMissWb2CanAccept, "FpgaTop init miss lane0 writeback buffer overflow")
    initMissWb2ValidReg := true.B
    initMissWb2BitsReg := wb2
  }
  when(postInitDispatch && !postInitQs(1).io.deq.bits.hit) {
    assert(initMissWb4CanAccept, "FpgaTop init miss lane1 writeback buffer overflow")
    initMissWb4ValidReg := true.B
    initMissWb4BitsReg := wb4
  }

  commitQueue.io.writeback2.valid := initMissWb2ValidReg
  commitQueue.io.writeback2.bits := initMissWb2BitsReg
  commitQueue.io.writeback4.valid := initMissWb4ValidReg
  commitQueue.io.writeback4.bits := initMissWb4BitsReg

  commitQueue.io.writeback3 <> sdfMissWbQs(0).io.deq
  commitQueue.io.writeback5 <> sdfMissWbQs(1).io.deq

  val issuedCount = RegInit(0.U(pixelCountW.W))
  val retiredCount = RegInit(0.U(pixelCountW.W))
  val remainingPairsReg = RegInit(0.U(pixelCountW.W))
  val rayIssueFire0 = WireDefault(false.B)
  val rayIssueFire1 = WireDefault(false.B)
  val rayIssueCount = WireDefault(0.U(2.W))
  val canLaunchPairReg = RegInit(false.B)
  val canLaunchPairNext = WireDefault(false.B)
  val issuePairFire = canLaunchPairReg && (state === rendering) && (remainingPairsReg =/= 0.U)

  val validationError = RegInit(false.B)   // Fix 8
  val enqFire = WireDefault(false.B)
  val retiredPixelsThisBeat = WireDefault(0.U(2.W))

  switch(state) {
    is(idle) {
      issuedCount      := 0.U
      retiredCount     := 0.U
      remainingPairsReg := 0.U
      canLaunchPairReg := false.B
      validationError := false.B   // Fix 8
      when(frameStartPulse && !setupReg) {
        remainingPairsReg := totalPairsRuntime
        state := rendering
      }
    }
    is(rendering) {
      canLaunchPairReg := canLaunchPairNext
      when(rayIssueCount =/= 0.U) {
        assert(issuedCount + rayIssueCount <= totalPixelsRuntime, "FpgaTop issued more rays than totalPixels")
        issuedCount := issuedCount + rayIssueCount
      }
      when(issuePairFire && (remainingPairsReg =/= 0.U)) {
        remainingPairsReg := remainingPairsReg - 1.U
      }
      when(enqFire) {
        assert(retiredCount + retiredPixelsThisBeat <= totalPixelsRuntime, "FpgaTop retired more pixels than totalPixels")
        retiredCount := retiredCount + retiredPixelsThisBeat
      }
      when(enqFire && (retiredCount + retiredPixelsThisBeat >= totalPixelsRuntime)) {
        state := frameComplete
      }
    }
    is(frameComplete) {
      frameDonePulse := true.B
      frameCountReg  := frameCountReg + 1.U
      canLaunchPairReg := false.B
      state          := idle
    }
  }

  val rayPush0 = rayDirCalcs(0).io.out_valid
  val rayPush1 = rayDirCalcs(1).io.out_valid
  val canFirePair = rayPush0 && rayPush1 && setupReady
  inputPair := canFirePair

  private val postInitCountLimit = postInitQueueDepth - (rayDirLatency + initOutLatency + 4)
  require(postInitCountLimit >= 2, "postInitQueueDepth leaves too little issue headroom")
  val postInitCanAbsorbTail = postInitQs.map { q =>
    q.io.count <= postInitCountLimit.U
  }.reduce(_ && _)
  canLaunchPairNext :=
    (state === rendering) &&
      (remainingPairsReg =/= 0.U) &&
      setupReady &&
      postInitCanAbsorbTail
  rayIssueFire0 := issuePairFire
  rayIssueFire1 := issuePairFire
  rayIssueCount := Mux(issuePairFire, 2.U, 0.U)

  for (lane <- 0 until 2) {
    rayDirCalcs(lane).io.clear := (state === idle)
    rayDirCalcs(lane).io.in_valid := (if (lane == 0) rayIssueFire0 else rayIssueFire1)
    rayDirCalcs(lane).io.out_ready := true.B
    rayDirCalcs(lane).io.width_in := resXReg
    rayDirCalcs(lane).io.height_in := resYReg
    rayDirCalcs(lane).io.z_scaled_fp := zScaledFpReg
    rayDirCalcs(lane).io.z_scaled_sq := zScaledSqReg
  }

  assert(rayPush0 === rayPush1, "[FpgaTop] BUG: rayDir lanes lost pair alignment")
  assert(!(rayPush0 && !setupReady), "[FpgaTop] BUG: direct rayDir output not accepted by init pipeline")
  assert(totalPixelsRuntime(0) === 0.U, "[FpgaTop] pair-only issue path requires an even total pixel count")

  initStages(0).io.in.bits.rd.x := rayDirCalcs(0).io.dir_x
  initStages(0).io.in.bits.rd.y := rayDirCalcs(0).io.dir_y
  initStages(0).io.in.bits.rd.z := rayDirCalcs(0).io.dir_z
  initStages(1).io.in.bits.rd.x := rayDirCalcs(1).io.dir_x
  initStages(1).io.in.bits.rd.y := rayDirCalcs(1).io.dir_y
  initStages(1).io.in.bits.rd.z := rayDirCalcs(1).io.dir_z

  class TraceRespPair extends Bundle {
    val lane0 = new RenderResult(c.cfg, c.addrWidth)
    val lane1 = new RenderResult(c.cfg, c.addrWidth)
  }

  val traceRespPairQ = Module(new Queue(new TraceRespPair, traceRespQueueDepth))

  val commitOutValid = commitQueue.io.out.valid
  val commitRespBufferFire = commitOutValid && traceRespPairQ.io.enq.ready

  traceRespPairQ.io.enq.valid := commitOutValid
  traceRespPairQ.io.enq.bits.lane0 := commitQueue.io.out.bits(0)
  traceRespPairQ.io.enq.bits.lane1 := commitQueue.io.out.bits(1)

  assert(
    !(commitOutValid && !traceRespPairQ.io.enq.ready),
    "[FpgaTop] BUG: trace response queue full! Increase top-side response buffering"
  )

  io.trace_resp_valid := traceRespPairQ.io.deq.valid
  io.trace_resp_slotId(0) := traceRespPairQ.io.deq.bits.lane0.meta.slotId
  io.trace_resp_slotId(1) := traceRespPairQ.io.deq.bits.lane1.meta.slotId
  io.trace_resp_hit(0) := traceRespPairQ.io.deq.bits.lane0.hit
  io.trace_resp_hit(1) := traceRespPairQ.io.deq.bits.lane1.hit
  io.trace_resp_hitId(0) := traceRespPairQ.io.deq.bits.lane0.hitId
  io.trace_resp_hitId(1) := traceRespPairQ.io.deq.bits.lane1.hitId
  traceRespPairQ.io.deq.ready := io.trace_resp_ready

  val traceRespFire = traceRespPairQ.io.deq.fire
  retiredPixelsThisBeat := Mux(traceRespFire, 2.U, 0.U)
  enqFire := traceRespFire


  val underflow = retiredCount > issuedCount
  val overflow  = issuedCount > totalPixelsRuntime

  when(underflow || overflow) {
    validationError := true.B
  }

  val stallCounter = RegInit(0.U(16.W))
  val progressMade = rayIssueFire0 || rayIssueFire1 || commitRespBufferFire || traceRespFire

  when(state === idle) {
    stallCounter := 0.U           // Fix 7
  }.elsewhen(progressMade) {
    stallCounter := 0.U
  }.elsewhen(state === rendering) {
    stallCounter := stallCounter + 1.U
  }

  val stallTimeout = stallCounter > 10000.U

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
            maxWidth = GlobalConfig.frameWidth,
            maxHeight = GlobalConfig.frameHeight,
            traceRespQueueDepth = GlobalConfig.pixelQueueDepth
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
