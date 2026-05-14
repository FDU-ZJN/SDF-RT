import chisel3._
import chisel3.util._
import raytrace_utils._
import raytrace_utils.fudian._
import DDA.DDA
import Render.{NormalMemWriteIO, RenderStage}
import SDF.{InitStage, SdfMemWriteIO, SdfStage, SetupUnit}
import Trace.TraceController

import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Paths}
import scala.io.Source

class FpgaTop(
               width:           Int = GlobalConfig.frameWidth,
               height:          Int = GlobalConfig.frameHeight,
               pixelQueueDepth: Int = GlobalConfig.pixelQueueDepth
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
    val setup_ready      = Output(Bool())

    val frame_start      = Input(Bool())

    val pixel_valid      = Output(UInt(2.W))
    val pixel_ready      = Input(Bool())
    val pixel_rgb8       = Output(UInt((24 * 2).W))
    val pixel_hit_id     = Output(UInt((c.addrWidth * 2).W))

    val frame_done       = Output(Bool())
    val busy             = Output(Bool())
    val frame_count      = Output(UInt(32.W))
    val validation_error = Output(Bool())
    val stall_detected   = Output(Bool())

    // SDF memory write port for PS initialization
    val sdf_mem_wr = Flipped(new SdfMemWriteIO)
    val normal_mem_wr = Flipped(new NormalMemWriteIO)
  })

  val rayDirCalcs = Seq.tabulate(2)(lane => Module(new RayDirCalc(c.cfg, width, height, laneId = lane, numLanes = 2)))
  val initStages = Seq.fill(2)(Module(new InitStage(c.cfg, c.addrWidth)))
  val setupUnit = Module(new SetupUnit(c.cfg, sdfCfg))
  val sdfStage = Module(new SdfStage(c.cfg, c.addrWidth))
  val ddaStage = Module(new DDA(c.cfg, c.addrWidth, globalRes = sdfCfg.DDAGlobalRes, subRes = sdfCfg.SubRes, maxTraversalSteps = sdfCfg.DDAMaxSteps))
  val traceController = Module(new TraceController(c, sdfCfg.DDAMaxSteps))
  val traceJobQueue = Module(new Queue(new DdaTraceJobDesc(c.cfg, c.addrWidth, sdfCfg.DDAMaxSteps), GlobalConfig.triBatchQueueDepth))
  val renderStage = Module(new RenderStage(c.cfg))
  val commitQueue = Module(new CommitQueue(c.cfg))
  val postHitQs = Seq.fill(2)(Module(new Queue(new RayIssue(c.cfg, c.addrWidth), postHitQueueDepth)))
  val postHitArb = Module(new RRArbiter(new RayIssue(c.cfg, c.addrWidth), 2))
  val postInitQs = Seq.fill(2)(Module(new Queue(new InitStageResp(c.cfg, c.addrWidth), postInitQueueDepth)))

  val setupReg        = RegInit(false.B)
  val setupOriginReg  = RegInit(0.U.asTypeOf(new Vec3(c.cfg)))
  val setupGridMinReg = RegInit(0.U.asTypeOf(new Vec3(c.cfg)))
  val setupGridMaxReg = RegInit(0.U.asTypeOf(new Vec3(c.cfg)))

  when(io.setup_valid && !setupReg) {
    setupOriginReg  := io.setup_origin
    setupGridMinReg := io.setup_grid_min
    setupGridMaxReg := io.setup_grid_max
    setupReg        := true.B
  }.elsewhen(setupUnit.io.setup_finish) {
    setupReg := false.B
  }

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
  traceController.io.cmd_write.valid := ddaStage.io.cmd_write.valid
  traceController.io.cmd_write.bits := ddaStage.io.cmd_write.bits
  traceJobQueue.io.enq <> ddaStage.io.trace_job_out
  when(traceJobQueue.io.enq.valid) {
    assert(traceJobQueue.io.enq.ready, "FpgaTop traceJobQueue overflow")
  }
  traceController.io.job_in <> traceJobQueue.io.deq
  renderStage.io.in <> traceController.io.result_out
  renderStage.io.normal_mem_wr <> io.normal_mem_wr

  val idle :: rendering :: frameComplete :: Nil = Enum(3)
  val state = RegInit(idle)

  val pixelCountW   = log2Ceil(width * height + 1)
  val totalPixels   = (width * height).U(pixelCountW.W)
  val frameCountReg = RegInit(0.U(32.W))
  io.frame_count := frameCountReg

  val frameStartReg   = RegNext(io.frame_start, false.B)
  val frameStartPulse = io.frame_start && !frameStartReg

  val frameDonePulse = WireInit(false.B)
  io.frame_done := frameDonePulse

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
    postHitArb.io.in(lane) <> postHitQs(lane).io.deq
    postInitQs(lane).io.deq.ready := postInitDispatch
    when(hitPush) {
      assert(postHitQs(lane).io.enq.ready, s"FpgaTop postHitQ lane $lane overflow")
    }
  }

  sdfStage.io.issue_in <> postHitArb.io.out

  val sdfOut = sdfStage.io.out.bits
  val sdfOutValid = sdfStage.io.out.valid
  val sdfOutHit = sdfOut.hit

  val sdfHitQ = Module(new Queue(new DdaTraversalReq(c.cfg, c.addrWidth), GlobalConfig.simSdfHitQueueDepth))
  sdfHitQ.io.enq.valid := sdfOutValid && sdfOutHit
  sdfHitQ.io.enq.bits.ray := sdfOut.ray
  sdfHitQ.io.enq.bits.meta := sdfOut.meta
  sdfHitQ.io.enq.bits.traceSlot := 0.U
  when(sdfHitQ.io.enq.valid) {
    assert(sdfHitQ.io.enq.ready, "FpgaTop sdfHitQ overflow")
  }
  ddaStage.io.in <> sdfHitQ.io.deq

  commitQueue.io.writeback.valid := renderStage.io.out.valid
  commitQueue.io.writeback.bits := renderStage.io.out.bits
  renderStage.io.out.ready := commitQueue.io.writeback.ready

  val wb2 = Wire(new RenderResult(c.cfg, c.addrWidth))
  wb2.meta := postInitQs(0).io.deq.bits.meta
  wb2.meta.slotId := commitQueue.io.allocSlot(0)
  wb2.hit := false.B
  wb2.hitId := 0.U
  wb2.rgb8 := Cat(0.U(8.W), 255.U(8.W), 0.U(8.W))

  val wb4 = Wire(new RenderResult(c.cfg, c.addrWidth))
  wb4.meta := postInitQs(1).io.deq.bits.meta
  wb4.meta.slotId := commitQueue.io.allocSlot(1)
  wb4.hit := false.B
  wb4.hitId := 0.U
  wb4.rgb8 := Cat(0.U(8.W), 255.U(8.W), 0.U(8.W))

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

  val wb3 = Wire(new RenderResult(c.cfg, c.addrWidth))
  wb3.meta := sdfOut.meta
  wb3.hit := false.B
  wb3.hitId := 0.U
  wb3.rgb8 := Cat(0.U(8.W), 0.U(8.W), 128.U(8.W))
  val sdfWriteback3ValidReg = RegInit(false.B)
  val sdfWriteback3BitsReg = Reg(new RenderResult(c.cfg, c.addrWidth))
  when(commitQueue.io.writeback3.ready) {
    sdfWriteback3ValidReg := false.B
  }
  when(sdfOutValid && !sdfOutHit) {
    assert(!sdfWriteback3ValidReg || commitQueue.io.writeback3.ready, "FpgaTop sdf miss writeback buffer overflow")
    sdfWriteback3ValidReg := true.B
    sdfWriteback3BitsReg := wb3
  }
  commitQueue.io.writeback3.valid := sdfWriteback3ValidReg
  commitQueue.io.writeback3.bits := sdfWriteback3BitsReg
  sdfStage.io.out.ready := true.B

  val issuedCount = RegInit(0.U(pixelCountW.W))
  val retiredCount = RegInit(0.U(pixelCountW.W))
  val totalPairs = (width * height / 2).U(log2Ceil(width * height / 2 + 1).W)
  val remainingPairsReg = RegInit(0.U(log2Ceil(width * height / 2 + 1).W))
  val rayIssueFire0 = WireDefault(false.B)
  val rayIssueFire1 = WireDefault(false.B)
  val rayIssueCount = WireDefault(0.U(2.W))
  val canLaunchPairReg = RegInit(false.B)
  val canLaunchPairNext = WireDefault(false.B)
  val issuePairFire = canLaunchPairReg && (state === rendering)

  val validationError = RegInit(false.B)   // Fix 8
  io.validation_error := validationError

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
        remainingPairsReg := totalPairs
        state := rendering
      }
    }
    is(rendering) {
      canLaunchPairReg := canLaunchPairNext
      when(rayIssueCount =/= 0.U) {
        issuedCount := issuedCount + rayIssueCount
      }
      when(issuePairFire && (remainingPairsReg =/= 0.U)) {
        remainingPairsReg := remainingPairsReg - 1.U
      }
      when(enqFire) {
        retiredCount := retiredCount + retiredPixelsThisBeat
      }
      when(enqFire && (retiredCount + retiredPixelsThisBeat >= totalPixels)) {
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

  io.busy := (state === rendering)

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
  }

  assert(rayPush0 === rayPush1, "[FpgaTop] BUG: rayDir lanes lost pair alignment")
  assert(!(rayPush0 && !setupReady), "[FpgaTop] BUG: direct rayDir output not accepted by init pipeline")
  assert((totalPixels(0) === 0.U), "[FpgaTop] pair-only issue path requires an even pixel count")

  initStages(0).io.in.bits.rd.x := rayDirCalcs(0).io.dir_x
  initStages(0).io.in.bits.rd.y := rayDirCalcs(0).io.dir_y
  initStages(0).io.in.bits.rd.z := rayDirCalcs(0).io.dir_z
  initStages(1).io.in.bits.rd.x := rayDirCalcs(1).io.dir_x
  initStages(1).io.in.bits.rd.y := rayDirCalcs(1).io.dir_y
  initStages(1).io.in.bits.rd.z := rayDirCalcs(1).io.dir_z


  class PixelBundle extends Bundle {
    val valid_mask = UInt(2.W)
    val rgb8   = UInt((24 * 2).W)
    val hit_id = UInt((c.addrWidth * 2).W)
  }

  val pixelQueue = Module(new Queue(new PixelBundle, pixelQueueDepth))

  // Fix 3: enqFire = 真实握手成功（valid && ready 同时为真）
  val commitOutValid = Cat(commitQueue.io.out(1).valid, commitQueue.io.out(0).valid)
  retiredPixelsThisBeat := PopCount(commitOutValid.asBools)
  enqFire := commitOutValid.orR && pixelQueue.io.enq.ready

  pixelQueue.io.enq.valid        := commitOutValid.orR
  pixelQueue.io.enq.bits.valid_mask := commitOutValid
  pixelQueue.io.enq.bits.rgb8 := Cat(
    commitQueue.io.out(1).bits.rgb8,
    commitQueue.io.out(0).bits.rgb8
  )
  pixelQueue.io.enq.bits.hit_id := Cat(
    commitQueue.io.out(1).bits.hitId,
    commitQueue.io.out(0).bits.hitId
  )
  pixelQueue.io.deq.ready       := io.pixel_ready

  assert(
    !(commitOutValid.orR && !pixelQueue.io.enq.ready),
    "[FpgaTop] BUG: pixelQueue full! Increase pixelQueueDepth or topMaxInflight"
  )

  io.pixel_valid  := Mux(pixelQueue.io.deq.valid, pixelQueue.io.deq.bits.valid_mask, 0.U)
  io.pixel_rgb8   := pixelQueue.io.deq.bits.rgb8
  io.pixel_hit_id := pixelQueue.io.deq.bits.hit_id


  val underflow = retiredCount > issuedCount
  val overflow  = issuedCount > totalPixels

  when(underflow || overflow) {
    validationError := true.B
  }

  val stallCounter = RegInit(0.U(16.W))
  val progressMade = rayIssueFire0 || rayIssueFire1 || enqFire

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
