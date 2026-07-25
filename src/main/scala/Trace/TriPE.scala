package Trace

import chisel3._
import chisel3.util._
import raytrace_utils._
import raytrace_utils.fudian._

class TriPE(val c: TriPeConfig) extends Module {
  require(c.numPEs == 1, s"TriPE currently has one triangle intersector per worker, got ${c.numPEs}")

  private val ctxCount = 2
  private val ctxW = 1
  private val memTagW = ctxW + 1
  private val lineSelW = math.max(1, log2Ceil(c.cacheLineTriangles))

  val io = IO(new Bundle {
    val ray_in = Input(new Ray(c.cfg))
    val ray_meta = Input(new RayMeta(c.addrWidth))
    val ray_ctx = Input(UInt(ctxW.W))
    val ray_valid = Input(Bool())

    val tri_batch_in = Input(new TriBatch(c.addrWidth))
    val tri_batch_ctx = Input(UInt(ctxW.W))
    val tri_batch_valid = Input(Bool())
    val dda_done = Input(Vec(ctxCount, Bool()))
    val clear_ctx = Input(Vec(ctxCount, Bool()))

    val mem_req = Decoupled(new TriMemReq(c, memTagW))
    val mem_resp = Flipped(Decoupled(new TriMemResp(c, memTagW)))

    val start_ready = Output(Vec(ctxCount, Bool()))
    val output_ready = Output(Vec(ctxCount, Bool()))
    val out_ctx = Output(UInt(ctxW.W))
    val out_best_hit = Output(Bool())
    val hit_id = Output(UInt(c.addrWidth.W))
    val t_best = Output(UInt(c.cfg.totalWidth.W))
    val out_meta = Output(new RayMeta(c.addrWidth))
    val out_done = Output(Bool())
    val out_early_stop = Output(Bool())
  })

  private val missT = 0x7F7FFFFF.U(c.cfg.totalWidth.W)
  private val rayReg = Reg(Vec(ctxCount, new Ray(c.cfg)))
  private val rayMetaReg = Reg(Vec(ctxCount, new RayMeta(c.addrWidth)))
  private val ctxBusy = RegInit(VecInit(Seq.fill(ctxCount)(false.B)))
  private val activeEpoch = RegInit(VecInit(Seq.fill(ctxCount)(false.B)))

  private val batchTriIdx = Reg(Vec(ctxCount, UInt(GlobalConfig.subgridMetaMemTriStartWidth.W)))
  private val batchTrisRemaining = Reg(Vec(ctxCount, UInt(GlobalConfig.subgridMetaMemTriCountWidth.W)))
  private val batchInProgress = RegInit(VecInit(Seq.fill(ctxCount)(false.B)))
  private val resultCapturePending = RegInit(VecInit(Seq.fill(ctxCount)(false.B)))

  private val bestT = Reg(Vec(ctxCount, UInt(c.cfg.totalWidth.W)))
  private val bestId = Reg(Vec(ctxCount, UInt(c.addrWidth.W)))
  private val hasHit = Reg(Vec(ctxCount, Bool()))
  private val hitStop = RegInit(VecInit(Seq.fill(ctxCount)(false.B)))
  private val triOutstanding = Reg(Vec(ctxCount, UInt(10.W)))

  private val resultHitReg = Reg(Vec(ctxCount, Bool()))
  private val resultIdReg = Reg(Vec(ctxCount, UInt(c.addrWidth.W)))
  private val resultTReg = Reg(Vec(ctxCount, UInt(c.cfg.totalWidth.W)))
  private val resultMetaReg = Reg(Vec(ctxCount, new RayMeta(c.addrWidth)))
  private val resultValidReg = RegInit(VecInit(Seq.fill(ctxCount)(false.B)))
  private val resultEarlyStopReg = RegInit(VecInit(Seq.fill(ctxCount)(false.B)))

  private def clearContext(idx: Int): Unit = {
    ctxBusy(idx) := false.B
    batchTriIdx(idx) := 0.U
    batchTrisRemaining(idx) := 0.U
    batchInProgress(idx) := false.B
    resultCapturePending(idx) := false.B
    bestT(idx) := missT
    bestId(idx) := 0.U
    hasHit(idx) := false.B
    hitStop(idx) := false.B
    triOutstanding(idx) := 0.U
    resultHitReg(idx) := false.B
    resultIdReg(idx) := 0.U
    resultTReg(idx) := missT
    resultMetaReg(idx) := 0.U.asTypeOf(new RayMeta(c.addrWidth))
    resultValidReg(idx) := false.B
    resultEarlyStopReg(idx) := false.B
    activeEpoch(idx) := !activeEpoch(idx)
  }

  private val outSel = Mux(resultValidReg(0), 0.U(ctxW.W), 1.U(ctxW.W))

  for (ctx <- 0 until ctxCount) {
    when(io.clear_ctx(ctx)) {
      clearContext(ctx)
    }
  }

  for (ctx <- 0 until ctxCount) {
    io.start_ready(ctx) := !ctxBusy(ctx) && !resultValidReg(ctx)
    io.output_ready(ctx) := ctxBusy(ctx) &&
      !resultCapturePending(ctx) &&
      !resultValidReg(ctx) &&
      !hitStop(ctx) &&
      !batchInProgress(ctx)
  }

  when(io.ray_valid && io.start_ready(io.ray_ctx)) {
    ctxBusy(io.ray_ctx) := true.B
    rayReg(io.ray_ctx) := io.ray_in
    rayMetaReg(io.ray_ctx) := io.ray_meta
    batchTriIdx(io.ray_ctx) := 0.U
    batchTrisRemaining(io.ray_ctx) := 0.U
    batchInProgress(io.ray_ctx) := false.B
    resultCapturePending(io.ray_ctx) := false.B
    triOutstanding(io.ray_ctx) := 0.U
    bestT(io.ray_ctx) := missT
    bestId(io.ray_ctx) := 0.U
    hasHit(io.ray_ctx) := false.B
    hitStop(io.ray_ctx) := false.B
    resultValidReg(io.ray_ctx) := false.B
    resultEarlyStopReg(io.ray_ctx) := false.B
  }

  for (ctx <- 0 until ctxCount) {
    val batchToCtx = io.tri_batch_valid && io.tri_batch_ctx === ctx.U && io.output_ready(ctx)
    when(batchToCtx) {
      assert(io.tri_batch_in.count =/= 0.U, "TriPE received an empty triangle batch")
      batchTriIdx(ctx) := io.tri_batch_in.base_addr
      batchTrisRemaining(ctx) := io.tri_batch_in.count
      batchInProgress(ctx) := true.B
    }
  }

  class MemReq extends Bundle {
    val blockAddr = UInt(GlobalConfig.triMemAddrWidth.W)
    val mask = UInt(c.cacheLineTriangles.W)
    val count = UInt(log2Ceil(c.cacheLineTriangles + 1).W)
    val ctx = UInt(ctxW.W)
    val epoch = Bool()
  }

  class PeHit extends Bundle {
    val ctx = UInt(ctxW.W)
    val epoch = Bool()
    val id = UInt(c.addrWidth.W)
    val t = UInt(c.cfg.totalWidth.W)
  }

  private val memCanEnq = Wire(Vec(ctxCount, Bool()))
  for (ctx <- 0 until ctxCount) {
    memCanEnq(ctx) := batchInProgress(ctx) && batchTrisRemaining(ctx) =/= 0.U
  }

  private val memRr = RegInit(false.B)
  private val memSel = Wire(UInt(ctxW.W))
  memSel := Mux(memCanEnq(0) && (!memRr || !memCanEnq(1)), 0.U, 1.U)

  private val laneBase = batchTriIdx(memSel)(lineSelW - 1, 0)
  private val blockAddr = batchTriIdx(memSel) >> lineSelW
  private val laneMask = Wire(Vec(c.cacheLineTriangles, Bool()))
  for (lane <- 0 until c.cacheLineTriangles) {
    laneMask(lane) := lane.U >= laneBase && ((lane.U - laneBase) < batchTrisRemaining(memSel))
  }
  private val blockMask = laneMask.asUInt
  private val blockIssueCount = PopCount(laneMask)

  val memReqQ = Module(new Queue(new MemReq, 4))
  memReqQ.io.enq.valid := memCanEnq.asUInt.orR
  memReqQ.io.enq.bits.blockAddr := blockAddr
  memReqQ.io.enq.bits.mask := blockMask
  memReqQ.io.enq.bits.count := blockIssueCount
  memReqQ.io.enq.bits.ctx := memSel
  memReqQ.io.enq.bits.epoch := activeEpoch(memSel)

  when(memReqQ.io.enq.fire) {
    batchTriIdx(memSel) := batchTriIdx(memSel) + blockIssueCount
    batchTrisRemaining(memSel) := batchTrisRemaining(memSel) - blockIssueCount
    memRr := !memSel(0)
  }

  val memReqStale = memReqQ.io.deq.valid &&
    ((memReqQ.io.deq.bits.epoch =/= activeEpoch(memReqQ.io.deq.bits.ctx)) ||
      io.clear_ctx(memReqQ.io.deq.bits.ctx))
  val memReqToMemValid = memReqQ.io.deq.valid && !memReqStale
  val memReqToMemReady = io.mem_req.ready
  io.mem_req.valid := memReqToMemValid
  io.mem_req.bits.addr := memReqQ.io.deq.bits.blockAddr
  io.mem_req.bits.mask := memReqQ.io.deq.bits.mask
  io.mem_req.bits.tag := Cat(memReqQ.io.deq.bits.epoch, memReqQ.io.deq.bits.ctx)
  memReqQ.io.deq.ready := memReqStale || (memReqToMemValid && memReqToMemReady)

  private val lineValid = RegInit(false.B)
  private val lineBlock = Reg(new TriangleBlock(c))
  private val lineCtx = Reg(UInt(ctxW.W))
  private val lineEpoch = Reg(Bool())
  private val lineLane = RegInit(0.U(lineSelW.W))
  private val lineLive = lineValid &&
    lineEpoch === activeEpoch(lineCtx) &&
    !io.clear_ctx(lineCtx)
  io.mem_resp.ready := !lineValid
  when(io.mem_resp.fire) {
    lineValid := true.B
    lineBlock := io.mem_resp.bits.block
    lineCtx := io.mem_resp.bits.tag(0)
    lineEpoch := io.mem_resp.bits.tag(1)
    lineLane := 0.U
  }.elsewhen(lineValid && !lineLive) {
    lineValid := false.B
  }.elsewhen(lineLive) {
    when(lineLane === (c.cacheLineTriangles - 1).U) {
      lineValid := false.B
      lineLane := 0.U
    }.otherwise {
      lineLane := lineLane + 1.U
    }
  }

  private val pes = Seq(Module(new RayTriangleIntersection(c.cfg)))
  pes.head.io.ray := rayReg(lineCtx)
  pes.head.io.tri := lineBlock.tris(lineLane)
  pes.head.io.in_valid := lineLive && lineBlock.mask(lineLane)

  private val peLatency = RayTriangleIntersection.totalLatency(c.cfg)
  // The triangle intersector is a fixed-latency pipeline. ctx/epoch must advance
  // every cycle with the same delay; compacting only on memRespLive breaks
  // alignment as soon as cache misses introduce bubbles between responses.
  private val peCtxOut = PipeUtils.pipeData(lineCtx, peLatency)
  private val peEpochOut = PipeUtils.pipeData(lineEpoch, peLatency)
  private val peOutLive = VecInit(Seq(
    pes.head.io.out_valid && peEpochOut === activeEpoch(peCtxOut)
  ))

  private val hitQs = Seq.fill(c.numPEs)(Module(new Queue(new PeHit, 4)))
  for (lane <- 0 until c.numPEs) {
    hitQs(lane).io.enq.valid := peOutLive(lane) && pes(lane).io.hit
    hitQs(lane).io.enq.bits.ctx := peCtxOut
    hitQs(lane).io.enq.bits.epoch := peEpochOut
    hitQs(lane).io.enq.bits.id := pes(lane).io.id
    hitQs(lane).io.enq.bits.t := pes(lane).io.t
    when(hitQs(lane).io.enq.valid) {
      assert(hitQs(lane).io.enq.ready, s"TriPE hit update queue overflow on lane $lane")
    }
  }

  val hitArb = Module(new RRArbiter(new PeHit, c.numPEs))
  for (lane <- 0 until c.numPEs) {
    hitArb.io.in(lane) <> hitQs(lane).io.deq
  }
  hitArb.io.out.ready := true.B

  for (ctx <- 0 until ctxCount) {
    // batchTrisRemaining advances when a request enters memReqQ, so the
    // outstanding count must use that same boundary. Counting only memory
    // issues lets a backpressured queued request escape the batch-done check.
    val triEnqueued = Mux(memReqQ.io.enq.fire && memReqQ.io.enq.bits.ctx === ctx.U,
      memReqQ.io.enq.bits.count, 0.U)
    val triRetired = PopCount(VecInit((0 until c.numPEs).map(lane =>
      peOutLive(lane) && peCtxOut === ctx.U
    )))
    val nextOutstanding = triOutstanding(ctx) +& triEnqueued
    when(io.clear_ctx(ctx)) {
      triOutstanding(ctx) := 0.U
    }.otherwise {
      assert(nextOutstanding >= triRetired, "TriPE triangle outstanding counter underflow")
      assert(nextOutstanding <= 1023.U, "TriPE triangle outstanding counter overflow")
      triOutstanding(ctx) := nextOutstanding - triRetired
    }
  }

  // Intersections marked as hits always have positive t. For non-negative
  // IEEE-754 values, unsigned bit ordering is numeric ordering, allowing the
  // feedback minimum to accept consecutive hits without a stale pipeline read.
  when(hitArb.io.out.fire) {
    val hit = hitArb.io.out.bits
    when(hit.epoch === activeEpoch(hit.ctx) && !io.clear_ctx(hit.ctx)) {
      when(SimpleFPCompare.nonNegLt(hit.t, bestT(hit.ctx)) || !hasHit(hit.ctx)) {
        bestT(hit.ctx) := hit.t
        bestId(hit.ctx) := hit.id
        hasHit(hit.ctx) := true.B
      }
    }
  }

  for (ctx <- 0 until ctxCount) {
    val hitUpdatesDrained = hitQs.map(q => !q.io.deq.valid).reduce(_ && _) && !hitArb.io.out.valid
    val batchDoneNow = batchInProgress(ctx) &&
      batchTrisRemaining(ctx) === 0.U &&
      triOutstanding(ctx) === 0.U &&
      hitUpdatesDrained
    val resultDoneNow =
      (hitStop(ctx) || io.dda_done(ctx)) &&
        !batchInProgress(ctx) &&
        triOutstanding(ctx) === 0.U &&
        hitUpdatesDrained

    when(resultValidReg(ctx) && outSel === ctx.U && !io.clear_ctx(ctx)) {
      resultValidReg(ctx) := false.B
    }

    when(batchDoneNow && !io.clear_ctx(ctx)) {
      batchInProgress(ctx) := false.B
      when(hasHit(ctx)) {
        hitStop(ctx) := true.B
      }
    }

    when(resultDoneNow && !resultCapturePending(ctx) && !resultValidReg(ctx) && !io.clear_ctx(ctx)) {
      resultCapturePending(ctx) := true.B
    }

    when(resultCapturePending(ctx) && !resultValidReg(ctx) && !io.clear_ctx(ctx)) {
      resultCapturePending(ctx) := false.B
      resultHitReg(ctx) := hasHit(ctx)
      resultIdReg(ctx) := bestId(ctx)
      resultTReg(ctx) := bestT(ctx)
      resultMetaReg(ctx) := rayMetaReg(ctx)
      resultEarlyStopReg(ctx) := hitStop(ctx)
      resultValidReg(ctx) := true.B
    }
  }

  io.out_done := resultValidReg.asUInt.orR
  io.out_ctx := outSel
  io.out_best_hit := resultHitReg(outSel)
  io.hit_id := resultIdReg(outSel)
  io.t_best := resultTReg(outSel)
  io.out_meta := resultMetaReg(outSel)
  io.out_early_stop := resultEarlyStopReg(outSel)
}
