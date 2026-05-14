package Trace

import chisel3._
import chisel3.util._
import raytrace_utils._
import raytrace_utils.fudian._

class TriPE(val c: TriPeConfig) extends Module {
  require(c.numPEs == 1, s"TriPE two-level fetch currently requires numPEs=1, got ${c.numPEs}")

  private val ctxCount = 2
  private val ctxW = 1
  private val triIdWidth = GlobalConfig.triRefIdWidth
  private val refPackFactor = GlobalConfig.triRefPackFactor
  private val refPackShift = log2Ceil(refPackFactor)
  private val refCountWidth = log2Ceil(refPackFactor + 1)
  private val refPrefetchLowWatermark = math.max(1, refPackFactor / 4)

  val io = IO(new Bundle {
    val ray_in: Ray = Input(new Ray(c.cfg))
    val ray_meta: RayMeta = Input(new RayMeta(c.addrWidth))
    val ray_ctx = Input(UInt(ctxW.W))
    val ray_valid: Bool = Input(Bool())

    val tri_batch_in: TriBatch = Input(new TriBatch(c.addrWidth))
    val tri_batch_ctx = Input(UInt(ctxW.W))
    val tri_batch_valid: Bool = Input(Bool())
    val end_exec: Bool = Input(Bool())
    val clear_ctx = Input(Vec(ctxCount, Bool()))

    val ref_mem_req = Decoupled(UInt(c.addrWidth.W))
    val ref_mem_resp = Flipped(Decoupled(UInt(GlobalConfig.triRefMemDataWidth.W)))

    val mem_req: DecoupledIO[UInt] = Decoupled(UInt(c.addrWidth.W))
    val mem_req_mask: DecoupledIO[UInt] = Decoupled(UInt(c.numPEs.W))
    val mem_resp: DecoupledIO[TriangleBlock] = Flipped(Decoupled(new TriangleBlock(c)))

    val start_ready = Output(Vec(ctxCount, Bool()))
    val output_ready = Output(Vec(ctxCount, Bool()))
    val out_ctx: UInt = Output(UInt(ctxW.W))
    val out_best_hit: Bool = Output(Bool())
    val hit_id: UInt = Output(UInt(c.addrWidth.W))
    val t_best: UInt = Output(UInt(c.cfg.totalWidth.W))
    val out_meta: RayMeta = Output(new RayMeta(c.addrWidth))
    val out_done: Bool = Output(Bool())
  })

  private val missT = 0x7F7FFFFF.U(c.cfg.totalWidth.W)
  private val rayReg = Reg(Vec(ctxCount, new Ray(c.cfg)))
  private val rayMetaReg = Reg(Vec(ctxCount, new RayMeta(c.addrWidth)))
  private val ctxBusy = RegInit(VecInit(Seq.fill(ctxCount)(false.B)))
  private val noMoreBatches = RegInit(VecInit(Seq.fill(ctxCount)(false.B)))
  private val activeEpoch = RegInit(VecInit(Seq.fill(ctxCount)(false.B)))

  private val batchRefIdx = RegInit(VecInit(Seq.fill(ctxCount)(0.U(c.addrWidth.W))))
  private val batchRefsRemaining = RegInit(VecInit(Seq.fill(ctxCount)(0.U(16.W))))
  private val batchInProgress = RegInit(VecInit(Seq.fill(ctxCount)(false.B)))
  private val resultCapturePending = RegInit(VecInit(Seq.fill(ctxCount)(false.B)))

  private val refBufferA = Reg(Vec(ctxCount, Vec(refPackFactor, UInt(triIdWidth.W))))
  private val refBufferB = Reg(Vec(ctxCount, Vec(refPackFactor, UInt(triIdWidth.W))))
  private val refBufIdxA = RegInit(VecInit(Seq.fill(ctxCount)(0.U(refPackShift.W))))
  private val refBufIdxB = RegInit(VecInit(Seq.fill(ctxCount)(0.U(refPackShift.W))))
  private val refBufCountA = RegInit(VecInit(Seq.fill(ctxCount)(0.U(refCountWidth.W))))
  private val refBufCountB = RegInit(VecInit(Seq.fill(ctxCount)(0.U(refCountWidth.W))))

  private val bestT = RegInit(VecInit(Seq.fill(ctxCount)(missT)))
  private val bestId = RegInit(VecInit(Seq.fill(ctxCount)(0.U(c.addrWidth.W))))
  private val hasHit = RegInit(VecInit(Seq.fill(ctxCount)(false.B)))
  private val inflightCnt = RegInit(VecInit(Seq.fill(ctxCount)(0.U(10.W))))
  private val memReqQueuedCnt = RegInit(VecInit(Seq.fill(ctxCount)(0.U(4.W))))

  private val resultHitReg = RegInit(VecInit(Seq.fill(ctxCount)(false.B)))
  private val resultIdReg = RegInit(VecInit(Seq.fill(ctxCount)(0.U(c.addrWidth.W))))
  private val resultTReg = RegInit(VecInit(Seq.fill(ctxCount)(missT)))
  private val resultMetaReg = Reg(Vec(ctxCount, new RayMeta(c.addrWidth)))
  private val resultValidReg = RegInit(VecInit(Seq.fill(ctxCount)(false.B)))

  private def clearContext(idx: Int): Unit = {
    ctxBusy(idx) := false.B
    noMoreBatches(idx) := false.B
    batchRefIdx(idx) := 0.U
    batchRefsRemaining(idx) := 0.U
    batchInProgress(idx) := false.B
    resultCapturePending(idx) := false.B
    refBufIdxA(idx) := 0.U
    refBufIdxB(idx) := 0.U
    refBufCountA(idx) := 0.U
    refBufCountB(idx) := 0.U
    bestT(idx) := missT
    bestId(idx) := 0.U
    hasHit(idx) := false.B
    inflightCnt(idx) := 0.U
    memReqQueuedCnt(idx) := 0.U
    resultHitReg(idx) := false.B
    resultIdReg(idx) := 0.U
    resultTReg(idx) := missT
    resultMetaReg(idx) := 0.U.asTypeOf(new RayMeta(c.addrWidth))
    resultValidReg(idx) := false.B
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
      !batchInProgress(ctx) &&
      !resultCapturePending(ctx) &&
      !resultValidReg(ctx)
  }

  when(io.ray_valid && io.start_ready(io.ray_ctx)) {
    ctxBusy(io.ray_ctx) := true.B
    rayReg(io.ray_ctx) := io.ray_in
    rayMetaReg(io.ray_ctx) := io.ray_meta
    noMoreBatches(io.ray_ctx) := false.B
    resultValidReg(io.ray_ctx) := false.B
  }

  when(io.tri_batch_valid && io.output_ready(io.tri_batch_ctx)) {
    val ctx = io.tri_batch_ctx
    batchRefIdx(ctx) := io.tri_batch_in.base_addr
    batchRefsRemaining(ctx) := io.tri_batch_in.count
    batchInProgress(ctx) := true.B
    noMoreBatches(ctx) := io.end_exec
    refBufIdxA(ctx) := 0.U
    refBufIdxB(ctx) := 0.U
    refBufCountA(ctx) := 0.U
    refBufCountB(ctx) := 0.U
    bestT(ctx) := missT
    bestId(ctx) := 0.U
    hasHit(ctx) := false.B
  }

  private val refReqPipeLen = GlobalConfig.triRefMemDpiLatency
  private val refReqLivePipe = RegInit(VecInit(Seq.fill(refReqPipeLen)(false.B)))
  private val refReqCtxPipe = RegInit(VecInit(Seq.fill(refReqPipeLen)(0.U(ctxW.W))))
  private val refReqEpochPipe = RegInit(VecInit(Seq.fill(refReqPipeLen)(false.B)))
  private val refReqOffsetPipe = RegInit(VecInit(Seq.fill(refReqPipeLen)(0.U(refPackShift.W))))
  private val refReqCountPipe = RegInit(VecInit(Seq.fill(refReqPipeLen)(0.U(refCountWidth.W))))
  private val refReqTargetBPipe = RegInit(VecInit(Seq.fill(refReqPipeLen)(false.B)))

  private val refPending = Wire(Vec(ctxCount, Bool()))
  for (ctx <- 0 until ctxCount) {
    refPending(ctx) := refReqLivePipe.zip(refReqCtxPipe).map { case (v, c) => v && c === ctx.U }.reduce(_ || _)
  }

  private val activeBufSel = Wire(Vec(ctxCount, Bool()))
  private val activeBufIdx = Wire(Vec(ctxCount, UInt(refPackShift.W)))
  private val activeBufCount = Wire(Vec(ctxCount, UInt(refCountWidth.W)))
  private val sourceHasData = Wire(Vec(ctxCount, Bool()))
  private val prefetchBufEmpty = Wire(Vec(ctxCount, Bool()))
  private val requestToBufB = Wire(Vec(ctxCount, Bool()))
  private val refWordAddr = Wire(Vec(ctxCount, UInt(c.addrWidth.W)))
  private val refWordOff = Wire(Vec(ctxCount, UInt(refPackShift.W)))
  private val refChunkCount = Wire(Vec(ctxCount, UInt(refCountWidth.W)))
  private val refCanReq = Wire(Vec(ctxCount, Bool()))
  private val memCanEnq = Wire(Vec(ctxCount, Bool()))
  private val geomTriId = Wire(Vec(ctxCount, UInt(triIdWidth.W)))

  for (ctx <- 0 until ctxCount) {
    activeBufSel(ctx) := refBufCountA(ctx) === 0.U && refBufCountB(ctx) =/= 0.U
    activeBufIdx(ctx) := Mux(activeBufSel(ctx), refBufIdxB(ctx), refBufIdxA(ctx))
    activeBufCount(ctx) := Mux(activeBufSel(ctx), refBufCountB(ctx), refBufCountA(ctx))
    sourceHasData(ctx) := refBufCountA(ctx) =/= 0.U || refBufCountB(ctx) =/= 0.U
    prefetchBufEmpty(ctx) := Mux(activeBufSel(ctx), refBufCountA(ctx) === 0.U, refBufCountB(ctx) === 0.U)
    requestToBufB(ctx) := Mux(sourceHasData(ctx), !activeBufSel(ctx), false.B)
    refWordAddr(ctx) := batchRefIdx(ctx) >> refPackShift
    refWordOff(ctx) := batchRefIdx(ctx)(refPackShift - 1, 0)
    val refChunkAvailWide = refPackFactor.U(16.W) - refWordOff(ctx)
    val refChunkCountWide = Mux(batchRefsRemaining(ctx) < refChunkAvailWide, batchRefsRemaining(ctx), refChunkAvailWide)
    refChunkCount(ctx) := refChunkCountWide(refCountWidth - 1, 0)
    val needInitialFill = !sourceHasData(ctx)
    val needPrefetch = sourceHasData(ctx) && prefetchBufEmpty(ctx) && (activeBufCount(ctx) <= refPrefetchLowWatermark.U)
    refCanReq(ctx) := batchInProgress(ctx) &&
      batchRefsRemaining(ctx) =/= 0.U &&
      !refPending(ctx) &&
      (needInitialFill || needPrefetch)
    geomTriId(ctx) := Mux(activeBufSel(ctx), refBufferB(ctx)(activeBufIdx(ctx)), refBufferA(ctx)(activeBufIdx(ctx)))
    memCanEnq(ctx) := batchInProgress(ctx) && sourceHasData(ctx)
  }

  private val refRr = RegInit(false.B)
  private val refSel = Wire(UInt(ctxW.W))
  private val refReqValid = refCanReq.asUInt.orR
  refSel := Mux(refCanReq(0) && (!refRr || !refCanReq(1)), 0.U, 1.U)
  io.ref_mem_req.valid := refReqValid
  io.ref_mem_req.bits := refWordAddr(refSel)

  when(io.ref_mem_req.fire) {
    batchRefIdx(refSel) := batchRefIdx(refSel) + refChunkCount(refSel)
    batchRefsRemaining(refSel) := batchRefsRemaining(refSel) - refChunkCount(refSel)
    refRr := !refSel(0)
  }

  io.ref_mem_resp.ready := true.B
  refReqLivePipe(0) := io.ref_mem_req.fire
  refReqCtxPipe(0) := Mux(io.ref_mem_req.fire, refSel, 0.U)
  refReqEpochPipe(0) := Mux(io.ref_mem_req.fire, activeEpoch(refSel), false.B)
  refReqOffsetPipe(0) := Mux(io.ref_mem_req.fire, refWordOff(refSel), 0.U)
  refReqCountPipe(0) := Mux(io.ref_mem_req.fire, refChunkCount(refSel), 0.U)
  refReqTargetBPipe(0) := Mux(io.ref_mem_req.fire, requestToBufB(refSel), false.B)
  for (i <- 1 until refReqPipeLen) {
    refReqLivePipe(i) := refReqLivePipe(i - 1)
    refReqCtxPipe(i) := refReqCtxPipe(i - 1)
    refReqEpochPipe(i) := refReqEpochPipe(i - 1)
    refReqOffsetPipe(i) := refReqOffsetPipe(i - 1)
    refReqCountPipe(i) := refReqCountPipe(i - 1)
    refReqTargetBPipe(i) := refReqTargetBPipe(i - 1)
  }

  private val refRespCtx = refReqCtxPipe.last
  private val refRespLive =
    io.ref_mem_resp.fire &&
      refReqLivePipe.last &&
      (refReqEpochPipe.last === activeEpoch(refRespCtx)) &&
      !io.clear_ctx(refRespCtx)

  private val refWordIds = Wire(Vec(refPackFactor, UInt(triIdWidth.W)))
  for (i <- 0 until refPackFactor) {
    refWordIds(i) := io.ref_mem_resp.bits((i + 1) * triIdWidth - 1, i * triIdWidth)
  }
  private val compactedIds = Wire(Vec(refPackFactor, UInt(triIdWidth.W)))
  for (i <- 0 until refPackFactor) {
    compactedIds(i) := 0.U
    for (off <- 0 until refPackFactor) {
      if (off + i < refPackFactor) {
        when(refReqOffsetPipe.last === off.U) {
          compactedIds(i) := refWordIds(off + i)
        }
      }
    }
  }

  when(refRespLive) {
    when(refReqTargetBPipe.last) {
      for (i <- 0 until refPackFactor) {
        refBufferB(refRespCtx)(i) := compactedIds(i)
      }
      refBufIdxB(refRespCtx) := 0.U
      refBufCountB(refRespCtx) := refReqCountPipe.last
    }.otherwise {
      for (i <- 0 until refPackFactor) {
        refBufferA(refRespCtx)(i) := compactedIds(i)
      }
      refBufIdxA(refRespCtx) := 0.U
      refBufCountA(refRespCtx) := refReqCountPipe.last
    }
  }

  class MemReq extends Bundle {
    val addr = UInt(GlobalConfig.triMemAddrWidth.W)
    val ctx = UInt(ctxW.W)
    val epoch = Bool()
  }

  val memReqQ = Module(new Queue(new MemReq, 4))
  private val memRr = RegInit(false.B)
  private val memSel = Wire(UInt(ctxW.W))
  private val memEnqValid = memCanEnq.asUInt.orR
  memSel := Mux(memCanEnq(0) && (!memRr || !memCanEnq(1)), 0.U, 1.U)

  memReqQ.io.enq.valid := memEnqValid
  memReqQ.io.enq.bits.addr := geomTriId(memSel)
  memReqQ.io.enq.bits.ctx := memSel
  memReqQ.io.enq.bits.epoch := activeEpoch(memSel)

  when(memReqQ.io.enq.fire) {
    when(activeBufSel(memSel)) {
      refBufIdxB(memSel) := refBufIdxB(memSel) + 1.U
      refBufCountB(memSel) := refBufCountB(memSel) - 1.U
    }.otherwise {
      refBufIdxA(memSel) := refBufIdxA(memSel) + 1.U
      refBufCountA(memSel) := refBufCountA(memSel) - 1.U
    }
    memRr := !memSel(0)
  }

  val memReqStale = memReqQ.io.deq.valid &&
    ((memReqQ.io.deq.bits.epoch =/= activeEpoch(memReqQ.io.deq.bits.ctx)) || io.clear_ctx(memReqQ.io.deq.bits.ctx))
  val memReqToMemValid = memReqQ.io.deq.valid && !memReqStale
  val memReqToMemReady = io.mem_req.ready && io.mem_req_mask.ready
  val memReqDequeued = memReqQ.io.deq.fire

  io.mem_req.valid := memReqToMemValid
  io.mem_req.bits := memReqQ.io.deq.bits.addr
  io.mem_req_mask.valid := memReqToMemValid
  io.mem_req_mask.bits := 1.U
  memReqQ.io.deq.ready := memReqStale || (memReqToMemValid && memReqToMemReady)

  for (ctx <- 0 until ctxCount) {
    val inc = memReqQ.io.enq.fire && memSel === ctx.U
    val dec = memReqDequeued && memReqQ.io.deq.bits.ctx === ctx.U
    when(io.clear_ctx(ctx)) {
      memReqQueuedCnt(ctx) := 0.U
    }.elsewhen(inc || dec) {
      memReqQueuedCnt(ctx) := memReqQueuedCnt(ctx) + inc.asUInt - dec.asUInt
    }
  }

  io.mem_resp.ready := true.B

  private val memReqPipeLen = GlobalConfig.triMemDpiLatency
  private val memReqLivePipe = RegInit(VecInit(Seq.fill(memReqPipeLen)(false.B)))
  private val memReqCtxPipe = RegInit(VecInit(Seq.fill(memReqPipeLen)(0.U(ctxW.W))))
  private val memReqEpochPipe = RegInit(VecInit(Seq.fill(memReqPipeLen)(false.B)))
  private val memReqIssued = memReqQ.io.deq.fire && !memReqStale
  memReqLivePipe(0) := memReqIssued
  memReqCtxPipe(0) := Mux(memReqIssued, memReqQ.io.deq.bits.ctx, 0.U)
  memReqEpochPipe(0) := Mux(memReqIssued, memReqQ.io.deq.bits.epoch, false.B)
  for (i <- 1 until memReqPipeLen) {
    memReqLivePipe(i) := memReqLivePipe(i - 1)
    memReqCtxPipe(i) := memReqCtxPipe(i - 1)
    memReqEpochPipe(i) := memReqEpochPipe(i - 1)
  }

  private val memLiveForCtx = Wire(Vec(ctxCount, Bool()))
  for (ctx <- 0 until ctxCount) {
    memLiveForCtx(ctx) := memReqLivePipe.zip(memReqCtxPipe).map { case (v, c) => v && c === ctx.U }.reduce(_ || _)
  }

  private val memRespCtx = memReqCtxPipe.last
  private val memRespLive =
    io.mem_resp.fire &&
      memReqLivePipe.last &&
      (memReqEpochPipe.last === activeEpoch(memRespCtx)) &&
      !io.clear_ctx(memRespCtx)

  private val pe = Module(new RayTriangleIntersection(c.cfg))
  pe.io.ray := rayReg(memRespCtx)
  pe.io.tri := io.mem_resp.bits.tris(0)
  pe.io.in_valid := memRespLive && io.mem_resp.bits.mask(0)

  private val peLatency =
    c.cfg.faddLatency + (c.cfg.fmulLatency + c.cfg.faddLatency) + (c.cfg.fmulLatency + c.cfg.faddLatency + c.cfg.faddLatency) +
      math.max(c.cfg.fdivLatency, c.cfg.fmulLatency + c.cfg.faddLatency + c.cfg.faddLatency) +
      c.cfg.fmulLatency + c.cfg.faddLatency
  private val peCtxOut = PipeUtils.pipeData(memRespCtx, peLatency)

  val fcmp = Module(new FCMP(c.cfg))
  fcmp.io.a := pe.io.t
  fcmp.io.b := bestT(peCtxOut)
  fcmp.io.signaling := false.B

  for (ctx <- 0 until ctxCount) {
    val incoming = memRespLive && io.mem_resp.bits.mask(0) && memRespCtx === ctx.U
    val outgoing = pe.io.out_valid && peCtxOut === ctx.U
    when(io.clear_ctx(ctx)) {
      inflightCnt(ctx) := 0.U
    }.otherwise {
      inflightCnt(ctx) := inflightCnt(ctx) + incoming.asUInt - outgoing.asUInt
    }
  }

  when(pe.io.out_valid && pe.io.hit) {
    when(fcmp.io.lt || !hasHit(peCtxOut)) {
      bestT(peCtxOut) := pe.io.t
      bestId(peCtxOut) := pe.io.id
      hasHit(peCtxOut) := true.B
    }
  }

  for (ctx <- 0 until ctxCount) {
    val incoming = memRespLive && io.mem_resp.bits.mask(0) && memRespCtx === ctx.U
    val outgoing = pe.io.out_valid && peCtxOut === ctx.U
    val inflightNext = inflightCnt(ctx) + incoming.asUInt - outgoing.asUInt
    val batchSourceDrained =
      batchRefsRemaining(ctx) === 0.U &&
        refBufCountA(ctx) === 0.U &&
        refBufCountB(ctx) === 0.U &&
        !refPending(ctx) &&
        memReqQueuedCnt(ctx) === 0.U &&
        !memLiveForCtx(ctx)
    val batchDoneNow = batchInProgress(ctx) && batchSourceDrained && inflightNext === 0.U

    when(resultValidReg(ctx) && outSel === ctx.U && !io.clear_ctx(ctx)) {
      resultValidReg(ctx) := false.B
    }

    when(batchDoneNow && !resultCapturePending(ctx) && !resultValidReg(ctx) && !io.clear_ctx(ctx)) {
      batchInProgress(ctx) := false.B
      resultCapturePending(ctx) := true.B
    }

    when(resultCapturePending(ctx) && !resultValidReg(ctx) && !io.clear_ctx(ctx)) {
      resultCapturePending(ctx) := false.B
      resultHitReg(ctx) := hasHit(ctx)
      resultIdReg(ctx) := bestId(ctx)
      resultTReg(ctx) := bestT(ctx)
      resultMetaReg(ctx) := rayMetaReg(ctx)
      resultValidReg(ctx) := true.B
    }
  }

  io.out_done := resultValidReg.asUInt.orR
  io.out_ctx := outSel
  io.out_best_hit := resultHitReg(outSel)
  io.hit_id := resultIdReg(outSel)
  io.t_best := resultTReg(outSel)
  io.out_meta := resultMetaReg(outSel)
}
