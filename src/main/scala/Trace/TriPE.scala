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

  private val batchRefIdx = Reg(Vec(ctxCount, UInt(c.addrWidth.W)))
  private val batchRefsRemaining = Reg(Vec(ctxCount, UInt(16.W)))
  private val batchInProgress = RegInit(VecInit(Seq.fill(ctxCount)(false.B)))
  private val resultCapturePending = RegInit(VecInit(Seq.fill(ctxCount)(false.B)))

  private val refBuffer = Reg(Vec(ctxCount, Vec(refPackFactor, UInt(triIdWidth.W))))
  private val refBufIdx = Reg(Vec(ctxCount, UInt(refPackShift.W)))
  private val refBufCount = Reg(Vec(ctxCount, UInt(refCountWidth.W)))
  private val refInflight = RegInit(VecInit(Seq.fill(ctxCount)(false.B)))

  class RefWord extends Bundle {
    val ids = Vec(refPackFactor, UInt(triIdWidth.W))
    val count = UInt(refCountWidth.W)
  }
  private val refShadowQ = Seq.fill(ctxCount)(Module(new Queue(new RefWord, 1, hasFlush = true)))

  private val bestT = Reg(Vec(ctxCount, UInt(c.cfg.totalWidth.W)))
  private val bestId = Reg(Vec(ctxCount, UInt(c.addrWidth.W)))
  private val hasHit = Reg(Vec(ctxCount, Bool()))
  private val triOutstanding = Reg(Vec(ctxCount, UInt(10.W)))

  private val resultHitReg = Reg(Vec(ctxCount, Bool()))
  private val resultIdReg = Reg(Vec(ctxCount, UInt(c.addrWidth.W)))
  private val resultTReg = Reg(Vec(ctxCount, UInt(c.cfg.totalWidth.W)))
  private val resultMetaReg = Reg(Vec(ctxCount, new RayMeta(c.addrWidth)))
  private val resultValidReg = RegInit(VecInit(Seq.fill(ctxCount)(false.B)))

  private def clearContext(idx: Int): Unit = {
    ctxBusy(idx) := false.B
    noMoreBatches(idx) := false.B
    batchRefIdx(idx) := 0.U
    batchRefsRemaining(idx) := 0.U
    batchInProgress(idx) := false.B
    resultCapturePending(idx) := false.B
    refBufIdx(idx) := 0.U
    refBufCount(idx) := 0.U
    refInflight(idx) := false.B
    bestT(idx) := missT
    bestId(idx) := 0.U
    hasHit(idx) := false.B
    triOutstanding(idx) := 0.U
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
    refBufIdx(ctx) := 0.U
    refBufCount(ctx) := 0.U
    refInflight(ctx) := false.B
    triOutstanding(ctx) := 0.U
    bestT(ctx) := missT
    bestId(ctx) := 0.U
    hasHit(ctx) := false.B
  }

  private val refReqPipeLen = GlobalConfig.triRefMemDpiLatency
  private val refReqLivePipe = RegInit(VecInit(Seq.fill(refReqPipeLen)(false.B)))
  private val refReqCtxPipe = Reg(Vec(refReqPipeLen, UInt(ctxW.W)))
  private val refReqEpochPipe = Reg(Vec(refReqPipeLen, Bool()))
  private val refReqOffsetPipe = Reg(Vec(refReqPipeLen, UInt(refPackShift.W)))
  private val refReqCountPipe = Reg(Vec(refReqPipeLen, UInt(refCountWidth.W)))

  private val refWordAddr = Wire(Vec(ctxCount, UInt(c.addrWidth.W)))
  private val refWordOff = Wire(Vec(ctxCount, UInt(refPackShift.W)))
  private val refChunkCount = Wire(Vec(ctxCount, UInt(refCountWidth.W)))
  private val refCanReq = Wire(Vec(ctxCount, Bool()))
  private val memCanEnq = Wire(Vec(ctxCount, Bool()))
  private val geomTriId = Wire(Vec(ctxCount, UInt(triIdWidth.W)))
  private val refShadowEmpty = Wire(Vec(ctxCount, Bool()))

  for (ctx <- 0 until ctxCount) {
    refShadowQ(ctx).io.enq.valid := false.B
    refShadowQ(ctx).io.enq.bits := 0.U.asTypeOf(new RefWord)
    refShadowQ(ctx).io.deq.ready := false.B
    refShadowQ(ctx).io.flush.get := io.clear_ctx(ctx) ||
      (io.tri_batch_valid && io.output_ready(io.tri_batch_ctx) && io.tri_batch_ctx === ctx.U)
    refShadowEmpty(ctx) := !refShadowQ(ctx).io.deq.valid
    refWordAddr(ctx) := batchRefIdx(ctx) >> refPackShift
    refWordOff(ctx) := batchRefIdx(ctx)(refPackShift - 1, 0)
    val refChunkAvailWide = refPackFactor.U(16.W) - refWordOff(ctx)
    val refChunkCountWide = Mux(batchRefsRemaining(ctx) < refChunkAvailWide, batchRefsRemaining(ctx), refChunkAvailWide)
    refChunkCount(ctx) := refChunkCountWide(refCountWidth - 1, 0)
    refCanReq(ctx) := batchInProgress(ctx) &&
      batchRefsRemaining(ctx) =/= 0.U &&
      refShadowQ(ctx).io.enq.ready &&
      !refInflight(ctx)
    geomTriId(ctx) := refBuffer(ctx)(refBufIdx(ctx))
    memCanEnq(ctx) := batchInProgress(ctx) && refBufCount(ctx) =/= 0.U
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
    refInflight(refSel) := true.B
    refRr := !refSel(0)
  }

  io.ref_mem_resp.ready := true.B
  refReqLivePipe(0) := io.ref_mem_req.fire
  refReqCtxPipe(0) := Mux(io.ref_mem_req.fire, refSel, 0.U)
  refReqEpochPipe(0) := Mux(io.ref_mem_req.fire, activeEpoch(refSel), false.B)
  refReqOffsetPipe(0) := Mux(io.ref_mem_req.fire, refWordOff(refSel), 0.U)
  refReqCountPipe(0) := Mux(io.ref_mem_req.fire, refChunkCount(refSel), 0.U)
  for (i <- 1 until refReqPipeLen) {
    refReqLivePipe(i) := refReqLivePipe(i - 1)
    refReqCtxPipe(i) := refReqCtxPipe(i - 1)
    refReqEpochPipe(i) := refReqEpochPipe(i - 1)
    refReqOffsetPipe(i) := refReqOffsetPipe(i - 1)
    refReqCountPipe(i) := refReqCountPipe(i - 1)
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

  for (ctx <- 0 until ctxCount) {
    val refRespToCtx = refRespLive && refRespCtx === ctx.U
    when(refRespToCtx) {
      assert(refShadowQ(ctx).io.enq.ready, "TriPE ref shadow queue overflow")
    }
    refShadowQ(ctx).io.enq.valid := refRespToCtx
    for (i <- 0 until refPackFactor) {
      refShadowQ(ctx).io.enq.bits.ids(i) := compactedIds(i)
    }
    refShadowQ(ctx).io.enq.bits.count := refReqCountPipe.last
  }

  when(refRespLive) {
    refInflight(refRespCtx) := false.B
  }

  for (ctx <- 0 until ctxCount) {
    refShadowQ(ctx).io.deq.ready := batchInProgress(ctx) && refBufCount(ctx) === 0.U && !io.clear_ctx(ctx)
    when(refShadowQ(ctx).io.deq.fire) {
      for (i <- 0 until refPackFactor) {
        refBuffer(ctx)(i) := refShadowQ(ctx).io.deq.bits.ids(i)
      }
      refBufIdx(ctx) := 0.U
      refBufCount(ctx) := refShadowQ(ctx).io.deq.bits.count
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
    refBufIdx(memSel) := refBufIdx(memSel) + 1.U
    refBufCount(memSel) := refBufCount(memSel) - 1.U
    memRr := !memSel(0)
  }

  val memReqStale = memReqQ.io.deq.valid &&
    ((memReqQ.io.deq.bits.epoch =/= activeEpoch(memReqQ.io.deq.bits.ctx)) || io.clear_ctx(memReqQ.io.deq.bits.ctx))
  val memReqToMemValid = memReqQ.io.deq.valid && !memReqStale
  val memReqToMemReady = io.mem_req.ready && io.mem_req_mask.ready
  io.mem_req.valid := memReqToMemValid
  io.mem_req.bits := memReqQ.io.deq.bits.addr
  io.mem_req_mask.valid := memReqToMemValid
  io.mem_req_mask.bits := 1.U
  memReqQ.io.deq.ready := memReqStale || (memReqToMemValid && memReqToMemReady)

  io.mem_resp.ready := true.B

  private val memReqPipeLen = GlobalConfig.triMemDpiLatency
  private val memReqLivePipe = RegInit(VecInit(Seq.fill(memReqPipeLen)(false.B)))
  private val memReqCtxPipe = Reg(Vec(memReqPipeLen, UInt(ctxW.W)))
  private val memReqEpochPipe = Reg(Vec(memReqPipeLen, Bool()))
  private val memReqIssued = memReqQ.io.deq.fire && !memReqStale
  memReqLivePipe(0) := memReqIssued
  memReqCtxPipe(0) := Mux(memReqIssued, memReqQ.io.deq.bits.ctx, 0.U)
  memReqEpochPipe(0) := Mux(memReqIssued, memReqQ.io.deq.bits.epoch, false.B)
  for (i <- 1 until memReqPipeLen) {
    memReqLivePipe(i) := memReqLivePipe(i - 1)
    memReqCtxPipe(i) := memReqCtxPipe(i - 1)
    memReqEpochPipe(i) := memReqEpochPipe(i - 1)
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
    val triIssued = memReqQ.io.enq.fire && memSel === ctx.U
    val invalidMemResp = memRespLive && !io.mem_resp.bits.mask(0) && memRespCtx === ctx.U
    val triDone = pe.io.out_valid && peCtxOut === ctx.U
    val triRetired = invalidMemResp || triDone
    when(invalidMemResp && triDone) {
      assert(false.B, "TriPE invalid mem response and PE done for same context in same cycle")
    }
    when(io.clear_ctx(ctx)) {
      triOutstanding(ctx) := 0.U
    }.otherwise {
      switch(Cat(triIssued, triRetired)) {
        is("b10".U) { triOutstanding(ctx) := triOutstanding(ctx) + 1.U }
        is("b01".U) { triOutstanding(ctx) := triOutstanding(ctx) - 1.U }
      }
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
    val batchSourceDrained =
      batchRefsRemaining(ctx) === 0.U &&
        refBufCount(ctx) === 0.U &&
        refShadowEmpty(ctx) &&
        !refInflight(ctx)
    val batchDoneNow = batchInProgress(ctx) && batchSourceDrained && triOutstanding(ctx) === 0.U

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
