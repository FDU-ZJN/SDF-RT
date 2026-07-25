package Trace

import chisel3._
import chisel3.util._
import raytrace_utils._

class TraceController(c: TriPeConfig = TriPeConfig()) extends Module {
  private val numWorkers = GlobalConfig.traceNumWorkers
  private val ctxCount = 2
  private val traceLanes = numWorkers * ctxCount
  private val traceLaneBits = log2Ceil(traceLanes)
  private val contextBits = GlobalConfig.ddaTraceSlotBits

  val io = IO(new Bundle {
    val trace_claim = Flipped(Decoupled(new TraceContextDesc(c.cfg, c.addrWidth)))
    val trace_cmd_context = Output(Vec(traceLanes, UInt(contextBits.W)))
    val trace_cmd_active = Output(Vec(traceLanes, Bool()))
    val trace_cmd = Flipped(Vec(traceLanes, Decoupled(new TriBatch(c.addrWidth))))
    val trace_dda_done = Input(Vec(traceLanes, Bool()))
    val trace_stop = Vec(numWorkers, Valid(UInt(contextBits.W)))
    val trace_release = Vec(numWorkers, Valid(UInt(contextBits.W)))
    val result_out = Vec(numWorkers, Decoupled(new TraceResult(c.cfg, c.addrWidth)))

    val triangleBaseAddress = Input(UInt(64.W))
    val triDmaCmd = Decoupled(UInt(80.W))
    val triDmaData = Flipped(Decoupled(new TriMemDmaReadData(128)))
    val triDmaStatus = Flipped(Decoupled(UInt(8.W)))
    val triDmaReadError = Output(Bool())
    val triDmaMalformedLineCount = Output(UInt(32.W))
    val triDmaStatusErrorCount = Output(UInt(32.W))
    val triDmaTagMismatchCount = Output(UInt(32.W))
    val triAxiIssuedCount = Output(UInt(32.W))
    val triAxiCompletedCount = Output(UInt(32.W))
    val triAxiOutstandingCount = Output(UInt(8.W))
  })

  val workers = Seq.fill(numWorkers)(Module(new TriPE(c)))
  val mem = Module(new TriangleMemMultiPort(c, numWorkers, tagWidth = 2))
  mem.io.triangleBaseAddress := io.triangleBaseAddress
  io.triDmaCmd <> mem.io.dmaCmd
  mem.io.dmaData <> io.triDmaData
  mem.io.dmaStatus <> io.triDmaStatus
  io.triDmaReadError := mem.io.dmaReadError
  io.triDmaMalformedLineCount := mem.io.dmaMalformedLineCount
  io.triDmaStatusErrorCount := mem.io.dmaStatusErrorCount
  io.triDmaTagMismatchCount := mem.io.dmaTagMismatchCount
  io.triAxiIssuedCount := mem.io.axiIssuedCount
  io.triAxiCompletedCount := mem.io.axiCompletedCount
  io.triAxiOutstandingCount := mem.io.axiOutstandingCount

  val laneBound = RegInit(VecInit(Seq.fill(traceLanes)(false.B)))
  val laneStarted = RegInit(VecInit(Seq.fill(traceLanes)(false.B)))
  val laneFinished = RegInit(VecInit(Seq.fill(traceLanes)(false.B)))
  val laneContext = Reg(Vec(traceLanes, UInt(contextBits.W)))
  val laneResult = Reg(Vec(traceLanes, new TraceResult(c.cfg, c.addrWidth)))
  val laneResultValid = RegInit(VecInit(Seq.fill(traceLanes)(false.B)))

  val claimRr = RegInit(0.U(traceLaneBits.W))
  val freeRotated = Wire(Vec(traceLanes, Bool()))
  for (offset <- 0 until traceLanes) {
    val lane = (claimRr + offset.U)(traceLaneBits - 1, 0)
    val startReady = VecInit((0 until traceLanes).map(i =>
      workers(i / ctxCount).io.start_ready(i % ctxCount)
    ))
    freeRotated(offset) := !laneBound(lane) && startReady(lane)
  }
  val claimOH = PriorityEncoderOH(freeRotated)
  val claimValid = freeRotated.asUInt.orR
  val claimLane = (OHToUInt(claimOH) + claimRr)(traceLaneBits - 1, 0)
  io.trace_claim.ready := claimValid
  when(io.trace_claim.fire) {
    laneBound(claimLane) := true.B
    laneStarted(claimLane) := true.B
    laneFinished(claimLane) := false.B
    laneContext(claimLane) := io.trace_claim.bits.contextId
    claimRr := claimLane + 1.U
  }

  for (lane <- 0 until traceLanes) {
    io.trace_cmd_context(lane) := laneContext(lane)
    io.trace_cmd_active(lane) := laneBound(lane) && laneStarted(lane) && !laneFinished(lane)
    io.trace_cmd(lane).ready := false.B
  }

  val batchRr = RegInit(VecInit(Seq.fill(numWorkers)(false.B)))
  for (w <- 0 until numWorkers) {
    val lane0 = w * ctxCount
    val lane1 = lane0 + 1
    val worker = workers(w)

    worker.io.ray_in := 0.U.asTypeOf(new Ray(c.cfg))
    worker.io.ray_meta := 0.U.asTypeOf(new RayMeta(c.addrWidth))
    worker.io.ray_ctx := 0.U
    worker.io.ray_valid := false.B
    worker.io.tri_batch_in := 0.U.asTypeOf(new TriBatch(c.addrWidth))
    worker.io.tri_batch_ctx := 0.U
    worker.io.tri_batch_valid := false.B
    worker.io.clear_ctx := VecInit(Seq.fill(ctxCount)(false.B))
    for (ctx <- 0 until ctxCount) {
      val lane = lane0 + ctx
      worker.io.dda_done(ctx) := laneBound(lane) && laneStarted(lane) &&
        !laneFinished(lane) && io.trace_dda_done(lane)
    }

    val claim0 = io.trace_claim.fire && claimLane === lane0.U
    val claim1 = io.trace_claim.fire && claimLane === lane1.U
    when(claim0 || claim1) {
      val local = Mux(claim0, 0.U, 1.U)
      worker.io.ray_in := io.trace_claim.bits.ray
      worker.io.ray_meta := io.trace_claim.bits.meta
      worker.io.ray_ctx := local
      worker.io.ray_valid := true.B
    }

    val cmd0 = io.trace_cmd(lane0).valid && io.trace_cmd_active(lane0) && worker.io.output_ready(0)
    val cmd1 = io.trace_cmd(lane1).valid && io.trace_cmd_active(lane1) && worker.io.output_ready(1)
    val choose0 = cmd0 && (!batchRr(w) || !cmd1)
    val choose1 = cmd1 && (batchRr(w) || !cmd0)
    when(choose0 || choose1) {
      val local = Mux(choose0, 0.U, 1.U)
      val lane = Mux(choose0, lane0.U(traceLaneBits.W), lane1.U(traceLaneBits.W))
      worker.io.tri_batch_in := io.trace_cmd(lane).bits
      worker.io.tri_batch_ctx := local
      worker.io.tri_batch_valid := true.B
      io.trace_cmd(lane).ready := true.B
      batchRr(w) := !local(0)
    }

    mem.io.req(w) <> worker.io.mem_req
    worker.io.mem_resp <> mem.io.resp(w)

    io.trace_stop(w).valid := false.B
    io.trace_stop(w).bits := 0.U
    when(worker.io.out_done) {
      val local = worker.io.out_ctx
      val lane = Cat(w.U(log2Ceil(numWorkers).W), local)
      assert(laneBound(lane) && laneStarted(lane) && !laneFinished(lane),
        "TriPE completed an inactive Trace lane")
      assert(!laneResultValid(lane), "TriPE overwrote a pending result")
      laneResult(lane).meta := worker.io.out_meta
      laneResult(lane).hit := worker.io.out_best_hit
      laneResult(lane).hitId := worker.io.hit_id
      laneResult(lane).hitT := worker.io.t_best
      laneResultValid(lane) := true.B
      laneFinished(lane) := true.B
      worker.io.clear_ctx(local) := true.B
      io.trace_stop(w).valid := worker.io.out_early_stop
      io.trace_stop(w).bits := laneContext(lane)
    }

    val resultArb = Module(new RRArbiter(new TraceResult(c.cfg, c.addrWidth), ctxCount))
    for (ctx <- 0 until ctxCount) {
      val lane = lane0 + ctx
      resultArb.io.in(ctx).valid := laneResultValid(lane)
      resultArb.io.in(ctx).bits := laneResult(lane)
      when(resultArb.io.in(ctx).fire) {
        laneResultValid(lane) := false.B
        laneBound(lane) := false.B
        laneStarted(lane) := false.B
        laneFinished(lane) := false.B
      }
    }
    io.result_out(w) <> resultArb.io.out
    io.trace_release(w).valid := resultArb.io.out.fire
    io.trace_release(w).bits := Mux(resultArb.io.chosen === 0.U,
      laneContext(lane0), laneContext(lane1))
  }
}
