package Trace

import chisel3._
import chisel3.util._
import raytrace_utils._

/** Fixed ownership trace fabric: context ctx is handled by worker ctx/4. */
class TraceController(c: TriPeConfig = TriPeConfig()) extends Module {
  private val numWorkers = GlobalConfig.traceNumWorkers
  private val contextsPerWorker = GlobalConfig.traceContextsPerWorker
  private val contextCount = GlobalConfig.traceContextCount
  private val contextBits = GlobalConfig.backendContextBits
  require(GlobalConfig.backendContextCount % numWorkers == 0)
  require(contextsPerWorker == 4)

  val io = IO(new Bundle {
    val trace_start = Flipped(Vec(contextCount, Decoupled(new TraceContextDesc(c.cfg, c.addrWidth))))
    val trace_cmd = Flipped(Vec(contextCount, Decoupled(new TriBatch(c.addrWidth))))
    val trace_dda_done = Input(Vec(contextCount, Bool()))
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
  val workers = Seq.fill(numWorkers)(Module(new TriPE(c, contextsPerWorker)))
  val mem = Module(new TriangleMemMultiPort(c, numWorkers,
    tagWidth = log2Ceil(contextsPerWorker) + 1))
  mem.io.triangleBaseAddress := io.triangleBaseAddress
  io.triDmaCmd <> mem.io.dmaCmd; mem.io.dmaData <> io.triDmaData; mem.io.dmaStatus <> io.triDmaStatus
  io.triDmaReadError := mem.io.dmaReadError; io.triDmaMalformedLineCount := mem.io.dmaMalformedLineCount
  io.triDmaStatusErrorCount := mem.io.dmaStatusErrorCount; io.triDmaTagMismatchCount := mem.io.dmaTagMismatchCount
  io.triAxiIssuedCount := mem.io.axiIssuedCount; io.triAxiCompletedCount := mem.io.axiCompletedCount
  io.triAxiOutstandingCount := mem.io.axiOutstandingCount
  val results = Seq.fill(numWorkers)(Reg(Vec(contextsPerWorker, new TraceResult(c.cfg, c.addrWidth))))
  val resultValid = Seq.fill(numWorkers)(RegInit(VecInit(Seq.fill(contextsPerWorker)(false.B))))
  for (w <- 0 until numWorkers) {
    val p = workers(w); p.io.ray_in := 0.U.asTypeOf(new Ray(c.cfg)); p.io.ray_meta := 0.U.asTypeOf(new RayMeta(c.addrWidth)); p.io.ray_ctx := 0.U; p.io.ray_valid := false.B
    p.io.tri_batch_in := 0.U.asTypeOf(new TriBatch(c.addrWidth)); p.io.tri_batch_ctx := 0.U; p.io.tri_batch_valid := false.B
    val startArb = Module(new RRArbiter(new TraceContextDesc(c.cfg, c.addrWidth), contextsPerWorker))
    val cmdArb = Module(new RRArbiter(new TriBatch(c.addrWidth), contextsPerWorker))
    for (l <- 0 until contextsPerWorker) {
      val ctx = w * contextsPerWorker + l
      startArb.io.in(l).valid := io.trace_start(ctx).valid && p.io.start_ready(l)
      startArb.io.in(l).bits := io.trace_start(ctx).bits
      io.trace_start(ctx).ready := startArb.io.in(l).ready && p.io.start_ready(l)
      cmdArb.io.in(l).valid := io.trace_cmd(ctx).valid && p.io.output_ready(l)
      cmdArb.io.in(l).bits := io.trace_cmd(ctx).bits
      io.trace_cmd(ctx).ready := cmdArb.io.in(l).ready && p.io.output_ready(l)
      p.io.dda_done(l) := io.trace_dda_done(ctx)
      p.io.clear_ctx(l) := false.B
    }
    startArb.io.out.ready := true.B
    when(startArb.io.out.fire) { p.io.ray_in := startArb.io.out.bits.ray; p.io.ray_meta := startArb.io.out.bits.meta; p.io.ray_ctx := startArb.io.chosen; p.io.ray_valid := true.B }
    cmdArb.io.out.ready := true.B
    when(cmdArb.io.out.fire) { p.io.tri_batch_in := cmdArb.io.out.bits; p.io.tri_batch_ctx := cmdArb.io.chosen; p.io.tri_batch_valid := true.B }
    mem.io.req(w) <> p.io.mem_req; p.io.mem_resp <> mem.io.resp(w)
    val arb = Module(new RRArbiter(new TraceResult(c.cfg, c.addrWidth), contextsPerWorker))
    for (l <- 0 until contextsPerWorker) { arb.io.in(l).valid := resultValid(w)(l); arb.io.in(l).bits := results(w)(l) }
    io.result_out(w) <> arb.io.out
    io.trace_release(w).valid := arb.io.out.fire; io.trace_release(w).bits := (w * contextsPerWorker).U + arb.io.chosen
    io.trace_stop(w).valid := false.B; io.trace_stop(w).bits := 0.U
    // Capture a PE result exactly once.  TriPE keeps out_done asserted until
    // its context is cleared; without this guard a backpressured result can
    // be re-enqueued every cycle and later appear as duplicate RayIDs.
    when(p.io.out_done && !resultValid(w)(p.io.out_ctx)) {
      val l = p.io.out_ctx
      results(w)(l).meta := p.io.out_meta; results(w)(l).hit := p.io.out_best_hit; results(w)(l).hitId := p.io.hit_id; results(w)(l).hitT := p.io.t_best
      resultValid(w)(l) := true.B; p.io.clear_ctx(l) := true.B
      io.trace_stop(w).valid := p.io.out_early_stop; io.trace_stop(w).bits := (w * contextsPerWorker).U + l
    }
    when(arb.io.out.fire) { resultValid(w)(arb.io.chosen) := false.B }
  }
}
