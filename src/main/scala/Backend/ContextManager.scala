package Backend

import chisel3._
import chisel3.util._
import raytrace_utils._

/** Owns the 32 backend contexts and schedules one stateless DDA step at a time.
  *
  * Traversal state remains in the context table.  The only per-context command
  * storage is the FIFO of triangle batches consumed directly by Trace.
  */
class ContextManager(
  cfg: FloatConfig = FloatConfig.FP32,
  addrWidth: Int = 32
) extends Module {
  private val contextCount = GlobalConfig.backendContextCount
  private val contextBits = GlobalConfig.backendContextBits
  private val traceLanes = GlobalConfig.traceNumWorkers * 2
  private val numWorkers = GlobalConfig.traceNumWorkers

  val io = IO(new Bundle {
    val init_alloc = Decoupled(UInt(contextBits.W))
    val context_in = Flipped(Decoupled(new BackendContextInit(cfg, addrWidth)))
    val step_in = Decoupled(new DdaTraversalState(cfg))
    val step_out = Flipped(Decoupled(new DdaStepResult(cfg, addrWidth)))

    val trace_claim = Decoupled(new TraceContextDesc(cfg, addrWidth))
    val trace_cmd_context = Input(Vec(traceLanes, UInt(contextBits.W)))
    val trace_cmd_active = Input(Vec(traceLanes, Bool()))
    val trace_cmd = Vec(traceLanes, Decoupled(new TriBatch(addrWidth)))
    val trace_dda_done = Output(Vec(traceLanes, Bool()))
    val trace_stop = Vec(numWorkers, Flipped(Valid(UInt(contextBits.W))))
    val trace_release = Vec(numWorkers, Flipped(Valid(UInt(contextBits.W))))
    val watchdog_error = Output(Bool())
  })

  val immutableTable = Reg(Vec(contextCount, new ImmutableRayRecord(cfg, addrWidth)))
  val traversalTable = Reg(Vec(contextCount, new DdaTraversalState(cfg)))
  val active = RegInit(VecInit(Seq.fill(contextCount)(false.B)))
  val initReserved = RegInit(VecInit(Seq.fill(contextCount)(false.B)))
  val claimed = RegInit(VecInit(Seq.fill(contextCount)(false.B)))
  val ddaDone = RegInit(VecInit(Seq.fill(contextCount)(false.B)))
  val traceStopped = RegInit(VecInit(Seq.fill(contextCount)(false.B)))
  val resultAccepted = RegInit(VecInit(Seq.fill(contextCount)(false.B)))
  val stepInFlight = RegInit(VecInit(Seq.fill(contextCount)(false.B)))
  val watchdogError = RegInit(false.B)
  io.watchdog_error := watchdogError

  val cmdFifos = Seq.fill(contextCount)(Module(new Queue(
    new TriBatch(addrWidth),
    GlobalConfig.backendContextCmdFifoDepth,
    hasFlush = true
  )))

  val stopNow = Wire(Vec(contextCount, Bool()))
  val releaseNow = Wire(Vec(contextCount, Bool()))
  val initCommit = Wire(Vec(contextCount, Bool()))
  for (ctx <- 0 until contextCount) {
    stopNow(ctx) := io.trace_stop.map(s => s.valid && s.bits === ctx.U).reduce(_ || _)
    releaseNow(ctx) := io.trace_release.map(r => r.valid && r.bits === ctx.U).reduce(_ || _)
    initCommit(ctx) := io.context_in.fire && io.context_in.bits.contextId === ctx.U
    cmdFifos(ctx).io.enq.valid := false.B
    cmdFifos(ctx).io.enq.bits := io.step_out.bits.tri
    cmdFifos(ctx).io.deq.ready := false.B
    cmdFifos(ctx).io.flush.get := stopNow(ctx) || releaseNow(ctx) || initCommit(ctx)
  }

  val freeForInit = VecInit((0 until contextCount).map(ctx =>
    !active(ctx) && !initReserved(ctx)
  ))
  val initAllocOH = PriorityEncoderOH(freeForInit)
  val initAllocId = OHToUInt(initAllocOH)
  io.init_alloc.valid := freeForInit.asUInt.orR
  io.init_alloc.bits := initAllocId
  when(io.init_alloc.fire) {
    initReserved(initAllocId) := true.B
  }

  io.context_in.ready := true.B
  when(io.context_in.fire) {
    val id = io.context_in.bits.contextId
    assert(initReserved(id), "Backend initialized an unreserved context")
    assert(id === io.context_in.bits.traversal.contextId,
      "Backend init context ID disagrees with traversal state")
    immutableTable(id) := io.context_in.bits.immutable
    traversalTable(id) := io.context_in.bits.traversal
    active(id) := true.B
    initReserved(id) := false.B
    claimed(id) := false.B
    ddaDone(id) := false.B
    traceStopped(id) := false.B
    resultAccepted(id) := false.B
    stepInFlight(id) := false.B
  }

  val stepEligible = Wire(Vec(contextCount, Bool()))
  for (ctx <- 0 until contextCount) {
    stepEligible(ctx) := active(ctx) && !ddaDone(ctx) && !traceStopped(ctx) &&
      !resultAccepted(ctx) && !stepInFlight(ctx) && cmdFifos(ctx).io.enq.ready
  }
  val stepRr = RegInit(0.U(contextBits.W))
  val stepRotated = Wire(Vec(contextCount, Bool()))
  for (offset <- 0 until contextCount) {
    val ctx = (stepRr + offset.U)(contextBits - 1, 0)
    stepRotated(offset) := stepEligible(ctx)
  }
  val stepPickOH = PriorityEncoderOH(stepRotated)
  val stepPickId = (OHToUInt(stepPickOH) + stepRr)(contextBits - 1, 0)
  io.step_in.valid := stepRotated.asUInt.orR
  io.step_in.bits := traversalTable(stepPickId)
  when(io.step_in.fire) {
    assert(stepEligible(stepPickId), "Backend issued an ineligible context")
    stepInFlight(stepPickId) := true.B
    stepRr := stepPickId + 1.U
  }

  io.step_out.ready := true.B
  val responseFire = io.step_out.fire
  val responseId = io.step_out.bits.state.contextId
  val responseLive = responseFire && !resultAccepted(responseId) &&
    !traceStopped(responseId) && !stopNow(responseId) && !releaseNow(responseId)
  when(responseFire) {
    assert(stepInFlight(responseId), "DDA step returned for a context with no request in flight")
    stepInFlight(responseId) := false.B
    when(resultAccepted(responseId)) {
      active(responseId) := false.B
      claimed(responseId) := false.B
      ddaDone(responseId) := false.B
      traceStopped(responseId) := false.B
      resultAccepted(responseId) := false.B
    }.elsewhen(responseLive) {
      traversalTable(responseId) := io.step_out.bits.state
      when(io.step_out.bits.done) {
        ddaDone(responseId) := true.B
        when(io.step_out.bits.watchdog) {
          watchdogError := true.B
        }
      }
    }
  }
  for (ctx <- 0 until contextCount) {
    val enqueue = responseLive && responseId === ctx.U &&
      !io.step_out.bits.done && io.step_out.bits.emitCmd
    cmdFifos(ctx).io.enq.valid := enqueue
    when(enqueue) {
      assert(cmdFifos(ctx).io.enq.ready, "Backend context command FIFO overflow")
    }
  }

  val claimEligible = Wire(Vec(contextCount, Bool()))
  for (ctx <- 0 until contextCount) {
    claimEligible(ctx) := active(ctx) && !claimed(ctx) && !traceStopped(ctx) &&
      (cmdFifos(ctx).io.deq.valid || ddaDone(ctx))
  }
  val claimRr = RegInit(0.U(contextBits.W))
  val claimRotated = Wire(Vec(contextCount, Bool()))
  for (offset <- 0 until contextCount) {
    val ctx = (claimRr + offset.U)(contextBits - 1, 0)
    claimRotated(offset) := claimEligible(ctx)
  }
  val claimPickOH = PriorityEncoderOH(claimRotated)
  val claimPickId = (OHToUInt(claimPickOH) + claimRr)(contextBits - 1, 0)
  io.trace_claim.valid := claimRotated.asUInt.orR
  io.trace_claim.bits.contextId := claimPickId
  io.trace_claim.bits.ray.origin := immutableTable(claimPickId).origin
  io.trace_claim.bits.ray.dir := immutableTable(claimPickId).dir
  io.trace_claim.bits.ray.dist := 0.U
  io.trace_claim.bits.meta := immutableTable(claimPickId).meta
  when(io.trace_claim.fire) {
    assert(claimEligible(claimPickId), "Trace claimed an ineligible backend context")
    claimed(claimPickId) := true.B
    claimRr := claimPickId + 1.U
  }

  val queueDeqValid = VecInit(cmdFifos.map(_.io.deq.valid))
  val queueDeqBits = VecInit(cmdFifos.map(_.io.deq.bits))
  for (lane <- 0 until traceLanes) {
    val ctx = io.trace_cmd_context(lane)
    io.trace_cmd(lane).valid := io.trace_cmd_active(lane) && queueDeqValid(ctx)
    io.trace_cmd(lane).bits := queueDeqBits(ctx)
    io.trace_dda_done(lane) := io.trace_cmd_active(lane) &&
      ddaDone(ctx) && !queueDeqValid(ctx)
  }
  for (ctx <- 0 until contextCount) {
    val laneMatch = VecInit((0 until traceLanes).map(lane =>
      io.trace_cmd_active(lane) && io.trace_cmd_context(lane) === ctx.U
    ))
    assert(PopCount(laneMatch) <= 1.U,
      "A backend context is bound to multiple Trace lanes")
    cmdFifos(ctx).io.deq.ready := (0 until traceLanes).map(lane =>
      laneMatch(lane) && io.trace_cmd(lane).ready
    ).reduce(_ || _)
  }

  for (ctx <- 0 until contextCount) {
    when(stopNow(ctx)) {
      assert(active(ctx) && claimed(ctx), "Trace stopped an inactive or unclaimed context")
      traceStopped(ctx) := true.B
    }
    when(releaseNow(ctx)) {
      assert(active(ctx) && claimed(ctx), "Trace released an inactive or unclaimed context")
      resultAccepted(ctx) := true.B
      val returningNow = responseFire && responseId === ctx.U
      when(!stepInFlight(ctx) || returningNow) {
        active(ctx) := false.B
        claimed(ctx) := false.B
        ddaDone(ctx) := false.B
        traceStopped(ctx) := false.B
        resultAccepted(ctx) := false.B
      }
    }
    assert(!(active(ctx) && initReserved(ctx)),
      "Backend context is active and init-reserved simultaneously")
  }
}
