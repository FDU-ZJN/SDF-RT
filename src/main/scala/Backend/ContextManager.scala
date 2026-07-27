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
  private val traceLanes = GlobalConfig.traceNumWorkers * 2 // legacy width
  private val numWorkers = GlobalConfig.traceNumWorkers

  val io = IO(new Bundle {
    val init_alloc = Decoupled(UInt(contextBits.W))
    val context_in = Flipped(Decoupled(new BackendContextInit(cfg, addrWidth)))
    val step_in = Decoupled(new DdaTraversalState(cfg))
    // DDA and Trace are event producers.  Context state is changed only by
    // the context owner after these registered Decoupled event boundaries.
    val dda_event = Flipped(Decoupled(new DdaStepResult(cfg, addrWidth)))

    val trace_start = Vec(contextCount, Decoupled(new TraceContextDesc(cfg, addrWidth)))
    val trace_cmd = Vec(contextCount, Decoupled(new TriBatch(addrWidth)))
    val trace_dda_done = Output(Vec(contextCount, Bool()))
    // Deprecated dynamic-lane ports retained for source compatibility.
    val trace_claim = Decoupled(new TraceContextDesc(cfg, addrWidth))
    val trace_cmd_context = Input(Vec(traceLanes, UInt(contextBits.W)))
    val trace_cmd_active = Input(Vec(traceLanes, Bool()))
    val trace_cmd_legacy = Vec(traceLanes, Decoupled(new TriBatch(addrWidth)))
    val trace_stop_event = Vec(numWorkers, Flipped(Decoupled(UInt(contextBits.W))))
    val trace_release_event = Vec(numWorkers, Flipped(Decoupled(UInt(contextBits.W))))
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

  val cmdFifos = Seq.fill(contextCount)(Module(new Queue(
    new TriBatch(addrWidth),
    GlobalConfig.backendContextCmdFifoDepth,
    hasFlush = true
  )))
  private val cmdCountWidth = log2Ceil(GlobalConfig.backendContextCmdFifoDepth + 1)
  val cmdOccupancy = RegInit(VecInit(Seq.fill(contextCount)(0.U(cmdCountWidth.W))))

  val stopNow = Wire(Vec(contextCount, Bool()))
  val releaseNow = Wire(Vec(contextCount, Bool()))
  val initCommit = Wire(Vec(contextCount, Bool()))
  for (ctx <- 0 until contextCount) {
    stopNow(ctx) := io.trace_stop_event.map { event =>
      event.valid && event.bits === ctx.U
    }.reduce(_ || _)
    releaseNow(ctx) := io.trace_release_event.map { event =>
      event.valid && event.bits === ctx.U
    }.reduce(_ || _)
    initCommit(ctx) := io.context_in.fire && io.context_in.bits.contextId === ctx.U
    cmdFifos(ctx).io.enq.valid := false.B
    cmdFifos(ctx).io.enq.bits := io.dda_event.bits.tri
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
      !resultAccepted(ctx) && !stepInFlight(ctx) && !stopNow(ctx) &&
      !releaseNow(ctx) && !initCommit(ctx) &&
      cmdOccupancy(ctx) < GlobalConfig.backendContextCmdFifoDepth.U
  }
  val stepRr = RegInit(0.U(contextBits.W))
  val stepRotated = Wire(Vec(contextCount, Bool()))
  for (offset <- 0 until contextCount) {
    val ctx = (stepRr + offset.U)(contextBits - 1, 0)
    stepRotated(offset) := stepEligible(ctx)
  }
  val stepPickOH = PriorityEncoderOH(stepRotated)
  val stepPickId = (OHToUInt(stepPickOH) + stepRr)(contextBits - 1, 0)
  // The scheduler first reserves only a context ID.  A second registered
  // stage uses that ID to read the traversal table into the DDA issue
  // register.  This keeps FIFO credit and the 16-way eligibility selector off
  // the wide traversal-state register inputs while retaining one step/cycle.
  val stepSelectValid = RegInit(false.B)
  val stepSelectId = Reg(UInt(contextBits.W))
  val stepPendingValid = RegInit(false.B)
  val stepPendingState = Reg(new DdaTraversalState(cfg))
  io.step_in.valid := stepPendingValid
  io.step_in.bits := stepPendingState

  val stepPendingCanAdvance = !stepPendingValid || io.step_in.ready
  val stepSelectCanAdvance = !stepSelectValid || stepPendingCanAdvance
  when(stepPendingCanAdvance) {
    stepPendingValid := stepSelectValid
    when(stepSelectValid) {
      stepPendingState := traversalTable(stepSelectId)
    }
  }
  when(stepSelectCanAdvance) {
    when(stepRotated.asUInt.orR) {
      assert(stepEligible(stepPickId), "Backend selected an ineligible context")
      stepSelectValid := true.B
      stepSelectId := stepPickId
      stepInFlight(stepPickId) := true.B
      stepRr := stepPickId + 1.U
    }.otherwise {
      stepSelectValid := false.B
    }
  }

  val responseId = io.dda_event.bits.state.contextId
  // Eligibility reserves the only possible command slot before issuing a
  // step.  With at most one step in flight per context, every live DDA event
  // is therefore commit-ready and the non-stallable DDA pipeline never sees
  // backpressure from Trace.
  io.dda_event.ready := true.B
  val responseFire = io.dda_event.fire
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
      traversalTable(responseId) := io.dda_event.bits.state
      when(io.dda_event.bits.done) {
        ddaDone(responseId) := true.B
      }
    }
  }
  for (ctx <- 0 until contextCount) {
    val enqueue = responseLive && responseId === ctx.U &&
      !io.dda_event.bits.done && io.dda_event.bits.emitCmd
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

  for (worker <- 0 until numWorkers) {
    io.trace_stop_event(worker).ready := true.B
    io.trace_release_event(worker).ready := true.B
  }
  io.trace_claim.valid := false.B
  io.trace_claim.bits := 0.U.asTypeOf(io.trace_claim.bits)
  for (ctx <- 0 until contextCount) {
    io.trace_start(ctx).valid := claimEligible(ctx)
    io.trace_start(ctx).bits.contextId := ctx.U
    io.trace_start(ctx).bits.ray.origin := immutableTable(ctx).origin
    io.trace_start(ctx).bits.ray.dir := immutableTable(ctx).dir
    io.trace_start(ctx).bits.ray.dist := 0.U
    io.trace_start(ctx).bits.meta := immutableTable(ctx).meta
    when(io.trace_start(ctx).fire) { claimed(ctx) := true.B }
    io.trace_cmd(ctx).valid := claimed(ctx) && !traceStopped(ctx) && cmdFifos(ctx).io.deq.valid
    io.trace_cmd(ctx).bits := cmdFifos(ctx).io.deq.bits
    cmdFifos(ctx).io.deq.ready := io.trace_cmd(ctx).ready
    io.trace_dda_done(ctx) := claimed(ctx) && ddaDone(ctx) && cmdOccupancy(ctx) === 0.U

    val cmdEnqueue = cmdFifos(ctx).io.enq.fire
    val cmdDequeue = cmdFifos(ctx).io.deq.fire
    when(stopNow(ctx) || releaseNow(ctx) || initCommit(ctx)) {
      cmdOccupancy(ctx) := 0.U
    }.elsewhen(cmdEnqueue =/= cmdDequeue) {
      cmdOccupancy(ctx) := Mux(cmdEnqueue,
        cmdOccupancy(ctx) + 1.U,
        cmdOccupancy(ctx) - 1.U)
    }
    assert(cmdOccupancy(ctx) <= GlobalConfig.backendContextCmdFifoDepth.U,
      "Backend command occupancy exceeded FIFO depth")
    assert(cmdFifos(ctx).io.deq.valid === (cmdOccupancy(ctx) =/= 0.U),
      "Backend command occupancy shadow disagrees with FIFO")
  }
  for (lane <- 0 until traceLanes) { io.trace_cmd_legacy(lane).valid := false.B; io.trace_cmd_legacy(lane).bits := 0.U.asTypeOf(new TriBatch(addrWidth)) }

  for (ctx <- 0 until contextCount) {
    when(stopNow(ctx)) {
      when(active(ctx) && claimed(ctx)) { traceStopped(ctx) := true.B }
    }
    when(releaseNow(ctx)) {
      when(active(ctx) && claimed(ctx)) {
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
    }
    assert(!(active(ctx) && initReserved(ctx)),
      "Backend context is active and init-reserved simultaneously")
  }
}
