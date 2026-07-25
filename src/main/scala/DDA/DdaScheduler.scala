package DDA

import chisel3._
import chisel3.util._
import raytrace_utils._

class DdaScheduler(
  cfg: FloatConfig = FloatConfig.FP32,
  addrWidth: Int = 32,
  maxTraversalSteps: Int = GlobalConfig.ddaMaxSteps
) extends Module {
  private val contextCount = GlobalConfig.ddaTraceSlotCount
  private val contextBits = GlobalConfig.ddaTraceSlotBits
  private val traceLanes = GlobalConfig.traceNumWorkers * 2
  private val traceCmdDepth = GlobalConfig.ddaTraceCmdDepth
  private val numWorkers = GlobalConfig.traceNumWorkers

  val io = IO(new Bundle {
    val init_alloc = Decoupled(UInt(contextBits.W))
    val context_in = Flipped(Decoupled(new DdaContext(cfg, addrWidth)))
    val step_in = Decoupled(new DdaContext(cfg, addrWidth))
    val step_out = Flipped(Decoupled(new DdaStepResult(cfg, addrWidth)))

    val trace_claim = Decoupled(new TraceContextDesc(cfg, addrWidth))
    val trace_cmd_context = Input(Vec(traceLanes, UInt(contextBits.W)))
    val trace_cmd_active = Input(Vec(traceLanes, Bool()))
    val trace_cmd = Vec(traceLanes, Decoupled(new TraceCellCmd(cfg, addrWidth)))
    val trace_dda_done = Output(Vec(traceLanes, Bool()))
    val trace_stop = Vec(numWorkers, Flipped(Valid(UInt(contextBits.W))))
    val trace_release = Vec(numWorkers, Flipped(Valid(UInt(contextBits.W))))
    val watchdog_error = Output(Bool())
  })

  val contexts = Reg(Vec(contextCount, new DdaContext(cfg, addrWidth)))
  val active = RegInit(VecInit(Seq.fill(contextCount)(false.B)))
  val initReserved = RegInit(VecInit(Seq.fill(contextCount)(false.B)))
  val claimed = RegInit(VecInit(Seq.fill(contextCount)(false.B)))
  val ddaDone = RegInit(VecInit(Seq.fill(contextCount)(false.B)))
  val traceStopped = RegInit(VecInit(Seq.fill(contextCount)(false.B)))
  val resultAccepted = RegInit(VecInit(Seq.fill(contextCount)(false.B)))
  val stepInFlight = RegInit(VecInit(Seq.fill(contextCount)(false.B)))
  val watchdogError = RegInit(false.B)
  io.watchdog_error := watchdogError

  val cmdQueues = Seq.fill(contextCount)(Module(
    new Queue(new TraceCellCmd(cfg, addrWidth), traceCmdDepth, hasFlush = true)
  ))

  val stopNow = Wire(Vec(contextCount, Bool()))
  val releaseNow = Wire(Vec(contextCount, Bool()))
  val initCommit = Wire(Vec(contextCount, Bool()))
  for (ctx <- 0 until contextCount) {
    stopNow(ctx) := io.trace_stop.map(s => s.valid && s.bits === ctx.U).reduce(_ || _)
    releaseNow(ctx) := io.trace_release.map(r => r.valid && r.bits === ctx.U).reduce(_ || _)
    initCommit(ctx) := io.context_in.fire && io.context_in.bits.traceSlot === ctx.U
    cmdQueues(ctx).io.enq.valid := false.B
    cmdQueues(ctx).io.enq.bits := 0.U.asTypeOf(new TraceCellCmd(cfg, addrWidth))
    cmdQueues(ctx).io.flush.get := stopNow(ctx) || releaseNow(ctx) || initCommit(ctx)
    cmdQueues(ctx).io.deq.ready := false.B
  }

  // Reserve an ID before the fixed-latency DDA-init suffix accepts a ray.
  val freeForInit = VecInit((0 until contextCount).map(ctx => !active(ctx) && !initReserved(ctx)))
  val initAllocOH = PriorityEncoderOH(freeForInit)
  val initAllocValid = freeForInit.asUInt.orR
  val initAllocId = OHToUInt(initAllocOH)
  io.init_alloc.valid := initAllocValid
  io.init_alloc.bits := initAllocId
  when(io.init_alloc.fire) {
    initReserved(initAllocId) := true.B
  }

  io.context_in.ready := true.B
  when(io.context_in.fire) {
    val ctx = io.context_in.bits.traceSlot
    assert(initReserved(ctx), "DDA init completed for an unreserved context")
    contexts(ctx) := io.context_in.bits
    active(ctx) := true.B
    initReserved(ctx) := false.B
    claimed(ctx) := false.B
    ddaDone(ctx) := false.B
    traceStopped(ctx) := false.B
    resultAccepted(ctx) := false.B
    stepInFlight(ctx) := false.B
  }

  // C-slow round-robin scheduling: a context is revisited only after its
  // previous step has returned and its trace FIFO has capacity.
  val stepRr = RegInit(0.U(contextBits.W))
  val queueEnqReady = VecInit(cmdQueues.map(_.io.enq.ready))
  val stepRotated = Wire(Vec(contextCount, Bool()))
  for (offset <- 0 until contextCount) {
    val ctx = (stepRr + offset.U)(contextBits - 1, 0)
    stepRotated(offset) := active(ctx) && !ddaDone(ctx) && !traceStopped(ctx) &&
      !resultAccepted(ctx) && !stepInFlight(ctx) && queueEnqReady(ctx)
  }
  val stepPickOH = PriorityEncoderOH(stepRotated)
  val stepPickValid = stepRotated.asUInt.orR
  val stepPickId = (OHToUInt(stepPickOH) + stepRr)(contextBits - 1, 0)
  io.step_in.valid := stepPickValid
  io.step_in.bits := contexts(stepPickId)
  when(io.step_in.fire) {
    stepInFlight(stepPickId) := true.B
    stepRr := stepPickId + 1.U
  }

  io.step_out.ready := true.B
  val returningContext = io.step_out.fire
  val returningContextId = io.step_out.bits.ctx.traceSlot
  when(io.step_out.fire) {
    val ctx = io.step_out.bits.ctx.traceSlot
    assert(stepInFlight(ctx), "DDA step returned for a context with no request in flight")
    stepInFlight(ctx) := false.B
    when(resultAccepted(ctx)) {
      active(ctx) := false.B
      claimed(ctx) := false.B
      ddaDone(ctx) := false.B
      traceStopped(ctx) := false.B
      resultAccepted(ctx) := false.B
    }.elsewhen(!traceStopped(ctx) && !stopNow(ctx) && !releaseNow(ctx)) {
      contexts(ctx) := io.step_out.bits.ctx
      when(io.step_out.bits.done) {
        ddaDone(ctx) := true.B
        when(io.step_out.bits.watchdog) {
          watchdogError := true.B
        }
      }.elsewhen(io.step_out.bits.emitCmd) {
        for (entry <- 0 until contextCount) {
          when(ctx === entry.U) {
            assert(cmdQueues(entry).io.enq.ready, "DDA trace command reservation overflow")
            cmdQueues(entry).io.enq.valid := true.B
            cmdQueues(entry).io.enq.bits.tri := io.step_out.bits.tri
            cmdQueues(entry).io.enq.bits.cellExitT := io.step_out.bits.cellExitT
          }
        }
      }
    }
  }

  // Claim one ready context per cycle. Once claimed, the context remains bound
  // to its Trace lane until its result is accepted downstream.
  val claimRr = RegInit(0.U(contextBits.W))
  val queueDeqValid = VecInit(cmdQueues.map(_.io.deq.valid))
  val queueDeqBits = VecInit(cmdQueues.map(_.io.deq.bits))
  val claimRotated = Wire(Vec(contextCount, Bool()))
  for (offset <- 0 until contextCount) {
    val ctx = (claimRr + offset.U)(contextBits - 1, 0)
    claimRotated(offset) := active(ctx) && !claimed(ctx) && !traceStopped(ctx) &&
      (queueDeqValid(ctx) || ddaDone(ctx))
  }
  val claimPickOH = PriorityEncoderOH(claimRotated)
  val claimPickValid = claimRotated.asUInt.orR
  val claimPickId = (OHToUInt(claimPickOH) + claimRr)(contextBits - 1, 0)
  io.trace_claim.valid := claimPickValid
  io.trace_claim.bits.contextId := claimPickId
  io.trace_claim.bits.ray := contexts(claimPickId).ray
  io.trace_claim.bits.meta := contexts(claimPickId).meta
  when(io.trace_claim.fire) {
    claimed(claimPickId) := true.B
    claimRr := claimPickId + 1.U
  }

  for (lane <- 0 until traceLanes) {
    val ctx = io.trace_cmd_context(lane)
    io.trace_cmd(lane).valid := io.trace_cmd_active(lane) && queueDeqValid(ctx)
    io.trace_cmd(lane).bits := queueDeqBits(ctx)
    // Trace may finish only after every command produced by DDA has crossed
    // this context FIFO.  ddaDone alone can be true while commands remain.
    io.trace_dda_done(lane) := io.trace_cmd_active(lane) && ddaDone(ctx) && !queueDeqValid(ctx)
  }
  for (ctx <- 0 until contextCount) {
    val laneMatch = VecInit((0 until traceLanes).map(lane =>
      io.trace_cmd_active(lane) && io.trace_cmd_context(lane) === ctx.U
    ))
    assert(PopCount(laneMatch) <= 1.U, "A DDA context is bound to multiple Trace lanes")
    cmdQueues(ctx).io.deq.ready := (0 until traceLanes).map(lane =>
      laneMatch(lane) && io.trace_cmd(lane).ready
    ).reduce(_ || _)
  }

  for (worker <- 0 until numWorkers) {
    when(io.trace_stop(worker).valid) {
      val ctx = io.trace_stop(worker).bits
      assert(active(ctx) && claimed(ctx), "Trace stopped an inactive or unclaimed context")
      traceStopped(ctx) := true.B
    }
    when(io.trace_release(worker).valid) {
      val ctx = io.trace_release(worker).bits
      assert(active(ctx) && claimed(ctx), "Trace released an inactive or unclaimed context")
      resultAccepted(ctx) := true.B
      val stepReturnsNow = returningContext && returningContextId === ctx
      when(!stepInFlight(ctx) || stepReturnsNow) {
        active(ctx) := false.B
        claimed(ctx) := false.B
        ddaDone(ctx) := false.B
        traceStopped(ctx) := false.B
        resultAccepted(ctx) := false.B
      }
    }
  }
}
