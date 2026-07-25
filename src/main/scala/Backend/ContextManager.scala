package Backend

import chisel3._
import chisel3.util._
import raytrace_utils._

/** Coordinates context ownership between the DDA engine and Trace workers. */
class ContextManager(
  cfg: FloatConfig = FloatConfig.FP32,
  addrWidth: Int = 32
) extends Module {
  private val contextCount = GlobalConfig.ddaTraceSlotCount
  private val contextBits = GlobalConfig.ddaTraceSlotBits
  private val traceLanes = GlobalConfig.traceNumWorkers * 2
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

  val watchdogError = RegInit(false.B)
  io.watchdog_error := watchdogError

  val slots = Seq.fill(contextCount)(Module(
    new ContextSlot(cfg, addrWidth, GlobalConfig.ddaTraceCmdDepth)
  ))

  val stopNow = Wire(Vec(contextCount, Bool()))
  val releaseNow = Wire(Vec(contextCount, Bool()))
  val initCommit = Wire(Vec(contextCount, Bool()))
  for (ctx <- 0 until contextCount) {
    stopNow(ctx) := io.trace_stop.map(s => s.valid && s.bits === ctx.U).reduce(_ || _)
    releaseNow(ctx) := io.trace_release.map(r => r.valid && r.bits === ctx.U).reduce(_ || _)
    initCommit(ctx) := io.context_in.fire && io.context_in.bits.traceSlot === ctx.U

    slots(ctx).io.reserveInit := false.B
    slots(ctx).io.initCommit := initCommit(ctx)
    slots(ctx).io.initContext := io.context_in.bits
    slots(ctx).io.stepIssue := false.B
    slots(ctx).io.stepReturn := io.step_out.fire &&
      io.step_out.bits.ctx.traceSlot === ctx.U
    slots(ctx).io.stepResult := io.step_out.bits
    slots(ctx).io.claim := false.B
    slots(ctx).io.stop := stopNow(ctx)
    slots(ctx).io.release := releaseNow(ctx)
    slots(ctx).io.cmdOut.ready := false.B
  }

  val freeForInit = VecInit(slots.map(_.io.freeForInit))
  val initAllocOH = PriorityEncoderOH(freeForInit)
  val initAllocId = OHToUInt(initAllocOH)
  io.init_alloc.valid := freeForInit.asUInt.orR
  io.init_alloc.bits := initAllocId
  when(io.init_alloc.fire) {
    for (ctx <- 0 until contextCount) {
      slots(ctx).io.reserveInit := initAllocId === ctx.U
    }
  }

  io.context_in.ready := true.B

  val stepRr = RegInit(0.U(contextBits.W))
  val stepEligible = VecInit(slots.map(_.io.stepEligible))
  val stepRotated = Wire(Vec(contextCount, Bool()))
  for (offset <- 0 until contextCount) {
    val ctx = (stepRr + offset.U)(contextBits - 1, 0)
    stepRotated(offset) := stepEligible(ctx)
  }
  val stepPickOH = PriorityEncoderOH(stepRotated)
  val stepPickId = (OHToUInt(stepPickOH) + stepRr)(contextBits - 1, 0)
  io.step_in.valid := stepRotated.asUInt.orR
  io.step_in.bits := VecInit(slots.map(_.io.context))(stepPickId)
  when(io.step_in.fire) {
    for (ctx <- 0 until contextCount) {
      slots(ctx).io.stepIssue := stepPickId === ctx.U
    }
    stepRr := stepPickId + 1.U
  }

  io.step_out.ready := true.B
  when(VecInit(slots.map(_.io.watchdogPulse)).asUInt.orR) {
    watchdogError := true.B
  }

  val claimRr = RegInit(0.U(contextBits.W))
  val claimEligible = VecInit(slots.map(_.io.claimEligible))
  val claimRotated = Wire(Vec(contextCount, Bool()))
  for (offset <- 0 until contextCount) {
    val ctx = (claimRr + offset.U)(contextBits - 1, 0)
    claimRotated(offset) := claimEligible(ctx)
  }
  val claimPickOH = PriorityEncoderOH(claimRotated)
  val claimPickId = (OHToUInt(claimPickOH) + claimRr)(contextBits - 1, 0)
  val slotContexts = VecInit(slots.map(_.io.context))
  io.trace_claim.valid := claimRotated.asUInt.orR
  io.trace_claim.bits.contextId := claimPickId
  io.trace_claim.bits.ray := slotContexts(claimPickId).ray
  io.trace_claim.bits.meta := slotContexts(claimPickId).meta
  when(io.trace_claim.fire) {
    for (ctx <- 0 until contextCount) {
      slots(ctx).io.claim := claimPickId === ctx.U
    }
    claimRr := claimPickId + 1.U
  }

  val queueDeqValid = VecInit(slots.map(_.io.cmdOut.valid))
  val queueDeqBits = VecInit(slots.map(_.io.cmdOut.bits))
  val slotDdaDone = VecInit(slots.map(_.io.ddaDone))
  for (lane <- 0 until traceLanes) {
    val ctx = io.trace_cmd_context(lane)
    io.trace_cmd(lane).valid := io.trace_cmd_active(lane) && queueDeqValid(ctx)
    io.trace_cmd(lane).bits := queueDeqBits(ctx)
    io.trace_dda_done(lane) := io.trace_cmd_active(lane) &&
      slotDdaDone(ctx) && !queueDeqValid(ctx)
  }

  for (ctx <- 0 until contextCount) {
    val laneMatch = VecInit((0 until traceLanes).map(lane =>
      io.trace_cmd_active(lane) && io.trace_cmd_context(lane) === ctx.U
    ))
    assert(PopCount(laneMatch) <= 1.U,
      "A backend context is bound to multiple Trace lanes")
    slots(ctx).io.cmdOut.ready := (0 until traceLanes).map(lane =>
      laneMatch(lane) && io.trace_cmd(lane).ready
    ).reduce(_ || _)
  }
}
