package DDA

import chisel3._
import chisel3.util._
import raytrace_utils._

class DdaScheduler(
  cfg: FloatConfig = FloatConfig.FP32,
  addrWidth: Int = 32,
  maxTraversalSteps: Int = 1024,
  numWorkers: Int = GlobalConfig.ddaNumWorkers
) extends Module {
  require(numWorkers == 2, s"DdaScheduler currently supports exactly 2 step workers, got $numWorkers")

  private val traceSlotBits = GlobalConfig.ddaTraceSlotBits
  private val cmdIdxBits = math.max(1, log2Ceil(maxTraversalSteps))
  private val cmdCountW = log2Ceil(maxTraversalSteps + 1)
  private val maxInflight = GlobalConfig.ddaTraceSlotCount
  private val completionDepth = GlobalConfig.commitQueueDepth
  private val initOutDepth = GlobalConfig.commitQueueDepth

  val io = IO(new Bundle {
    val issue_in = Vec(numWorkers, Flipped(Decoupled(new DdaTraversalReq(cfg, addrWidth))))
    val init_in = Vec(numWorkers, Decoupled(new DdaTraversalReq(cfg, addrWidth)))
    val init_out = Vec(numWorkers, Flipped(Decoupled(new DdaContext(cfg, addrWidth))))
    val step_in = Vec(numWorkers, Decoupled(new DdaContext(cfg, addrWidth)))
    val step_out = Vec(numWorkers, Flipped(Decoupled(new DdaStepResult(cfg, addrWidth))))
    val trace_job_out = Decoupled(new DdaTraceJobDesc(cfg, addrWidth, maxTraversalSteps))
    val cmd_write = Vec(numWorkers, Valid(new DdaTraceCmdWrite(addrWidth, maxTraversalSteps)))
    val slot_release = Flipped(Valid(UInt(traceSlotBits.W)))
  })

  val initOutQs = Seq.fill(numWorkers)(Module(new Queue(new DdaContext(cfg, addrWidth), initOutDepth)))
  val laneRetryQ = Seq.fill(numWorkers)(Module(new Queue(new DdaContext(cfg, addrWidth), GlobalConfig.ddaRetryQueueDepth)))
  val laneCompletionQ = Seq.fill(numWorkers)(Module(new Queue(new DdaContext(cfg, addrWidth), completionDepth)))
  val freeSlots = RegInit(VecInit(Seq.fill(maxInflight)(true.B)))
  val slotCmdCounts = RegInit(VecInit(Seq.fill(maxInflight)(0.U(cmdCountW.W))))

  val laneInitIssuedCount = RegInit(VecInit(Seq.fill(numWorkers)(0.U(log2Ceil(initOutDepth + 1).W))))
  val laneInitPending = Wire(Vec(numWorkers, UInt(log2Ceil(initOutDepth + 1).W)))
  val laneInitHasSpace = Wire(Vec(numWorkers, Bool()))
  for (lane <- 0 until numWorkers) {
    laneInitPending(lane) := laneInitIssuedCount(lane) - initOutQs(lane).io.deq.fire.asUInt
    laneInitHasSpace(lane) := laneInitPending(lane) < initOutDepth.U
  }
  val allocOH0 = PriorityEncoderOH(freeSlots.asUInt)
  val freeSlotsAfter0 = freeSlots.asUInt & (~allocOH0).asUInt
  val allocOH1 = PriorityEncoderOH(freeSlotsAfter0)
  val allocIdx0 = OHToUInt(allocOH0)
  val allocIdx1 = OHToUInt(allocOH1)
  val hasFreeSlot0 = freeSlots.asUInt.orR
  val hasFreeSlot1 = freeSlotsAfter0.orR

  for (lane <- 0 until numWorkers) {
    io.init_in(lane).bits := io.issue_in(lane).bits
  }
  io.init_in(0).bits.traceSlot := allocIdx0
  io.init_in(1).bits.traceSlot := allocIdx1

  io.init_in(0).valid := io.issue_in(0).valid && hasFreeSlot0 && laneInitHasSpace(0)
  io.issue_in(0).ready := hasFreeSlot0 && laneInitHasSpace(0) && io.init_in(0).ready
  io.init_in(1).valid := io.issue_in(1).valid && hasFreeSlot1 && laneInitHasSpace(1)
  io.issue_in(1).ready := hasFreeSlot1 && laneInitHasSpace(1) && io.init_in(1).ready

  for (lane <- 0 until numWorkers) {
    initOutQs(lane).io.enq <> io.init_out(lane)
  }

  for (lane <- 0 until numWorkers) {
    val useRetry = laneRetryQ(lane).io.deq.valid
    io.step_in(lane).valid := laneRetryQ(lane).io.deq.valid || initOutQs(lane).io.deq.valid
    io.step_in(lane).bits := Mux(useRetry, laneRetryQ(lane).io.deq.bits, initOutQs(lane).io.deq.bits)
    laneRetryQ(lane).io.deq.ready := io.step_in(lane).ready && laneRetryQ(lane).io.deq.valid
    initOutQs(lane).io.deq.ready := io.step_in(lane).ready && !useRetry
  }

  for (lane <- 0 until numWorkers) {
    val outFire = io.step_out(lane).fire
    val slotIdx = io.step_out(lane).bits.ctx.traceSlot
    val routeFinal = io.step_out(lane).valid && io.step_out(lane).bits.done
    val routeRetry = io.step_out(lane).valid && !io.step_out(lane).bits.done

    io.step_out(lane).ready := true.B

    laneCompletionQ(lane).io.enq.valid := routeFinal
    laneCompletionQ(lane).io.enq.bits := io.step_out(lane).bits.ctx
    when(laneCompletionQ(lane).io.enq.valid) {
      assert(laneCompletionQ(lane).io.enq.ready, s"DDA completionQ lane $lane overflow")
    }

    laneRetryQ(lane).io.enq.valid := routeRetry
    laneRetryQ(lane).io.enq.bits := io.step_out(lane).bits.ctx
    when(laneRetryQ(lane).io.enq.valid) {
      assert(laneRetryQ(lane).io.enq.ready, s"DDA retryQ lane $lane overflow")
    }

    io.cmd_write(lane).valid := outFire && io.step_out(lane).bits.emitCmd
    io.cmd_write(lane).bits.slotIdx := slotIdx
    io.cmd_write(lane).bits.cmdIdx := slotCmdCounts(slotIdx)(cmdIdxBits - 1, 0)
    io.cmd_write(lane).bits.tri := io.step_out(lane).bits.tri
    when(outFire && io.step_out(lane).bits.emitCmd) {
      slotCmdCounts(slotIdx) := slotCmdCounts(slotIdx) + 1.U
    }
  }

  when(io.step_out(0).fire && io.step_out(1).fire) {
    val sameCmdSlot =
      io.step_out(0).bits.emitCmd &&
        io.step_out(1).bits.emitCmd &&
        (io.step_out(0).bits.ctx.traceSlot === io.step_out(1).bits.ctx.traceSlot)
    assert(!sameCmdSlot, "DDA dual-lane cmd writes must not target the same trace slot in one cycle")
  }

  for (lane <- 0 until numWorkers) {
    val issueFire = io.issue_in(lane).fire
    laneInitIssuedCount(lane) := laneInitPending(lane) + issueFire.asUInt
  }

  when(io.issue_in(0).fire) {
    slotCmdCounts(allocIdx0) := 0.U
    freeSlots(allocIdx0) := false.B
  }
  when(io.issue_in(1).fire) {
    slotCmdCounts(allocIdx1) := 0.U
    freeSlots(allocIdx1) := false.B
  }

  val completionArb = Module(new RRArbiter(new DdaContext(cfg, addrWidth), numWorkers))
  for (lane <- 0 until numWorkers) {
    completionArb.io.in(lane) <> laneCompletionQ(lane).io.deq
  }

  val completionSlot = completionArb.io.out.bits.traceSlot
  val completionLane = completionArb.io.chosen
  val completionAddsCmd =
    completionArb.io.out.valid &&
      io.step_out(completionLane).fire &&
      io.step_out(completionLane).bits.done &&
      io.step_out(completionLane).bits.emitCmd &&
      (io.step_out(completionLane).bits.ctx.traceSlot === completionSlot)
  val completionCmdCount =
    slotCmdCounts(completionSlot) + completionAddsCmd.asUInt

  io.trace_job_out.valid := completionArb.io.out.valid
  io.trace_job_out.bits.ray := completionArb.io.out.bits.ray
  io.trace_job_out.bits.meta := completionArb.io.out.bits.meta
  io.trace_job_out.bits.cmdCount := completionCmdCount
  io.trace_job_out.bits.traceSlot := completionSlot
  completionArb.io.out.ready := io.trace_job_out.ready

  when(io.slot_release.valid) {
    assert(!freeSlots(io.slot_release.bits), "DdaScheduler duplicate or invalid trace slot release")
    freeSlots(io.slot_release.bits) := true.B
  }
}
