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
  require(numWorkers == 2, s"DdaScheduler currently supports 2 workers, got $numWorkers")
  private val traceSlotBits = GlobalConfig.ddaTraceSlotBits
  private val cmdIdxBits = math.max(1, log2Ceil(maxTraversalSteps))
  private val cmdCountW = log2Ceil(maxTraversalSteps + 1)
  private val maxInflight = GlobalConfig.ddaTraceSlotCount
  private val completionDepth = GlobalConfig.commitQueueDepth
  private val initOutDepth = GlobalConfig.commitQueueDepth
  private val readyDepth = GlobalConfig.ddaRetryQueueDepth
  private val stepOutDepth = 8
  private val cmdWriteDepth = GlobalConfig.commitQueueDepth

  val io = IO(new Bundle {
    val issue_in = Flipped(Decoupled(new DdaTraversalReq(cfg, addrWidth)))
    val init_in = Decoupled(new DdaTraversalReq(cfg, addrWidth))
    val init_out = Flipped(Decoupled(new DdaContext(cfg, addrWidth)))
    val step_in = Vec(numWorkers, Decoupled(new DdaContext(cfg, addrWidth)))
    val step_out = Vec(numWorkers, Flipped(Decoupled(new DdaStepResult(cfg, addrWidth))))
    val trace_job_out = Decoupled(new DdaTraceJobDesc(cfg, addrWidth, maxTraversalSteps))
    val cmd_write = Decoupled(new DdaTraceCmdWrite(addrWidth, maxTraversalSteps))
    val slot_release = Flipped(Valid(UInt(traceSlotBits.W)))
  })

  val initOutQ = Module(new Queue(new DdaContext(cfg, addrWidth), initOutDepth))
  val readyQs = Seq.fill(numWorkers)(Module(new Queue(new DdaContext(cfg, addrWidth), readyDepth)))
  val stepOutQs = Seq.fill(numWorkers)(Module(new Queue(new DdaStepResult(cfg, addrWidth), stepOutDepth)))
  val completionQ = Module(new Queue(new DdaContext(cfg, addrWidth), completionDepth))
  val cmdWriteQ = Module(new Queue(new DdaTraceCmdWrite(addrWidth, maxTraversalSteps), cmdWriteDepth))
  val completionArb = Module(new Arbiter(new DdaContext(cfg, addrWidth), numWorkers))
  val cmdArb = Module(new Arbiter(new DdaTraceCmdWrite(addrWidth, maxTraversalSteps), numWorkers))

  val freeSlots = RegInit(VecInit(Seq.fill(maxInflight)(true.B)))
  val slotCmdCounts = RegInit(VecInit(Seq.fill(maxInflight)(0.U(cmdCountW.W))))
  val initDispatchSel = RegInit(0.U(math.max(1, log2Ceil(numWorkers)).W))

  val initIssuedCount = RegInit(0.U(log2Ceil(initOutDepth + 1).W))
  val initPending = initIssuedCount - initOutQ.io.deq.fire.asUInt
  val initHasSpace = initPending < initOutDepth.U
  val hasFreeSlot = freeSlots.asUInt.orR
  val allocOH = PriorityEncoderOH(freeSlots)
  val allocIdx = OHToUInt(allocOH)

  io.init_in.valid := io.issue_in.valid && hasFreeSlot && initHasSpace
  io.init_in.bits := io.issue_in.bits
  io.init_in.bits.traceSlot := allocIdx
  io.issue_in.ready := hasFreeSlot && initHasSpace && io.init_in.ready
  initOutQ.io.enq <> io.init_out

  val issueFire = io.issue_in.fire
  initIssuedCount := initPending + issueFire.asUInt
  when(issueFire) {
    slotCmdCounts(allocIdx) := 0.U
    freeSlots(allocIdx) := false.B
  }

  val retryValid = Wire(Vec(numWorkers, Bool()))
  val retryReady = Wire(Vec(numWorkers, Bool()))
  retryValid := VecInit(Seq.fill(numWorkers)(false.B))
  retryReady := VecInit(Seq.fill(numWorkers)(false.B))

  val initReadyMask = Wire(Vec(numWorkers, Bool()))
  for (i <- 0 until numWorkers) {
    io.step_in(i) <> readyQs(i).io.deq
    stepOutQs(i).io.enq <> io.step_out(i)
    initReadyMask(i) := readyQs(i).io.enq.ready && !retryValid(i)
  }

  val initCanDispatch = initReadyMask.asUInt.orR
  val initTargetOH = Wire(UInt(numWorkers.W))
  when(initDispatchSel === 0.U) {
    initTargetOH := Mux(initReadyMask(0), "b01".U, Mux(initReadyMask(1), "b10".U, 0.U))
  }.otherwise {
    initTargetOH := Mux(initReadyMask(1), "b10".U, Mux(initReadyMask(0), "b01".U, 0.U))
  }
  val initTargetIdx = OHToUInt(initTargetOH)

  for (i <- 0 until numWorkers) {
    val laneOut = stepOutQs(i).io.deq
    val retryFire = laneOut.valid && !laneOut.bits.done && laneOut.ready
    readyQs(i).io.enq.valid := retryFire || (initOutQ.io.deq.valid && initTargetOH(i) && !retryFire)
    readyQs(i).io.enq.bits := Mux(retryFire, laneOut.bits.ctx, initOutQ.io.deq.bits)
    retryValid(i) := laneOut.valid && !laneOut.bits.done
    retryReady(i) := readyQs(i).io.enq.ready
  }

  initOutQ.io.deq.ready := initOutQ.io.deq.valid && initCanDispatch
  when(initOutQ.io.deq.fire) {
    initDispatchSel := Mux(initTargetIdx === (numWorkers - 1).U, 0.U, initTargetIdx + 1.U)
  }

  for (i <- 0 until numWorkers) {
    val laneOut = stepOutQs(i).io.deq
    completionArb.io.in(i).valid := laneOut.valid && laneOut.bits.done
    completionArb.io.in(i).bits := laneOut.bits.ctx

    cmdArb.io.in(i).valid := laneOut.valid && !laneOut.bits.done && laneOut.bits.emitCmd
    cmdArb.io.in(i).bits.slotIdx := laneOut.bits.ctx.traceSlot
    cmdArb.io.in(i).bits.cmdIdx := slotCmdCounts(laneOut.bits.ctx.traceSlot)(cmdIdxBits - 1, 0)
    cmdArb.io.in(i).bits.tri := laneOut.bits.tri
  }

  completionQ.io.enq <> completionArb.io.out
  when(completionQ.io.enq.valid) {
    assert(completionQ.io.enq.ready, "DDA completionQ overflow")
  }

  val laneCmdReady = Wire(Vec(numWorkers, Bool()))
  for (i <- 0 until numWorkers) {
    val laneOut = stepOutQs(i).io.deq
    laneCmdReady(i) := !cmdArb.io.in(i).valid || cmdArb.io.in(i).ready
    laneOut.ready := Mux(
      laneOut.bits.done,
      completionArb.io.in(i).ready,
      retryReady(i) && laneCmdReady(i)
    )
  }

  cmdWriteQ.io.enq <> cmdArb.io.out
  io.cmd_write <> cmdWriteQ.io.deq
  when(cmdWriteQ.io.enq.fire) {
    slotCmdCounts(cmdWriteQ.io.enq.bits.slotIdx) := slotCmdCounts(cmdWriteQ.io.enq.bits.slotIdx) + 1.U
  }

  io.trace_job_out.valid := completionQ.io.deq.valid
  io.trace_job_out.bits.ray := completionQ.io.deq.bits.ray
  io.trace_job_out.bits.meta := completionQ.io.deq.bits.meta
  io.trace_job_out.bits.cmdCount := slotCmdCounts(completionQ.io.deq.bits.traceSlot)
  io.trace_job_out.bits.traceSlot := completionQ.io.deq.bits.traceSlot
  completionQ.io.deq.ready := io.trace_job_out.ready

  when(io.slot_release.valid) {
    freeSlots(io.slot_release.bits) := true.B
  }
}
