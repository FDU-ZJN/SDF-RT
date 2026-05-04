package DDA

import chisel3._
import chisel3.util._
import raytrace_utils._

class DdaScheduler(
  cfg: FloatConfig = FloatConfig.FP32,
  addrWidth: Int = 32,
  maxTraversalSteps: Int = 1024
) extends Module {
  private val traceSlotBits = GlobalConfig.ddaTraceSlotBits
  private val cmdIdxBits = math.max(1, log2Ceil(maxTraversalSteps))
  private val cmdCountW = log2Ceil(maxTraversalSteps + 1)
  private val maxInflight = GlobalConfig.ddaRetryQueueDepth
  private val completionDepth = GlobalConfig.commitQueueDepth
  private val initOutDepth = GlobalConfig.commitQueueDepth

  val io = IO(new Bundle {
    val issue_in = Flipped(Decoupled(new DdaTraversalReq(cfg, addrWidth)))
    val init_in = Decoupled(new DdaTraversalReq(cfg, addrWidth))
    val init_out = Flipped(Decoupled(new DdaContext(cfg, addrWidth)))
    val step_in = Decoupled(new DdaContext(cfg, addrWidth))
    val step_out = Flipped(Decoupled(new DdaStepResult(cfg, addrWidth)))
    val trace_job_out = Decoupled(new DdaTraceJobDesc(cfg, addrWidth, maxTraversalSteps))
    val cmd_write = Valid(new DdaTraceCmdWrite(addrWidth, maxTraversalSteps))
    val slot_release = Flipped(Valid(UInt(traceSlotBits.W)))
  })

  val retryQ = Module(new Queue(new DdaContext(cfg, addrWidth), GlobalConfig.ddaRetryQueueDepth))
  val initOutQ = Module(new Queue(new DdaContext(cfg, addrWidth), initOutDepth))
  val completionQ = Module(new Queue(new DdaContext(cfg, addrWidth), completionDepth))
  val freeSlots = RegInit(VecInit(Seq.fill(maxInflight)(true.B)))
  val slotCmdCounts = RegInit(VecInit(Seq.fill(maxInflight)(0.U(cmdCountW.W))))

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

  val takeRetry = retryQ.io.deq.valid
  io.step_in.valid := retryQ.io.deq.valid || initOutQ.io.deq.valid
  io.step_in.bits := Mux(takeRetry, retryQ.io.deq.bits, initOutQ.io.deq.bits)
  retryQ.io.deq.ready := io.step_in.ready && takeRetry
  initOutQ.io.deq.ready := io.step_in.ready && !takeRetry

  val outFire = io.step_out.fire
  val slotIdx = io.step_out.bits.ctx.traceSlot
  val routeFinal = io.step_out.valid && io.step_out.bits.done
  val routeRetry = io.step_out.valid && !io.step_out.bits.done

  completionQ.io.enq.valid := routeFinal
  completionQ.io.enq.bits := io.step_out.bits.ctx
  when(completionQ.io.enq.valid) {
    assert(completionQ.io.enq.ready, "DDA completionQ overflow")
  }

  io.step_out.ready := true.B

  retryQ.io.enq.valid := routeRetry
  retryQ.io.enq.bits := io.step_out.bits.ctx
  when(retryQ.io.enq.valid) {
    assert(retryQ.io.enq.ready, "DDA retryQ overflow")
  }

  val issueFire = io.issue_in.fire
  initIssuedCount := initPending + issueFire.asUInt

  when(issueFire) {
    slotCmdCounts(allocIdx) := 0.U
    freeSlots(allocIdx) := false.B
  }

  io.cmd_write.valid := outFire && io.step_out.bits.emitCmd
  io.cmd_write.bits.slotIdx := slotIdx
  io.cmd_write.bits.cmdIdx := slotCmdCounts(slotIdx)(cmdIdxBits - 1, 0)
  io.cmd_write.bits.tri := io.step_out.bits.tri
  when(outFire && io.step_out.bits.emitCmd) {
    slotCmdCounts(slotIdx) := slotCmdCounts(slotIdx) + 1.U
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
