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
  private val maxInflight = GlobalConfig.ddaTraceSlotCount
  private val completionDepth = GlobalConfig.ddaCompletionQueueDepth
  private val initOutDepth = GlobalConfig.ddaInitQueueDepth

  val io = IO(new Bundle {
    val issue_in = Flipped(Decoupled(new DdaTraversalReq(cfg, addrWidth)))
    val init_in = Decoupled(new DdaTraversalReq(cfg, addrWidth))
    val init_out = Flipped(Decoupled(new DdaContext(cfg, addrWidth)))
    val step_in = Decoupled(new DdaContext(cfg, addrWidth))
    val step_out = Flipped(Decoupled(new DdaStepResult(cfg, addrWidth)))
    val trace_job_out = Decoupled(new DdaTraceJobDesc(cfg, addrWidth, maxTraversalSteps))
    val cmd_write = Valid(new DdaTraceCmdWrite(addrWidth, maxTraversalSteps))
    val slot_release = Vec(GlobalConfig.traceNumWorkers, Flipped(Valid(UInt(traceSlotBits.W))))
  })

  val initOutQ = Module(new Queue(new DdaContext(cfg, addrWidth), initOutDepth))
  val retryQ = Module(new Queue(new DdaContext(cfg, addrWidth), GlobalConfig.ddaRetryQueueDepth))
  val completionQ = Module(new Queue(new DdaContext(cfg, addrWidth), completionDepth))
  val freeSlots = RegInit(VecInit(Seq.fill(maxInflight)(true.B)))
  val slotCmdCounts = RegInit(VecInit(Seq.fill(maxInflight)(0.U(cmdCountW.W))))
  val initIssuedCount = RegInit(0.U(log2Ceil(initOutDepth + 1).W))

  val initPending = initIssuedCount - initOutQ.io.deq.fire.asUInt
  val initHasSpace = initPending < initOutDepth.U
  val allocOH = PriorityEncoderOH(freeSlots.asUInt)
  val allocIdx = OHToUInt(allocOH)
  val hasFreeSlot = freeSlots.asUInt.orR

  io.init_in.bits := io.issue_in.bits
  io.init_in.bits.traceSlot := allocIdx
  io.init_in.valid := io.issue_in.valid && hasFreeSlot && initHasSpace
  io.issue_in.ready := hasFreeSlot && initHasSpace && io.init_in.ready
  initOutQ.io.enq <> io.init_out

  val useRetry = retryQ.io.deq.valid
  io.step_in.valid := retryQ.io.deq.valid || initOutQ.io.deq.valid
  io.step_in.bits := Mux(useRetry, retryQ.io.deq.bits, initOutQ.io.deq.bits)
  retryQ.io.deq.ready := io.step_in.ready && retryQ.io.deq.valid
  initOutQ.io.deq.ready := io.step_in.ready && !useRetry

  val slotIdx = io.step_out.bits.ctx.traceSlot
  val routeFinal = io.step_out.valid && io.step_out.bits.done
  val routeRetry = io.step_out.valid && !io.step_out.bits.done
  io.step_out.ready := Mux(io.step_out.bits.done, completionQ.io.enq.ready, retryQ.io.enq.ready)
  completionQ.io.enq.valid := routeFinal
  completionQ.io.enq.bits := io.step_out.bits.ctx
  retryQ.io.enq.valid := routeRetry
  retryQ.io.enq.bits := io.step_out.bits.ctx

  io.cmd_write.valid := io.step_out.fire && io.step_out.bits.emitCmd
  io.cmd_write.bits.slotIdx := slotIdx
  io.cmd_write.bits.cmdIdx := slotCmdCounts(slotIdx)(cmdIdxBits - 1, 0)
  io.cmd_write.bits.tri := io.step_out.bits.tri
  when(io.cmd_write.valid) {
    slotCmdCounts(slotIdx) := slotCmdCounts(slotIdx) + 1.U
  }

  initIssuedCount := initPending + io.issue_in.fire.asUInt
  when(io.issue_in.fire) {
    slotCmdCounts(allocIdx) := 0.U
    freeSlots(allocIdx) := false.B
  }

  val completionSlot = completionQ.io.deq.bits.traceSlot
  io.trace_job_out.valid := completionQ.io.deq.valid
  io.trace_job_out.bits.ray := completionQ.io.deq.bits.ray
  io.trace_job_out.bits.meta := completionQ.io.deq.bits.meta
  io.trace_job_out.bits.cmdCount := slotCmdCounts(completionSlot)
  io.trace_job_out.bits.traceSlot := completionSlot
  completionQ.io.deq.ready := io.trace_job_out.ready

  for (release <- io.slot_release) {
    when(release.valid) {
      assert(!freeSlots(release.bits), "DdaScheduler duplicate or invalid trace slot release")
      freeSlots(release.bits) := true.B
    }
  }
}
