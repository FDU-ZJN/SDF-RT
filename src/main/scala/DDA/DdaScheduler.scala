package DDA

import chisel3._
import chisel3.util._
import raytrace_utils._

class DdaScheduler(
  cfg: FloatConfig = FloatConfig.FP32,
  addrWidth: Int = 32,
  maxTraversalSteps: Int = 1024
) extends Module {
  private val slotBits = GlobalConfig.slotBits
  private val cmdIdxBits = math.max(1, log2Ceil(maxTraversalSteps))
  private val cmdCountW = log2Ceil(maxTraversalSteps + 1)
  private val maxInflight = GlobalConfig.ddaRetryQueueDepth
  private val inflightW = math.max(1, log2Ceil(maxInflight + 1))
  private val completionDepth = GlobalConfig.commitQueueDepth
  private val initOutDepth = GlobalConfig.commitQueueDepth

  val io = IO(new Bundle {
    val issue_in = Flipped(Decoupled(new DdaTraversalReq(cfg, addrWidth)))
    val init_in = Decoupled(new DdaTraversalReq(cfg, addrWidth))
    val init_out = Flipped(Decoupled(new DdaContext(cfg, addrWidth)))
    val step_in = Decoupled(new DdaContext(cfg, addrWidth))
    val step_out = Flipped(Decoupled(new DdaStepResult(cfg, addrWidth)))
    val trace_job_out = Decoupled(new DdaTraceJob(cfg, addrWidth, maxTraversalSteps))

    val cmd_clear = Valid(UInt(slotBits.W))
    val cmd_write = Valid(new DdaTraceCmdWrite(addrWidth, maxTraversalSteps))
    val cmd_read_slot = Output(UInt(slotBits.W))
    val cmd_read_count = Input(UInt(log2Ceil(maxTraversalSteps + 1).W))
    val cmd_read_cmds = Input(Vec(maxTraversalSteps, new TriBatch(addrWidth)))
  })

  val retryQ = Module(new Queue(new DdaContext(cfg, addrWidth), GlobalConfig.ddaRetryQueueDepth))
  val initOutQ = Module(new Queue(new DdaContext(cfg, addrWidth), initOutDepth))
  val completionQ = Module(new Queue(new DdaContext(cfg, addrWidth), completionDepth))
  val inflightCount = RegInit(0.U(inflightW.W))
  val slotCmdCounts = RegInit(VecInit(Seq.fill(GlobalConfig.commitQueueDepth)(0.U(cmdCountW.W))))

  val initIssuedCount = RegInit(0.U(log2Ceil(initOutDepth + 1).W))
  val initPending = initIssuedCount - initOutQ.io.deq.fire.asUInt
  val initHasSpace = initPending < initOutDepth.U
  val hasFreeSlot = inflightCount < maxInflight.U

  io.init_in.valid := io.issue_in.valid && hasFreeSlot && initHasSpace
  io.init_in.bits := io.issue_in.bits
  io.issue_in.ready := hasFreeSlot && initHasSpace && io.init_in.ready

  initOutQ.io.enq <> io.init_out

  val takeRetry = retryQ.io.deq.valid
  io.step_in.valid := retryQ.io.deq.valid || initOutQ.io.deq.valid
  io.step_in.bits := Mux(takeRetry, retryQ.io.deq.bits, initOutQ.io.deq.bits)
  retryQ.io.deq.ready := io.step_in.ready && takeRetry
  initOutQ.io.deq.ready := io.step_in.ready && !takeRetry

  val outFire = io.step_out.fire
  val slotIdx = io.step_out.bits.ctx.meta.slotId(slotBits - 1, 0)
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
  val retireFire = io.trace_job_out.fire
  val inflightInc = issueFire.asUInt
  val inflightDec = retireFire.asUInt
  inflightCount := inflightCount + inflightInc - inflightDec
  initIssuedCount := initPending + issueFire.asUInt

  io.cmd_clear.valid := issueFire
  io.cmd_clear.bits := io.issue_in.bits.meta.slotId(slotBits - 1, 0)
  when(issueFire) {
    slotCmdCounts(io.issue_in.bits.meta.slotId(slotBits - 1, 0)) := 0.U
  }

  io.cmd_write.valid := outFire && io.step_out.bits.emitCmd
  io.cmd_write.bits.slotIdx := slotIdx
  io.cmd_write.bits.cmdIdx := slotCmdCounts(slotIdx)(cmdIdxBits - 1, 0)
  io.cmd_write.bits.tri := io.step_out.bits.tri
  when(outFire && io.step_out.bits.emitCmd) {
    slotCmdCounts(slotIdx) := slotCmdCounts(slotIdx) + 1.U
  }

  io.cmd_read_slot := completionQ.io.deq.bits.meta.slotId(slotBits - 1, 0)

  io.trace_job_out.valid := completionQ.io.deq.valid
  io.trace_job_out.bits.ray := completionQ.io.deq.bits.ray
  io.trace_job_out.bits.meta := completionQ.io.deq.bits.meta
  io.trace_job_out.bits.cmdCount := slotCmdCounts(completionQ.io.deq.bits.meta.slotId(slotBits - 1, 0))
  io.trace_job_out.bits.cmds := io.cmd_read_cmds
  completionQ.io.deq.ready := io.trace_job_out.ready
}
