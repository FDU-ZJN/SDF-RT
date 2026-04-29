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
  private val slotBits = GlobalConfig.slotBits
  private val cmdIdxBits = math.max(1, log2Ceil(maxTraversalSteps))
  private val cmdCountW = log2Ceil(maxTraversalSteps + 1)
  private val maxInflight = GlobalConfig.ddaRetryQueueDepth
  private val inflightW = math.max(1, log2Ceil(maxInflight + 1))
  private val completionDepth = math.max(2, GlobalConfig.ddaFinalQueueDepth)

  val io = IO(new Bundle {
    val issue_in = Flipped(Decoupled(new DdaTraversalReq(cfg, addrWidth)))
    val pe_in = Vec(numWorkers, Decoupled(new DdaContext(cfg, addrWidth)))
    val pe_out = Flipped(Vec(numWorkers, Decoupled(new DdaStepResult(cfg, addrWidth))))
    val trace_job_out = Decoupled(new DdaTraceJob(cfg, addrWidth, maxTraversalSteps))

    val cmd_clear = Valid(UInt(slotBits.W))
    val cmd_write = Valid(new DdaTraceCmdWrite(addrWidth, maxTraversalSteps))
    val cmd_read_slot = Output(UInt(slotBits.W))
    val cmd_read_count = Input(UInt(log2Ceil(maxTraversalSteps + 1).W))
    val cmd_read_cmds = Input(Vec(maxTraversalSteps, new TriBatch(addrWidth)))
  })

  val retryQ = Module(new Queue(new DdaContext(cfg, addrWidth), GlobalConfig.ddaRetryQueueDepth))
  val completionQ = Module(new Queue(new DdaContext(cfg, addrWidth), completionDepth))
  val inflightCount = RegInit(0.U(inflightW.W))
  val slotCmdCounts = RegInit(VecInit(Seq.fill(GlobalConfig.commitQueueDepth)(0.U(cmdCountW.W))))

  val newCtx = Wire(new DdaContext(cfg, addrWidth))
  newCtx := 0.U.asTypeOf(new DdaContext(cfg, addrWidth))
  newCtx.ray := io.issue_in.bits.ray
  newCtx.meta := io.issue_in.bits.meta
  newCtx.reverseTraversal := io.issue_in.bits.reverseTraversal
  newCtx.initialized := false.B

  val hasFreeSlot = inflightCount < maxInflight.U
  val workerReady = Wire(Vec(numWorkers, Bool()))
  for (i <- 0 until numWorkers) {
    workerReady(i) := io.pe_in(i).ready
    io.pe_in(i).valid := false.B
    io.pe_in(i).bits := 0.U.asTypeOf(new DdaContext(cfg, addrWidth))
  }
  val hasReadyWorker = workerReady.asUInt.orR
  val workerSelOH = PriorityEncoderOH(workerReady)

  val canIssue = io.issue_in.valid && hasFreeSlot && hasReadyWorker
  val canRetry = retryQ.io.deq.valid && hasReadyWorker && !canIssue

  for (i <- 0 until numWorkers) {
    when(workerSelOH(i) && (canIssue || canRetry)) {
      io.pe_in(i).valid := true.B
      io.pe_in(i).bits := Mux(canIssue, newCtx, retryQ.io.deq.bits)
    }
  }

  io.issue_in.ready := canIssue
  retryQ.io.deq.ready := canRetry

  val outArb = Module(new RRArbiter(new DdaStepResult(cfg, addrWidth), numWorkers))
  for (i <- 0 until numWorkers) {
    outArb.io.in(i) <> io.pe_out(i)
  }

  val outFire = outArb.io.out.fire
  val slotIdx = outArb.io.out.bits.ctx.meta.slotId(slotBits - 1, 0)
  val routeFinal = outArb.io.out.valid && outArb.io.out.bits.done
  val routeRetry = outArb.io.out.valid && !outArb.io.out.bits.done

  completionQ.io.enq.valid := routeFinal
  completionQ.io.enq.bits := outArb.io.out.bits.ctx
  when(completionQ.io.enq.valid) {
    assert(completionQ.io.enq.ready, "DDA completionQ overflow")
  }

  outArb.io.out.ready := Mux(outArb.io.out.bits.done, completionQ.io.enq.ready, retryQ.io.enq.ready)

  retryQ.io.enq.valid := routeRetry
  retryQ.io.enq.bits := outArb.io.out.bits.ctx
  when(retryQ.io.enq.valid) {
    assert(retryQ.io.enq.ready, "DDA retryQ overflow")
  }

  val issueFire = io.issue_in.fire
  val retireFire = io.trace_job_out.fire
  val inflightInc = issueFire.asUInt
  val inflightDec = retireFire.asUInt
  inflightCount := inflightCount + inflightInc - inflightDec

  io.cmd_clear.valid := issueFire
  io.cmd_clear.bits := io.issue_in.bits.meta.slotId(slotBits - 1, 0)
  when(issueFire) {
    slotCmdCounts(io.issue_in.bits.meta.slotId(slotBits - 1, 0)) := 0.U
  }

  io.cmd_write.valid := outFire && outArb.io.out.bits.emitCmd
  io.cmd_write.bits.slotIdx := slotIdx
  io.cmd_write.bits.cmdIdx := slotCmdCounts(slotIdx)(cmdIdxBits - 1, 0)
  io.cmd_write.bits.tri := outArb.io.out.bits.tri
  when(outFire && outArb.io.out.bits.emitCmd) {
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
