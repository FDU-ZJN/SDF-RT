package Backend

import chisel3._
import chisel3.util._
import DDA.DDA
import Trace.{TraceController, TriMemDmaReadData}
import raytrace_utils._

/** Backend boundary for context ownership, DDA traversal, and triangle trace. */
class Backend(
  c: TriPeConfig = TriPeConfig(),
  globalRes: Int = GlobalConfig.GlobalDdaRes,
  subRes: Int = GlobalConfig.SubDdaRes,
  maxTraversalSteps: Int = GlobalConfig.ddaMaxSteps
) extends Module {
  private val numWorkers = GlobalConfig.traceNumWorkers

  val io = IO(new Bundle {
    val init_alloc = Decoupled(UInt(GlobalConfig.ddaTraceSlotBits.W))
    val context_in = Flipped(Decoupled(new BackendContextInit(c.cfg, c.addrWidth)))
    val result_out = Vec(numWorkers, Decoupled(new TraceResult(c.cfg, c.addrWidth)))
    val watchdog_error = Output(Bool())

    val triangleBaseAddress = Input(UInt(64.W))
    val triDmaCmd = Decoupled(UInt(80.W))
    val triDmaData = Flipped(Decoupled(new TriMemDmaReadData(128)))
    val triDmaStatus = Flipped(Decoupled(UInt(8.W)))
    val triDmaReadError = Output(Bool())
    val triDmaMalformedLineCount = Output(UInt(32.W))
    val triDmaStatusErrorCount = Output(UInt(32.W))
    val triDmaTagMismatchCount = Output(UInt(32.W))
    val triAxiIssuedCount = Output(UInt(32.W))
    val triAxiCompletedCount = Output(UInt(32.W))
    val triAxiOutstandingCount = Output(UInt(8.W))
  })

  val contexts = Module(new ContextManager(c.cfg, c.addrWidth))
  val dda = Module(new DDA(c.cfg, c.addrWidth, globalRes, subRes, maxTraversalSteps))
  val trace = Module(new TraceController(c))

  io.init_alloc <> contexts.io.init_alloc
  contexts.io.context_in <> io.context_in
  dda.io.in <> contexts.io.step_in
  contexts.io.step_out <> dda.io.out

  trace.io.trace_claim <> contexts.io.trace_claim
  contexts.io.trace_cmd_context := trace.io.trace_cmd_context
  contexts.io.trace_cmd_active := trace.io.trace_cmd_active
  for (lane <- 0 until GlobalConfig.traceNumWorkers * 2) {
    trace.io.trace_cmd(lane) <> contexts.io.trace_cmd(lane)
  }
  trace.io.trace_dda_done := contexts.io.trace_dda_done
  contexts.io.trace_stop := trace.io.trace_stop
  contexts.io.trace_release := trace.io.trace_release
  io.watchdog_error := contexts.io.watchdog_error

  trace.io.triangleBaseAddress := io.triangleBaseAddress
  io.triDmaCmd <> trace.io.triDmaCmd
  trace.io.triDmaData <> io.triDmaData
  trace.io.triDmaStatus <> io.triDmaStatus
  io.triDmaReadError := trace.io.triDmaReadError
  io.triDmaMalformedLineCount := trace.io.triDmaMalformedLineCount
  io.triDmaStatusErrorCount := trace.io.triDmaStatusErrorCount
  io.triDmaTagMismatchCount := trace.io.triDmaTagMismatchCount
  io.triAxiIssuedCount := trace.io.triAxiIssuedCount
  io.triAxiCompletedCount := trace.io.triAxiCompletedCount
  io.triAxiOutstandingCount := trace.io.triAxiOutstandingCount

  for (worker <- 0 until numWorkers) {
    io.result_out(worker) <> trace.io.result_out(worker)
  }
}
