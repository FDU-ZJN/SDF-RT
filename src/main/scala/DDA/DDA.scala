package DDA

import chisel3._
import chisel3.util._
import raytrace_utils._

class DDA(
  cfg: FloatConfig = FloatConfig.FP32,
  addrWidth: Int = 32,
  globalRes: Int = 16,
  subRes: Int = 16,
  maxTraversalSteps: Int = 1024
) extends Module {
  val io = IO(new Bundle {
    val init_alloc = Decoupled(UInt(GlobalConfig.ddaTraceSlotBits.W))
    val context_in = Flipped(Decoupled(new DdaContext(cfg, addrWidth)))
    val trace_claim = Decoupled(new TraceContextDesc(cfg, addrWidth))
    val trace_cmd_context = Input(Vec(GlobalConfig.traceNumWorkers * 2, UInt(GlobalConfig.ddaTraceSlotBits.W)))
    val trace_cmd_active = Input(Vec(GlobalConfig.traceNumWorkers * 2, Bool()))
    val trace_cmd = Vec(GlobalConfig.traceNumWorkers * 2, Decoupled(new TraceCellCmd(cfg, addrWidth)))
    val trace_dda_done = Output(Vec(GlobalConfig.traceNumWorkers * 2, Bool()))
    val trace_stop = Vec(GlobalConfig.traceNumWorkers, Flipped(Valid(UInt(GlobalConfig.ddaTraceSlotBits.W))))
    val trace_release = Vec(GlobalConfig.traceNumWorkers, Flipped(Valid(UInt(GlobalConfig.ddaTraceSlotBits.W))))
    val watchdog_error = Output(Bool())
  })

  val scheduler = Module(new DdaScheduler(cfg, addrWidth, maxTraversalSteps))
  val stepPE = Module(new DdaStepPipelinePE(cfg, addrWidth, globalRes, subRes, maxTraversalSteps))
  val subgridMem = Module(new SubgridMetaMemDPI(addrWidth, latency = GlobalConfig.subgridMemDpiLatency))

  io.init_alloc <> scheduler.io.init_alloc
  scheduler.io.context_in <> io.context_in
  io.trace_claim <> scheduler.io.trace_claim
  scheduler.io.trace_cmd_context := io.trace_cmd_context
  scheduler.io.trace_cmd_active := io.trace_cmd_active
  for (lane <- 0 until GlobalConfig.traceNumWorkers * 2) {
    io.trace_cmd(lane) <> scheduler.io.trace_cmd(lane)
  }
  io.trace_dda_done := scheduler.io.trace_dda_done
  scheduler.io.trace_stop := io.trace_stop
  scheduler.io.trace_release := io.trace_release
  io.watchdog_error := scheduler.io.watchdog_error

  scheduler.io.step_in <> stepPE.io.in
  scheduler.io.step_out <> stepPE.io.out

  subgridMem.io.clk := clock
  subgridMem.io.reset := reset
  subgridMem.io.globalIdx := stepPE.io.subgrid_mem_req.bits.globalIdx
  subgridMem.io.subIdx := stepPE.io.subgrid_mem_req.bits.subIdx
  subgridMem.io.en := stepPE.io.subgrid_mem_req.valid
  stepPE.io.subgrid_mem_resp.valid := subgridMem.io.valid
  stepPE.io.subgrid_mem_resp.bits.triStart := subgridMem.io.triStart
  stepPE.io.subgrid_mem_resp.bits.triCount := subgridMem.io.triCount
}
