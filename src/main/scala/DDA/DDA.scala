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
    val in = Flipped(Decoupled(new DdaTraversalState(cfg)))
    val out = Decoupled(new DdaStepResult(cfg, addrWidth))
  })

  val stepPE = Module(new DdaStepPipelinePE(cfg, addrWidth, globalRes, subRes, maxTraversalSteps))
  val subgridMem = Module(new SubgridMetaMemDPI(addrWidth, latency = GlobalConfig.subgridMemDpiLatency))

  stepPE.io.in <> io.in
  io.out <> stepPE.io.out

  subgridMem.io.clk := clock
  subgridMem.io.reset := reset
  subgridMem.io.globalIdx := stepPE.io.subgrid_mem_req.bits.globalIdx
  subgridMem.io.subIdx := stepPE.io.subgrid_mem_req.bits.subIdx
  subgridMem.io.en := stepPE.io.subgrid_mem_req.valid
  stepPE.io.subgrid_mem_resp.valid := subgridMem.io.valid
  stepPE.io.subgrid_mem_resp.bits.triStart := subgridMem.io.triStart
  stepPE.io.subgrid_mem_resp.bits.triCount := subgridMem.io.triCount
}
