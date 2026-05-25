package DDA

import chisel3._
import chisel3.util._
import raytrace_utils._

class DDA(
  cfg: FloatConfig = FloatConfig.FP32,
  addrWidth: Int = 32,
  globalRes: Int = 16,
  subRes: Int = 16,
  maxTraversalSteps: Int = 1024,
  numWorkers: Int = GlobalConfig.ddaNumWorkers
) extends Module {
  require(numWorkers == 2, s"DDA currently supports exactly 2 step workers, got $numWorkers")

  val io = IO(new Bundle {
    val in = Vec(numWorkers, Flipped(Decoupled(new DdaTraversalReq(cfg, addrWidth))))
    val grid_min = Input(new Vec3(cfg))
    val inv_sub_voxel = Input(new Vec3(cfg))
    val trace_job_out = Decoupled(new DdaTraceJobDesc(cfg, addrWidth, maxTraversalSteps))
    val cmd_write = Vec(numWorkers, Valid(new DdaTraceCmdWrite(addrWidth, maxTraversalSteps)))
    val slot_release = Flipped(Valid(UInt(GlobalConfig.ddaTraceSlotBits.W)))
  })

  val scheduler = Module(new DdaScheduler(cfg, addrWidth, maxTraversalSteps, numWorkers))
  val initPEs = Seq.fill(numWorkers)(Module(new DdaInitPE(cfg, addrWidth)))
  val stepPEs = Seq.fill(numWorkers)(Module(new DdaStepPipelinePE(cfg, addrWidth, globalRes, subRes, maxTraversalSteps)))
  val subgridMem = Module(new SubgridMetaMemDPIDualPort(addrWidth, latency = GlobalConfig.subgridMemDpiLatency))

  scheduler.io.issue_in <> io.in
  scheduler.io.trace_job_out <> io.trace_job_out
  scheduler.io.slot_release := io.slot_release
  io.cmd_write := scheduler.io.cmd_write

  for (lane <- 0 until numWorkers) {
    scheduler.io.init_in(lane) <> initPEs(lane).io.in
    scheduler.io.init_out(lane) <> initPEs(lane).io.out
    scheduler.io.step_in(lane) <> stepPEs(lane).io.in
    scheduler.io.step_out(lane) <> stepPEs(lane).io.out
    initPEs(lane).io.grid_min := io.grid_min
    initPEs(lane).io.inv_sub_voxel := io.inv_sub_voxel
  }

  subgridMem.io.clk := clock
  subgridMem.io.reset := reset
  subgridMem.io.globalIdx_a := stepPEs(0).io.subgrid_mem_req.bits.globalIdx
  subgridMem.io.subIdx_a := stepPEs(0).io.subgrid_mem_req.bits.subIdx
  subgridMem.io.en_a := stepPEs(0).io.subgrid_mem_req.valid
  stepPEs(0).io.subgrid_mem_resp.valid := subgridMem.io.valid_a
  stepPEs(0).io.subgrid_mem_resp.bits.triStart := subgridMem.io.triStart_a
  stepPEs(0).io.subgrid_mem_resp.bits.triCount := subgridMem.io.triCount_a

  subgridMem.io.globalIdx_b := stepPEs(1).io.subgrid_mem_req.bits.globalIdx
  subgridMem.io.subIdx_b := stepPEs(1).io.subgrid_mem_req.bits.subIdx
  subgridMem.io.en_b := stepPEs(1).io.subgrid_mem_req.valid
  stepPEs(1).io.subgrid_mem_resp.valid := subgridMem.io.valid_b
  stepPEs(1).io.subgrid_mem_resp.bits.triStart := subgridMem.io.triStart_b
  stepPEs(1).io.subgrid_mem_resp.bits.triCount := subgridMem.io.triCount_b
}
