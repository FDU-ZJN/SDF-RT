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
  val io = IO(new Bundle {
    val in = Flipped(Decoupled(new DdaTraversalReq(cfg, addrWidth)))
    val grid_min = Input(new Vec3(cfg))
    val inv_sub_voxel = Input(new Vec3(cfg))
    val trace_job_out = Decoupled(new DdaTraceJobDesc(cfg, addrWidth, maxTraversalSteps))
    val cmd_write = Decoupled(new DdaTraceCmdWrite(addrWidth, maxTraversalSteps))
    val slot_release = Flipped(Valid(UInt(GlobalConfig.ddaTraceSlotBits.W)))
  })

  val scheduler = Module(new DdaScheduler(cfg, addrWidth, maxTraversalSteps, numWorkers))
  val initPE = Module(new DdaInitPE(cfg, addrWidth))
  val stepPEs = Seq.fill(numWorkers)(Module(new DdaStepPipelinePE(cfg, addrWidth, globalRes, subRes, maxTraversalSteps)))
  val subgridMem = Module(new SubgridMetaMemMultiPort(numWorkers, addrWidth, GlobalConfig.subgridMetaMemNumBanks, subRes))

  scheduler.io.issue_in <> io.in
  scheduler.io.trace_job_out <> io.trace_job_out
  scheduler.io.slot_release := io.slot_release
  io.cmd_write.valid := scheduler.io.cmd_write.valid
  io.cmd_write.bits := scheduler.io.cmd_write.bits
  scheduler.io.cmd_write.ready := io.cmd_write.ready

  scheduler.io.init_in <> initPE.io.in
  scheduler.io.init_out <> initPE.io.out
  for (i <- 0 until numWorkers) {
    scheduler.io.step_in(i) <> stepPEs(i).io.in
    scheduler.io.step_out(i) <> stepPEs(i).io.out
    subgridMem.io.req(i) <> stepPEs(i).io.mem_req
    stepPEs(i).io.mem_resp := subgridMem.io.resp(i)
  }

  initPE.io.grid_min := io.grid_min
  initPE.io.inv_sub_voxel := io.inv_sub_voxel
}
