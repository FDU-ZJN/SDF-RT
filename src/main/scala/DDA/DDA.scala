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
    val trace_job_out = Decoupled(new DdaTraceJob(cfg, addrWidth, maxTraversalSteps))
  })

  val scheduler = Module(new DdaScheduler(cfg, addrWidth, maxTraversalSteps))
  val initPE = Module(new DdaInitPE(cfg, addrWidth))
  val stepPE = Module(new DdaStepPipelinePE(cfg, addrWidth, globalRes, subRes, maxTraversalSteps))
  val cmdBuffer = Module(new DdaTraceCmdBuffer(cfg, addrWidth, maxTraversalSteps))

  scheduler.io.issue_in <> io.in
  scheduler.io.trace_job_out <> io.trace_job_out

  scheduler.io.init_in <> initPE.io.in
  scheduler.io.init_out <> initPE.io.out
  scheduler.io.step_in <> stepPE.io.in
  scheduler.io.step_out <> stepPE.io.out

  initPE.io.grid_min := io.grid_min
  initPE.io.inv_sub_voxel := io.inv_sub_voxel

  cmdBuffer.io.clear := scheduler.io.cmd_clear
  cmdBuffer.io.write := scheduler.io.cmd_write
  cmdBuffer.io.readSlot := scheduler.io.cmd_read_slot
  scheduler.io.cmd_read_count := cmdBuffer.io.readCmdCount
  scheduler.io.cmd_read_cmds := cmdBuffer.io.readCmds
}
