package SDF

import chisel3._
import chisel3.util._
import raytrace_utils._

class SdfStage(cfg: FloatConfig, addrWidth: Int) extends Module {
  private val peCfg = SdfPeConfig(cfg = cfg)
  private val numWorkers = GlobalConfig.sdfStepNumWorkers
  require(numWorkers == 2, s"SdfStage dual-port SDF memory expects sdfStepNumWorkers=2, got $numWorkers")

  val io = IO(new Bundle {
    val grid_min = Input(new Vec3(cfg))
    val inv_voxel = Input(new Vec3(cfg))

    val issue_in = Vec(numWorkers, Flipped(Decoupled(new RayIssue(cfg, addrWidth))))

    val out_rgb = Output(new Vec3(cfg))
    val out_meta = Output(new RayMeta(addrWidth))
    val out_hit = Output(Bool())
    val out_ray = Output(new Ray(cfg))
    val out_valid = Output(Bool())
    
    // SDF memory write port for PS initialization
    val sdf_mem_wr = Flipped(new SdfMemWriteIO)
  })

  val scheduler = Module(new SdfSchedulerUnit(cfg, addrWidth, peCfg.maxSteps, numWorkers))
  val sdfPEs = Seq.fill(numWorkers)(Module(new SdfPE(peCfg)))
  val sdfMem = Module(new SdfMem2R(addrWidth, cfg.totalWidth, latency = GlobalConfig.sdfMemDpiLatency))

  scheduler.io.issue_in <> io.issue_in

  for (i <- 0 until numWorkers) {
    sdfPEs(i).io.in <> scheduler.io.pe_in(i)
    scheduler.io.pe_out_miss(i) <> sdfPEs(i).io.out
    scheduler.io.pe_out_hit(i) <> sdfPEs(i).io.out_hit

    sdfPEs(i).io.grid_min := io.grid_min
    sdfPEs(i).io.inv_voxel := io.inv_voxel
  }

  sdfMem.io.clk := clock
  sdfMem.io.reset := reset
  for (i <- 0 until numWorkers) {
    sdfMem.io.globalIdx(i) := sdfPEs(i).io.sdf_mem_req.bits.globalIdx
    sdfMem.io.localIdx(i) := sdfPEs(i).io.sdf_mem_req.bits.localIdx
    sdfMem.io.en(i) := sdfPEs(i).io.sdf_mem_req.fire
    sdfPEs(i).io.sdf_mem_req.ready := true.B
    sdfPEs(i).io.sdf_mem_resp.valid := sdfMem.io.valid(i)
    sdfPEs(i).io.sdf_mem_resp.bits := sdfMem.io.data(i)
  }
  
  // Connect write port
  sdfMem.io.wr <> io.sdf_mem_wr

  io.out_rgb := scheduler.io.out_rgb
  io.out_meta := scheduler.io.out_meta
  io.out_hit := scheduler.io.out_hit
  io.out_ray := scheduler.io.out_ray
  io.out_valid := scheduler.io.out_valid
}
