package SDF

import chisel3._
import chisel3.util._
import raytrace_utils._

class SdfStage(cfg: FloatConfig, addrWidth: Int) extends Module {
  private val peCfg = SdfPeConfig(cfg = cfg)
  private val numWorkers = GlobalConfig.sdfStepNumWorkers
  require(numWorkers == 2, s"SdfStage currently supports exactly 2 workers, got $numWorkers")

  val io = IO(new Bundle {
    val grid_min = Input(new Vec3(cfg))
    val inv_voxel = Input(new Vec3(cfg))

    val issue_in = Vec(numWorkers, Flipped(Decoupled(new RayIssue(cfg, addrWidth))))

    val out = Vec(numWorkers, Decoupled(new SdfRayResp(cfg, addrWidth)))
    
    // SDF memory write port for PS initialization
    val sdf_mem_wr = Flipped(new SdfMemWriteIO)
  })

  val scheduler = Module(new SdfSchedulerUnit(cfg, addrWidth, peCfg.maxSteps, numWorkers))
  val sdfPEs = Seq.fill(numWorkers)(Module(new SdfPE(peCfg)))
  val sdfMem = Module(new SdfMem2R(addrWidth, cfg.totalWidth, latency = GlobalConfig.sdfMemDpiLatency))

  scheduler.io.issue_in <> io.issue_in

  for (lane <- 0 until numWorkers) {
    sdfPEs(lane).io.in <> scheduler.io.pe_in(lane)
    scheduler.io.pe_out_miss(lane) <> sdfPEs(lane).io.out
    scheduler.io.pe_out_hit(lane) <> sdfPEs(lane).io.out_hit
    sdfPEs(lane).io.grid_min := io.grid_min
    sdfPEs(lane).io.inv_voxel := io.inv_voxel
  }

  sdfMem.io.clk := clock
  sdfMem.io.reset := reset
  for (lane <- 0 until numWorkers) {
    sdfMem.io.globalIdx(lane) := sdfPEs(lane).io.sdf_mem_req.bits.globalIdx
    sdfMem.io.localIdx(lane) := sdfPEs(lane).io.sdf_mem_req.bits.localIdx
    sdfMem.io.en(lane) := sdfPEs(lane).io.sdf_mem_req.fire
    sdfPEs(lane).io.sdf_mem_req.ready := true.B
    sdfPEs(lane).io.sdf_mem_resp.valid := sdfMem.io.valid(lane)
    sdfPEs(lane).io.sdf_mem_resp.bits := sdfMem.io.data(lane)
  }
  
  // Connect write port
  sdfMem.io.wr <> io.sdf_mem_wr

  io.out <> scheduler.io.out
}
