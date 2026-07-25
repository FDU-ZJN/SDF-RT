package SDF

import chisel3._
import chisel3.util._
import DDA.DdaInitPE
import raytrace_utils._

class SdfStage(cfg: FloatConfig, addrWidth: Int) extends Module {
  private val peCfg = SdfPeConfig(cfg = cfg)
  private val numWorkers = GlobalConfig.sdfStepNumWorkers
  require(numWorkers == 2, s"SdfStage currently supports exactly 2 workers, got $numWorkers")

  val io = IO(new Bundle {
    val grid_min = Input(new Vec3(cfg))
    val inv_voxel = Input(new Vec3(cfg))
    val inv_sub_voxel = Input(new Vec3(cfg))

    val issue_in = Vec(numWorkers, Flipped(Decoupled(new RayIssue(cfg, addrWidth))))

    val miss_out = Vec(numWorkers, Decoupled(new SdfRayResp(cfg, addrWidth)))
    val dda_context_id = Flipped(Decoupled(UInt(GlobalConfig.ddaTraceSlotBits.W)))
    val dda_context_out = Decoupled(new DdaContext(cfg, addrWidth))
    
    // SDF memory write port for PS initialization
    val sdf_mem_wr = Flipped(new SdfMemWriteIO)
  })

  val scheduler = Module(new SdfSchedulerUnit(cfg, addrWidth, peCfg.maxSteps, numWorkers))
  val sdfPEs = Seq.fill(numWorkers)(Module(new SdfPE(peCfg)))
  val sdfMem = Module(new SdfMem2R(addrWidth, cfg.totalWidth, latency = GlobalConfig.sdfMemDpiLatency))
  val hitArb = Module(new RRArbiter(new SdfRayResp(cfg, addrWidth), numWorkers))
  val ddaInit = Module(new DdaInitPE(cfg, addrWidth))

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

  for (lane <- 0 until numWorkers) {
    val out = scheduler.io.out(lane)
    io.miss_out(lane).valid := out.valid && !out.bits.hit
    io.miss_out(lane).bits := out.bits
    hitArb.io.in(lane).valid := out.valid && out.bits.hit
    hitArb.io.in(lane).bits := out.bits
    out.ready := Mux(out.bits.hit, hitArb.io.in(lane).ready, io.miss_out(lane).ready)
  }

  ddaInit.io.in.valid := hitArb.io.out.valid && io.dda_context_id.valid
  ddaInit.io.in.bits.ray := hitArb.io.out.bits.ray
  ddaInit.io.in.bits.meta := hitArb.io.out.bits.meta
  ddaInit.io.in.bits.traceSlot := io.dda_context_id.bits
  hitArb.io.out.ready := ddaInit.io.in.ready && io.dda_context_id.valid
  io.dda_context_id.ready := hitArb.io.out.valid && ddaInit.io.in.ready
  ddaInit.io.grid_min := io.grid_min
  ddaInit.io.inv_sub_voxel := io.inv_sub_voxel
  io.dda_context_out <> ddaInit.io.out
}
