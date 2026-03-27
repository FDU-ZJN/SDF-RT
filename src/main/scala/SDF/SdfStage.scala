package SDF

import chisel3._
import chisel3.util._
import raytrace_utils._

class SdfStage(cfg: FloatConfig, addrWidth: Int) extends Module {
  private val peCfg = SdfPeConfig(cfg = cfg, addrWidth = addrWidth)

  val io = IO(new Bundle {
    val setup_valid = Input(Bool())
    val setup_origin = Input(new Vec3(cfg))
    val setup_grid_min = Input(new Vec3(cfg))
    val setup_grid_max = Input(new Vec3(cfg))
    val setup_finish = Output(Bool())
    val grid_min = Output(new Vec3(cfg))
    val inv_voxel = Output(new Vec3(cfg))

    val issue_in = Flipped(Decoupled(new RayIssue(cfg, addrWidth)))

    val out_rgb = Output(new Vec3(cfg))
    val out_meta = Output(new RayMeta(addrWidth))
    val out_hit = Output(Bool())
    val out_ray = Output(new Ray(cfg))
    val out_valid = Output(Bool())
  })

  val setupUnit = Module(new SdfSetupUnit(cfg, peCfg))
  val scheduler = Module(new SdfSchedulerUnit(cfg, addrWidth, peCfg.maxSteps))
  val sdfPE = Module(new SdfPE(peCfg))
  val sdfMem = Module(new SdfMemDPI(addrWidth, cfg.totalWidth))

  setupUnit.io.setup_valid := io.setup_valid
  setupUnit.io.setup_origin := io.setup_origin
  setupUnit.io.setup_grid_min := io.setup_grid_min
  setupUnit.io.setup_grid_max := io.setup_grid_max
  io.setup_finish := setupUnit.io.setup_finish
  io.grid_min := setupUnit.io.gridMin
  io.inv_voxel := setupUnit.io.invVoxel

  scheduler.io.issue_in <> io.issue_in

  sdfPE.io.in <> scheduler.io.pe_in
  scheduler.io.pe_out_miss <> sdfPE.io.out
  scheduler.io.pe_out_hit <> sdfPE.io.out_hit

  sdfPE.io.grid_min := setupUnit.io.gridMin
  sdfPE.io.inv_voxel := setupUnit.io.invVoxel

  sdfMem.io.clk := clock
  sdfMem.io.reset := reset
  sdfMem.io.globalIdx := sdfPE.io.sdf_mem_req.bits.globalIdx
  sdfMem.io.localIdx := sdfPE.io.sdf_mem_req.bits.localIdx
  sdfMem.io.en := sdfPE.io.sdf_mem_req.fire
  sdfPE.io.sdf_mem_req.ready := true.B
  sdfPE.io.sdf_mem_resp.valid := sdfMem.io.valid
  sdfPE.io.sdf_mem_resp.bits := sdfMem.io.data

  io.out_rgb := scheduler.io.out_rgb
  io.out_meta := scheduler.io.out_meta
  io.out_hit := scheduler.io.out_hit
  io.out_ray := scheduler.io.out_ray
  io.out_valid := scheduler.io.out_valid
}
