package SDF

import chisel3._
import chisel3.util._
import DDA.DdaInitPE
import raytrace_utils._

class SdfStage(cfg: FloatConfig, addrWidth: Int) extends Module {
  private val peCfg = SdfPeConfig(cfg = cfg)

  val io = IO(new Bundle {
    val grid_min = Input(new Vec3(cfg))
    val inv_voxel = Input(new Vec3(cfg))
    val inv_sub_voxel = Input(new Vec3(cfg))
    val issue_in = Flipped(Decoupled(UInt(GlobalConfig.frontendContextBits.W)))
    val sdf_context_id = Output(UInt(GlobalConfig.frontendContextBits.W))
    val sdf_immutable = Input(new ImmutableRayRecord(cfg, addrWidth))
    val sdf_state = Input(new FrontendSdfState(cfg))
    val state_write = Valid(new FrontendStateWrite(cfg))
    val final_context_id = Output(UInt(GlobalConfig.frontendContextBits.W))
    val final_immutable = Input(new ImmutableRayRecord(cfg, addrWidth))
    val final_state = Input(new FrontendSdfState(cfg))
    val miss_out = Decoupled(new FrontendSdfResult(cfg))
    val dda_context_id = Flipped(Decoupled(UInt(GlobalConfig.ddaTraceSlotBits.W)))
    val dda_context_out = Decoupled(new BackendContextInit(cfg, addrWidth))
    val sdf_mem_wr = Flipped(new SdfMemWriteIO)
  })

  val scheduler = Module(new SdfSchedulerUnit(cfg, addrWidth))
  val sdfPE = Module(new SdfPE(peCfg))
  val sdfMem = Module(new SdfMemDPI(addrWidth, cfg.totalWidth, latency = GlobalConfig.sdfMemDpiLatency))
  val ddaInit = Module(new DdaInitPE(cfg, addrWidth))

  scheduler.io.issue_in <> io.issue_in
  io.sdf_context_id := scheduler.io.context_id
  scheduler.io.immutable := io.sdf_immutable
  scheduler.io.sdf_state := io.sdf_state
  scheduler.io.final_state := io.final_state
  io.state_write := scheduler.io.state_write
  sdfPE.io.in <> scheduler.io.pe_in
  scheduler.io.pe_out <> sdfPE.io.out
  sdfPE.io.grid_min := io.grid_min
  sdfPE.io.inv_voxel := io.inv_voxel

  sdfMem.io.clk := clock
  sdfMem.io.reset := reset
  sdfMem.io.globalIdx := sdfPE.io.sdf_mem_req.bits.globalIdx
  sdfMem.io.localIdx := sdfPE.io.sdf_mem_req.bits.localIdx
  sdfMem.io.en := sdfPE.io.sdf_mem_req.fire
  sdfPE.io.sdf_mem_req.ready := true.B
  sdfPE.io.sdf_mem_resp.valid := sdfMem.io.valid
  sdfPE.io.sdf_mem_resp.bits := sdfMem.io.data
  sdfMem.io.wr <> io.sdf_mem_wr

  val finalOut = scheduler.io.out
  io.final_context_id := scheduler.io.final_context_id
  io.miss_out.valid := finalOut.valid && !finalOut.bits.hit
  io.miss_out.bits := finalOut.bits

  ddaInit.io.in.valid := finalOut.valid && finalOut.bits.hit && io.dda_context_id.valid
  ddaInit.io.in.bits.frontendContextId := finalOut.bits.contextId
  ddaInit.io.in.bits.ray.origin := io.final_immutable.origin
  ddaInit.io.in.bits.ray.dir := io.final_immutable.dir
  ddaInit.io.in.bits.ray.dist := finalOut.bits.dist
  ddaInit.io.in.bits.meta := io.final_immutable.meta
  ddaInit.io.in.bits.traceSlot := io.dda_context_id.bits
  finalOut.ready := Mux(finalOut.bits.hit,
    ddaInit.io.in.ready && io.dda_context_id.valid,
    io.miss_out.ready)
  io.dda_context_id.ready := finalOut.valid && finalOut.bits.hit && ddaInit.io.in.ready
  ddaInit.io.grid_min := io.grid_min
  ddaInit.io.inv_sub_voxel := io.inv_sub_voxel
  io.dda_context_out <> ddaInit.io.out
}
