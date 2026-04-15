import SDF.{SdfMemDPI, SdfMemWriteIO, SdfPE}
import chisel3._
import chisel3.util._
import raytrace_utils._

class SdfTop extends Module {
  val c = SdfPeConfig()
  val io = IO(new Bundle {
    val grid_min = Input(new Vec3(c.cfg))
    val inv_voxel = Input(new Vec3(c.cfg))

    val in = Flipped(Decoupled(new SdfRayReq(c.cfg, c.addrWidth)))
    val out = Decoupled(new SdfRayResp(c.cfg, c.addrWidth))
    
    // SDF memory write port for PS initialization
    val sdf_mem_wr = Flipped(new SdfMemWriteIO)
  })

  val sdfPe = Module(new SdfPE(c))
  val sdfMem = Module(new SdfMemDPI(c.addrWidth, c.cfg.totalWidth))

  sdfPe.io.in <> io.in
  sdfPe.io.grid_min := io.grid_min
  sdfPe.io.inv_voxel := io.inv_voxel
  sdfPe.io.out <> io.out
  sdfPe.io.out_hit.ready := true.B

  // SdfPE 采用固定 1-cycle 内存响应假设，顶层将其绑定到 DPI 存储模型。
  sdfPe.io.sdf_mem_req.ready := true.B
  sdfMem.io.clk := clock
  sdfMem.io.reset := reset
  sdfMem.io.globalIdx := sdfPe.io.sdf_mem_req.bits.globalIdx
  sdfMem.io.localIdx := sdfPe.io.sdf_mem_req.bits.localIdx
  sdfMem.io.en := sdfPe.io.sdf_mem_req.valid

  sdfPe.io.sdf_mem_resp.valid := sdfMem.io.valid
  sdfPe.io.sdf_mem_resp.bits := sdfMem.io.data
  
  // Connect write port
  sdfMem.io.wr <> io.sdf_mem_wr
}

