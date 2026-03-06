package sdf_rt
import chisel3._
import chisel3.util._
import raytrace_utils._

class TriangleMemWrapper(val c: TriPeConfig) extends Module {
  val io = IO(new Bundle {
    // 与 TriPE 的 mem_req 对接
    val req  = Flipped(Decoupled(UInt(c.addrWidth.W)))
    // 与 TriPE 的 mem_resp 对接
    val resp = Decoupled(new TriangleBlock(c))
  })

  // 1. 实例化底层的 BlackBox DPI 模块
  val dpi_mem = Module(new TriangleMemDPI(c))

  dpi_mem.io.clk   := clock
  dpi_mem.io.reset := reset
  io.req.ready := true.B
  dpi_mem.io.addr := io.req.bits
  dpi_mem.io.en   := io.req.valid


  val block_data = Wire(new TriangleBlock(c))
  val bitsPerTri = 3 * 3 * c.cfg.totalWidth
  for(i <- 0 until  c.numPEs) {
    val hi = bitsPerTri*(i+1) - 1
    val lo = bitsPerTri*i
    val triBits = dpi_mem.io.data(hi, lo)
    block_data.tris(i).v0.x := triBits(31, 0)
    block_data.tris(i).v0.y := triBits(63, 32)
    block_data.tris(i).v0.z := triBits(95, 64)
    block_data.tris(i).v1.x := triBits(127, 96)
    block_data.tris(i).v1.y := triBits(159, 128)
    block_data.tris(i).v1.z := triBits(191, 160)
    block_data.tris(i).v2.x := triBits(223, 192)
    block_data.tris(i).v2.y := triBits(255, 224)
    block_data.tris(i).v2.z := triBits(287, 256)
    block_data.tris(i).id := dpi_mem.io.addr_q + i.U(c.addrWidth.W)
    block_data.mask(i) := dpi_mem.io.valid
  }

  io.resp.valid := dpi_mem.io.valid
  io.resp.bits  := block_data
}
