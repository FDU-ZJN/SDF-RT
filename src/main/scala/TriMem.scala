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

  // 2. 请求侧逻辑 (TriPE -> DPI)
  // 由于 DPI 是一周期固定延迟，我们假设它总是 ready
  io.req.ready := true.B
  dpi_mem.io.addr := io.req.bits
  dpi_mem.io.en   := io.req.valid

  // 3. 响应侧逻辑 (DPI -> TriPE)
  // 将 DPI 输出的扁平 UInt 转换为 TriangleBlock 结构
  val block_data = dpi_mem.io.data.asTypeOf(new TriangleBlock(c))

  // 处理一周期延迟后的有效信号
  io.resp.valid := dpi_mem.io.valid
  io.resp.bits  := block_data
}