package Render

import chisel3._
import chisel3.util._
import raytrace_utils._

class RenderStage(cfg: FloatConfig) extends Module {
  val io = IO(new Bundle {
    val in = Flipped(Decoupled(new TraceResult(cfg, cfg.addrWidth)))
    val out = Decoupled(new RenderResult(cfg, cfg.addrWidth))
  })
  // 1. 实例化核心计算单元 (PE)
  val pe = Module(new RenderPE(cfg))
  val mem = Module(new NormalMemDPI(cfg.addrWidth))
  mem.io.clk   := clock
  mem.io.reset := reset

  io.in.ready := true.B

  val zeroNormal = 0.U.asTypeOf(new Vec3(cfg))

  val launchId = RegNext(io.in.bits.hitId)
  val launchMeta = RegNext(io.in.bits.meta)
  val launchHit = RegNext(io.in.bits.hit)

  mem.io.addr := io.in.bits.hitId
  mem.io.en := true.B

  pe.io.in_meta := launchMeta
  pe.io.hit_id := launchId
  pe.io.in_hit := launchHit

  // B. 处理内存返回的数据并送回 PE
  // NormalMemDPI 返回的是 96 位原始数据，我们需要将其解包为 Vec3 浮点向量
  val normal_from_mem = Wire(new Vec3(cfg))
  normal_from_mem.x := mem.io.data(31, 0)
  normal_from_mem.y := mem.io.data(63, 32)
  normal_from_mem.z := mem.io.data(95, 64)

  pe.io.in_normal := Mux(launchHit, normal_from_mem, zeroNormal)
  pe.io.in_valid  := io.in.valid
  io.out.bits := pe.io.out_result
  io.out.valid := pe.io.out_valid
}
