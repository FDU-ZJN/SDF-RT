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
  val memLatency = GlobalConfig.normalMemDpiLatency
  val mem = Module(new NormalMemDPI(cfg.addrWidth, latency = memLatency))
  mem.io.clk   := clock
  mem.io.reset := reset

  io.in.ready := true.B

  val zeroNormal = 0.U.asTypeOf(new Vec3(cfg))

  val inFire = io.in.fire
  val launchHit = PipeUtils.pipeData(io.in.bits.hit, memLatency)
  val launchId = PipeUtils.pipeData(io.in.bits.hitId, memLatency)
  val launchMeta = PipeUtils.pipeData(io.in.bits.meta, memLatency)

  // Request only when input is accepted, so control/data timing stays cycle-aligned.
  mem.io.addr := Mux(io.in.bits.hit, io.in.bits.hitId, 0.U)
  mem.io.en := inFire

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
  pe.io.in_valid  := mem.io.valid
  io.out.bits := pe.io.out_result
  io.out.valid := pe.io.out_valid
}
