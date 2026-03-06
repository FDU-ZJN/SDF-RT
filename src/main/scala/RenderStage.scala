package sdf_rt
import chisel3._
import chisel3.util._
import raytrace_utils._
import raytrace_utils.fudian._

package sdf_rt

import chisel3._
import chisel3.util._
import raytrace_utils._
import raytrace_utils.fudian._

class RenderStage(cfg: FloatConfig) extends Module {
  val io = IO(new Bundle {
    // 来自上游（例如求交模块）的碰撞信息
    val hit_id    = Input(UInt(cfg.addrWidth.W))
    val in_hit    = Input(Bool())
    val in_valid = Input(Bool())

    // 输出到下游（例如 FrameBuffer 或像素写回）
    val out_rgb   = Output(new Vec3(cfg))
    val out_valid = Output(Bool())
  })
  // 1. 实例化核心计算单元 (PE)
  val pe = Module(new RenderPE(cfg))

  // 2. 实例化法线存储器 (DPI)
  val mem = Module(new NormalMemDPI(cfg.addrWidth))
  mem.io.clk   := clock
  mem.io.reset := reset

  pe.io.hit_id    := io.hit_id
  pe.io.hit_valid := io.in_valid
  pe.io.in_hit   := io.in_hit

  mem.io.addr     := pe.io.mem_req_id
  mem.io.en       := pe.io.mem_req_en

  // B. 处理内存返回的数据并送回 PE
  // NormalMemDPI 返回的是 96 位原始数据，我们需要将其解包为 Vec3 浮点向量
  val normal_from_mem = Wire(new Vec3(cfg))
  normal_from_mem.x := mem.io.data(31, 0)
  normal_from_mem.y := mem.io.data(63, 32)
  normal_from_mem.z := mem.io.data(95, 64)

  pe.io.in_normal := normal_from_mem
  pe.io.in_valid  := mem.io.valid

  // C. 输出最终计算结果
  io.out_rgb   := pe.io.out_rgb
  io.out_valid := pe.io.out_valid
}
