package sdf_rt

import chisel3._
import chisel3.util._
import raytrace_utils._
import raytrace_utils.fudian._

class RenderPE(cfg: FloatConfig) extends Module {
  val io = IO(new Bundle {
    val hit_id    = Input(UInt(cfg.addrWidth.W))
    val hit_valid = Input(Bool())
    val in_hit    = Input(Bool())
    val mem_req_id = Output(UInt(cfg.addrWidth.W))
    val mem_req_en = Output(Bool())

    val in_normal = Input(new Vec3(cfg))
    val in_valid  = Input(Bool())

    val out_rgb   = Output(new Vec3(cfg))
    val out_valid = Output(Bool())
    val out_id    = Output(UInt(cfg.addrWidth.W))
  })

  // 浮点常量定义 (IEEE 754)
  val val_0_15 = "h3E19999A".U // 0.15f
  val val_0_0  = "h00000000".U // 0.0f
  val val_1_0  = "h3F800000".U // 1.0f
  val color_coeffs = VecInit("h3F333333".U, "h3F4CCCCD".U, "h3F666666".U) // 0.7, 0.8, 0.9

  // --- 阶段 1: 内存请求 ---
  io.mem_req_id := io.hit_id
  io.mem_req_en := io.hit_valid && io.in_hit

  // --- 阶段 2: 渲染计算 ---
  // 1. Dot Product: diff = dot(normal, light)
  val dotUnit = Module(new DotProductUnit(cfg))
  dotUnit.io.a := io.in_normal
  dotUnit.io.b := "h3F3504F3".U.asTypeOf(new Vec3(cfg)) // 示例光照向量
  dotUnit.io.rm := 0.U

  // 2. Max(dot, 0.0)
  val cmpDot = Module(new FCMP(cfg))
  cmpDot.io.a := dotUnit.io.res
  cmpDot.io.b := val_0_0
  cmpDot.io.signaling:=false.B
  val diff = Mux(cmpDot.io.lt, val_0_0, dotUnit.io.res)

  // 3. Add Ambient: (diff + 0.15)
  val fadd = Module(new FADD(cfg))
  fadd.io.a := diff
  fadd.io.b := val_0_15
  fadd.io.rm := 0.U

  // 4. Multiply Color: (diff + 0.15) * Coeff
  val muls = Seq.fill(3)(Module(new FMUL(cfg)))
  for (i <- 0 until 3) {
    muls(i).io.a := fadd.io.res
    muls(i).io.b := color_coeffs(i)
    muls(i).io.rm := 0.U
  }

  // 5. Clamp: min(max(color, 0.0), 1.0)
  val clampedRGB = muls.map { mul =>
    val cmpMax = Module(new FCMP(cfg))
    cmpMax.io.a := val_1_0
    cmpMax.io.b := mul.io.result
    cmpMax.io.signaling:= false.B
    Mux(cmpMax.io.lt, val_1_0, mul.io.result)
  }

  // --- 阶段 3: 流水线同步 ---
  // 总延迟 = 点积 + 加法 + 乘法 + 比较
  val totalLatency = cfg.fdotLatency  + cfg.faddLatency + cfg.fmulLatency
  val hit_sync   = ShiftRegister(io.in_hit, totalLatency)
  val valid_sync = ShiftRegister(io.hit_valid, totalLatency)
  val id_sync = ShiftRegister(io.hit_id, totalLatency)

  io.out_rgb.x := Mux(hit_sync, clampedRGB(0), val_0_0)
  io.out_rgb.y := Mux(hit_sync, clampedRGB(1), val_0_0)
  io.out_rgb.z := Mux(hit_sync, clampedRGB(2), val_0_0)
  io.out_valid := valid_sync
  io.out_id    := id_sync
}