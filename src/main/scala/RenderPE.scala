package sdf_rt

import chisel3._
import chisel3.util._
import raytrace_utils._
import raytrace_utils.fudian._

class RenderPE(cfg: FloatConfig) extends Module {
  val io = IO(new Bundle {
    // 1. 发起读取请求
    val hit_id    = Input(UInt(cfg.addrWidth.S.W))
    val hit_valid = Input(Bool())  // 表示这一拍有像素进入处理
    val in_hit    = Input(Bool())  // 表示该像素是否真的命中了物体

    // 2. 外部读取接口 (连接 NormalMemDPI)
    val mem_req_id = Output(UInt(cfg.addrWidth.W))
    val mem_req_en = Output(Bool())

    // 3. 内存返回的数据与有效标志
    val in_normal = Input(new Vec3(cfg))
    val in_valid  = Input(Bool())  // 对应内存返回的 valid

    // 4. 最终渲染输出
    val out_rgb   = Output(new Vec3(cfg))
    val out_valid = Output(Bool())

  })

  // --- 阶段 1: 请求分发 ---
  // 只有在命中的情况下才去读内存，不命中也要传下去维持流水线节奏
  io.mem_req_id := io.hit_id
  io.mem_req_en := io.hit_valid && io.in_hit

  // --- 阶段 2: 核心计算 ---
  val light_dir = Wire(new Vec3(cfg))
  light_dir.x := 0.U
  light_dir.y := 0.U
  light_dir.z := "hBF800000".U // -1.0

  val dotUnit = Module(new DotProductUnit(cfg))
  dotUnit.io.a := io.in_normal
  dotUnit.io.b := light_dir
  dotUnit.io.rm := 0.U

  // 这里的 dot_valid 只是为了驱动后续逻辑，注意 miss 情况下 in_valid 可能不跳变
  // 因此在流水线同步时需要额外考虑，或者确保 NormalMem 对 miss 也返回 valid (数据全0)
  val dot_valid = ShiftRegister(io.hit_valid, cfg.fdotLatency)

  val zero = 0.U(32.W)
  val cmp = Module(new FCMP(cfg))
  cmp.io.a := dotUnit.io.res
  cmp.io.b := zero
  cmp.io.signaling := false.B

  val diffuse = Mux(cmp.io.lt, zero, dotUnit.io.res)

  val one = cfg.oneBigInt.U(32.W)
  val muls = Seq.fill(3)(Module(new FMUL(cfg)))
  muls.foreach { mul =>
    mul.io.a := diffuse
    mul.io.b := one
    mul.io.rm := 0.U
  }
  val hit_sync = ShiftRegister(io.in_hit, cfg.fdotLatency + cfg.fmulLatency)

  // 最终颜色选择：如果 hit_sync 为 false，强行输出黑色 (0,0,0)
  io.out_rgb.x := Mux(hit_sync, muls(0).io.result, zero)
  io.out_rgb.y := Mux(hit_sync, muls(1).io.result, zero)
  io.out_rgb.z := Mux(hit_sync, muls(2).io.result, zero)

  // 最终 valid 同步
  io.out_valid := ShiftRegister(dot_valid, cfg.fmulLatency)
}