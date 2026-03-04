package sdf_rt
import raytrace_utils._
import raytrace_utils.fudian._
import chisel3._
import chisel3.util._

class RayTriangleIntersection(cfg: FloatConfig = FloatConfig.FP32) extends Module {
  val io = IO(new Bundle {
    val orig, dir = Input(new Vec3(cfg.totalWidth))
    val v0, v1, v2 = Input(new Vec3(cfg.totalWidth))
    val in_valid = Input(Bool())
    val hit = Output(Bool())
    val t, u, v = Output(UInt(cfg.totalWidth.W))
    val out_valid = Output(Bool())
  })

  val latMUL = 3
  val latADD = 2
  val latDIV = 6
  val latCP  = latMUL + latADD      // 5
  val latDP  = latMUL + latADD + latADD // 7
  val rm = 0.U

  // ---------------- Stage A (T0→T2) ----------------
  def vecSub(a: Vec3, b: Vec3): Vec3 = {
    val res = Wire(new Vec3(cfg.totalWidth))
    val subs = Seq.fill(3)(Module(new FADD(cfg)))
    val as = Seq(a.x, a.y, a.z)
    val bs = Seq(b.x, b.y, b.z)

    for (i <- 0 until 3) {
      subs(i).io.a := as(i)
      subs(i).io.b := Cat(!bs(i)(cfg.totalWidth-1), bs(i)(cfg.totalWidth-2, 0))
      subs(i).io.rm := rm
    }

    res.x := subs(0).io.res
    res.y := subs(1).io.res
    res.z := subs(2).io.res
    res
  }

  val e1 = vecSub(io.v1, io.v0)
  val e2 = vecSub(io.v2, io.v0)
  val s  = vecSub(io.orig, io.v0)

  val dir_d2 = ShiftRegister(io.dir, latADD)

  // ---------------- Stage B (T2→T7) ----------------
  val cp_p = Module(new CrossProductUnit(cfg))
  cp_p.io.a := dir_d2
  cp_p.io.b := e2
  cp_p.io.rm := rm
  val p = cp_p.io.res   // T7

  val e1_d7  = ShiftRegister(e1, latCP)
  val s_d7   = ShiftRegister(s, latCP)
  val dir_d7 = ShiftRegister(dir_d2, latCP)
  val e2_d7  = ShiftRegister(e2, latCP)

  // ---------------- Stage C (T7→T14) ----------------
  val dp_det = Module(new DotProductUnit(cfg))
  dp_det.io.a := e1_d7
  dp_det.io.b := p
  dp_det.io.rm := rm
  val det = dp_det.io.res  // T14

  // 检测 det 是否为 0 (忽略符号位)
  val det_is_zero = det(cfg.totalWidth-2, 0) === 0.U

  val dp_u_prime = Module(new DotProductUnit(cfg))
  dp_u_prime.io.a := s_d7
  dp_u_prime.io.b := p
  dp_u_prime.io.rm := rm
  val u_prime = dp_u_prime.io.res  // T14

  val cp_q = Module(new CrossProductUnit(cfg))
  cp_q.io.a := s_d7
  cp_q.io.b := e1_d7
  cp_q.io.rm := rm
  val q_d14 = ShiftRegister(cp_q.io.res, latDP - latCP) // 2拍 → T14

  val dir_d14 = ShiftRegister(dir_d7, latDP)
  val e2_d14  = ShiftRegister(e2_d7, latDP)

  // ---------------- Stage D ----------------
  // det T14 → invDet T20
  val fdiv = Module(new FDIV(cfg))
  fdiv.io.a := cfg.oneBigInt.U(cfg.totalWidth.W)
  fdiv.io.b := det
  fdiv.io.in_valid := ShiftRegister(io.in_valid, latADD + latCP + latDP)

  val invDet_d21 = ShiftRegister(fdiv.io.result, 1) // 对齐到 T21

  // u' T14 → T21
  val u_prime_d21 = ShiftRegister(u_prime, latDIV + 1)

  // v' & t'：T14 → T21
  val dp_v_prime = Module(new DotProductUnit(cfg))
  dp_v_prime.io.a := dir_d14
  dp_v_prime.io.b := q_d14
  dp_v_prime.io.rm := rm
  val v_prime = dp_v_prime.io.res // T21

  val dp_t_prime = Module(new DotProductUnit(cfg))
  dp_t_prime.io.a := e2_d14
  dp_t_prime.io.b := q_d14
  dp_t_prime.io.rm := rm
  val t_prime = dp_t_prime.io.res // T21

  // ---------------- Stage E (T21→T24) ----------------
  def finalMul(a: UInt, b: UInt): UInt = {
    val m = Module(new FMUL(cfg))
    m.io.a := a
    m.io.b := b
    m.io.rm := rm
    m.io.result
  }

  val u_raw = finalMul(u_prime_d21, invDet_d21)
  val v_raw = finalMul(v_prime,     invDet_d21)
  val t_raw = finalMul(t_prime,     invDet_d21)

  // ---------------- Stage F (T24→T26) ----------------
  val uv_adder = Module(new FADD(cfg))
  uv_adder.io.a := u_raw
  uv_adder.io.b := v_raw
  uv_adder.io.rm := rm
  val uv_sum = uv_adder.io.res

  val t_d26 = ShiftRegister(t_raw, latADD)
  val u_d26 = ShiftRegister(u_raw, latADD)
  val v_d26 = ShiftRegister(v_raw, latADD)

  // 将 det_is_zero 从 T14 传递到 T26 (延迟 12 拍)
  val det_is_zero_d26 = ShiftRegister(det_is_zero, 12)

  // 总延迟修正为 26
  val totalLatency = 26
  val out_valid_final = ShiftRegister(io.in_valid, totalLatency)
  io.out_valid := out_valid_final

  // ---------------- Hit 判断 ----------------
  val fp_one = cfg.oneBigInt.U(cfg.totalWidth.W)

  val u_pos = !u_d26(cfg.totalWidth-1)
  val v_pos = !v_d26(cfg.totalWidth-1)
  val uv_le_one = uv_sum <= fp_one || uv_sum === fp_one
  val t_pos = !t_d26(cfg.totalWidth-1)


  io.hit := out_valid_final && !det_is_zero_d26 && u_pos && v_pos && uv_le_one && t_pos


  io.t := Mux(det_is_zero_d26, 0.U, t_d26)
  io.u := Mux(det_is_zero_d26, 0.U, u_d26)
  io.v := Mux(det_is_zero_d26, 0.U, v_d26)
}

object RayTriangleIntersectionGen extends App {
  val cfg = FloatConfig.FP32
  emitVerilog(new RayTriangleIntersection(cfg), Array("--target-dir", "build"))
}