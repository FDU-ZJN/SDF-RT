package Trace

import chisel3._
import chisel3.util._
import raytrace_utils._
import raytrace_utils.fudian._

class RayTriangleIntersection(cfg: FloatConfig = FloatConfig.FP32) extends Module {
  val io = IO(new Bundle {
    val ray      = Input(new Ray(cfg))
    val tri      = Input(new Triangle(cfg))
    val in_valid = Input(Bool())
    val hit = Output(Bool())
    val t, u, v = Output(UInt(cfg.totalWidth.W))
    val out_valid = Output(Bool())
    val id       = Output(UInt(cfg.addrWidth.W))
  })
  val v0=io.tri.v0
  val v1=io.tri.v1
  val v2=io.tri.v2
  val orig = io.ray.origin
  val dir = io.ray.dir
  val latMUL = cfg.fmulLatency
  val latADD = cfg.faddLatency
  val latDIV = cfg.fdivLatency
  val latCP  = latMUL + latADD
  val latDP  = latMUL + latADD + latADD
  // Timing derivation (parameterized, no hardcoded 3/2/6 assumptions):
  // latCP = MUL + ADD
  // latDP = MUL + ADD + ADD
  // stageCLatency = ADD + latCP + latDP
  // stageDAlignLatency = max(DIV, latDP)
  // preCmpLatency = stageCLatency + stageDAlignLatency + MUL + ADD
  // totalLatency = preCmpLatency + FCMP
  //
  // Verification example with:
  // fmul=8, fadd=7, fcmp=2, fptoint=6, fdiv=29
  // latCP=15, latDP=22, stageCLatency=44, stageDAlignLatency=29,
  // preCmpLatency=88, totalLatency=90
  // Key milestones: T0(in_valid) -> T44(det/u') -> T73(aligned div+dot)
  // -> T81(final mul) -> T88(uv add) -> T90(hit/out_valid)
  val rm = 0.U

  // ---------------- Stage A (T0 -> T0+ADD) ----------------
  def vecSub(a: Vec3, b: Vec3): Vec3 = {
    val res = Wire(new Vec3(cfg))
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

  val e1 = vecSub(v1, v0)
  val e2 = vecSub(v2, v0)
  val s  = vecSub(orig, v0)

  val dir_d2 = PipeUtils.pipeData(dir, latADD)

  // ---------------- Stage B (T0+ADD -> T0+ADD+latCP) ----------------
  val cp_p = Module(new CrossProductUnit(cfg))
  cp_p.io.a := dir_d2
  cp_p.io.b := e2
  cp_p.io.rm := rm
  val p = cp_p.io.res

  val e1_d7  = PipeUtils.pipeData(e1, latCP)
  val s_d7   = PipeUtils.pipeData(s, latCP)
  val dir_d7 = PipeUtils.pipeData(dir_d2, latCP)
  val e2_d7  = PipeUtils.pipeData(e2, latCP)

  // ---------------- Stage C (T0+ADD+latCP -> T0+stageCLatency) ----------------
  val dp_det = Module(new DotProductUnit(cfg))
  dp_det.io.a := e1_d7
  dp_det.io.b := p
  dp_det.io.rm := rm
  val det = dp_det.io.res

  // 检测 det 是否为 0 (忽略符号位)
  val det_is_zero = det(cfg.totalWidth-2, 0) === 0.U

  val dp_u_prime = Module(new DotProductUnit(cfg))
  dp_u_prime.io.a := s_d7
  dp_u_prime.io.b := p
  dp_u_prime.io.rm := rm
  val u_prime = dp_u_prime.io.res

  val cp_q = Module(new CrossProductUnit(cfg))
  cp_q.io.a := s_d7
  cp_q.io.b := e1_d7
  cp_q.io.rm := rm
  val q_d14 = PipeUtils.pipeData(cp_q.io.res, latDP - latCP)

  val dir_d14 = PipeUtils.pipeData(dir_d7, latDP)
  val e2_d14  = PipeUtils.pipeData(e2_d7, latDP)

  // ---------------- Stage D ----------------
  // Align division and dot-product branches with computed latency target
  // so timing stays correct when fmul/fadd/fdiv latencies are reconfigured.
  val stageCLatency = latADD + latCP + latDP
  val stageDAlignLatency = math.max(latDIV, latDP)

  // det(T0+stageCLatency) -> invDet(T0+stageCLatency+latDIV)
  val fdiv = Module(new FRQ(cfg))
  fdiv.io.in := det
  fdiv.io.in_valid := PipeUtils.pipeData(io.in_valid, stageCLatency)

  val invDet_aligned = PipeUtils.pipeData(fdiv.io.result, stageDAlignLatency - latDIV)

  // u'(T0+stageCLatency) -> aligned to T0+stageCLatency+stageDAlignLatency
  val u_prime_aligned = PipeUtils.pipeData(u_prime, stageDAlignLatency)

  // v' / t' paths naturally produce at T0+stageCLatency+latDP
  val dp_v_prime = Module(new DotProductUnit(cfg))
  dp_v_prime.io.a := dir_d14
  dp_v_prime.io.b := q_d14
  dp_v_prime.io.rm := rm
  val v_prime_aligned = PipeUtils.pipeData(dp_v_prime.io.res, stageDAlignLatency - latDP)

  val dp_t_prime = Module(new DotProductUnit(cfg))
  dp_t_prime.io.a := e2_d14
  dp_t_prime.io.b := q_d14
  dp_t_prime.io.rm := rm
  val t_prime_aligned = PipeUtils.pipeData(dp_t_prime.io.res, stageDAlignLatency - latDP)

  // ---------------- Stage E ----------------
  def finalMul(a: UInt, b: UInt): UInt = {
    val m = Module(new FMUL(cfg))
    m.io.a := a
    m.io.b := b
    m.io.rm := rm
    m.io.result
  }

  val u_raw = finalMul(u_prime_aligned, invDet_aligned)
  val v_raw = finalMul(v_prime_aligned, invDet_aligned)
  val t_raw = finalMul(t_prime_aligned, invDet_aligned)

  // ---------------- Stage F (post-mul add and compare prep) ----------------
  val uv_adder = Module(new FADD(cfg))
  uv_adder.io.a := u_raw
  uv_adder.io.b := v_raw
  uv_adder.io.rm := rm
  val uv_sum = uv_adder.io.res

  val t_d26 = PipeUtils.pipeData(t_raw, latADD)
  val u_d26 = PipeUtils.pipeData(u_raw, latADD)
  val v_d26 = PipeUtils.pipeData(v_raw, latADD)

  val preCmpLatency = stageCLatency + stageDAlignLatency + latMUL + latADD
  val det_is_zero_pre_cmp = PipeUtils.pipeData(det_is_zero, preCmpLatency - stageCLatency)

  // 总延迟包含最终 uv 比较路径。
  val totalLatency = preCmpLatency + cfg.fcmpLatency
  val out_valid_final = PipeUtils.pipeData(io.in_valid, totalLatency)
  io.id:=PipeUtils.pipeData(io.tri.id, totalLatency)
  io.out_valid := out_valid_final

  // ---------------- Hit 判断 ----------------
  val fp_one = cfg.oneBigInt.U(cfg.totalWidth.W)

  val u_pos = !u_d26(cfg.totalWidth-1)
  val v_pos = !v_d26(cfg.totalWidth-1)
  val fcmp_uv = Module(new FCMP(cfg))
    fcmp_uv.io.a := uv_sum
    fcmp_uv.io.b := fp_one
    fcmp_uv.io.signaling := false.B
    val uv_le_one = fcmp_uv.io.le
  val det_is_zero_final = PipeUtils.pipeData(det_is_zero_pre_cmp, cfg.fcmpLatency)
  val u_pos_final = PipeUtils.pipeData(u_pos, cfg.fcmpLatency)
  val v_pos_final = PipeUtils.pipeData(v_pos, cfg.fcmpLatency)
  val t_final = PipeUtils.pipeData(t_d26, cfg.fcmpLatency)
  val u_final = PipeUtils.pipeData(u_d26, cfg.fcmpLatency)
  val v_final = PipeUtils.pipeData(v_d26, cfg.fcmpLatency)

  io.hit := out_valid_final && !det_is_zero_final && u_pos_final && v_pos_final && uv_le_one

  io.t := Mux(det_is_zero_final, 0.U, t_final)
  io.u := Mux(det_is_zero_final, 0.U, u_final)
  io.v := Mux(det_is_zero_final, 0.U, v_final)
}
