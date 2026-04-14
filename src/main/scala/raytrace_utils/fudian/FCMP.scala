package raytrace_utils.fudian

import chisel3._
import chisel3.util._
import raytrace_utils.FloatConfig
import raytrace_utils.PipeUtils
class FCMP(cfg: FloatConfig = FloatConfig.FP32) extends Module {
  val expWidth=cfg.expWidth
  val precision=cfg.precision
  val io = IO(new Bundle() {
    val a, b = Input(UInt((expWidth + precision).W))
    val signaling = Input(Bool())
    val eq, le, lt = Output(Bool())
    val fflags = Output(UInt(5.W))
  })

  if (cfg.useFloatIP) {
    val bb = Module(new Fcmp)
    bb.io.aclk := clock
    bb.io.s_axis_a_tdata := io.a
    bb.io.s_axis_b_tdata := io.b
    bb.io.s_axis_a_tvalid := true.B
    bb.io.s_axis_b_tvalid := true.B

    // result bits: [0]=lt, [1]=eq, [2]=le
    io.lt := bb.io.m_axis_result_tvalid && bb.io.m_axis_result_tdata(1)
    io.eq := bb.io.m_axis_result_tvalid && bb.io.m_axis_result_tdata(0)
    io.le := io.lt||io.eq
    io.fflags := 0.U
  } else {
    val (a, b) = (io.a, io.b)
    val fp_a = FloatPoint.fromUInt(a, expWidth, precision)
    val fp_b = FloatPoint.fromUInt(b, expWidth, precision)
    val decode_a = fp_a.decode
    val decode_b = fp_b.decode

    val hasNaN = decode_a.isNaN || decode_b.isNaN
    val hasSNaN = decode_a.isSNaN || decode_b.isSNaN
    val bothZero = decode_a.isZero && decode_b.isZero

    val same_sign = fp_a.sign === fp_b.sign
    val a_minus_b = Cat(0.U(1.W), a) - Cat(0.U(1.W), b)
    val uint_eq = a_minus_b.tail(1) === 0.U
    val uint_less = fp_a.sign ^ a_minus_b.head(1).asBool

    val invalid = hasSNaN || (io.signaling && hasNaN)

    val eqRaw = !hasNaN && (uint_eq || bothZero)
    val leRaw = !hasNaN && Mux(
      same_sign,
      uint_less || uint_eq,
      fp_a.sign || bothZero
    )
    val ltRaw = !hasNaN && Mux(
      same_sign,
      uint_less && !uint_eq,
      fp_a.sign && !bothZero
    )
    val fflagsRaw = Cat(invalid, 0.U(4.W))

    io.eq := PipeUtils.pipeData(eqRaw, cfg.fcmpLatency)
    io.le := PipeUtils.pipeData(leRaw, cfg.fcmpLatency)
    io.lt := PipeUtils.pipeData(ltRaw, cfg.fcmpLatency)
    io.fflags := PipeUtils.pipeData(fflagsRaw, cfg.fcmpLatency)
  }
}

class Fcmp extends BlackBox with HasBlackBoxResource {
  val io = IO(new Bundle() {
    val aclk = Input(Clock())
    val s_axis_a_tdata = Input(UInt(32.W))
    val s_axis_a_tvalid = Input(Bool())
    val s_axis_b_tdata = Input(UInt(32.W))
    val s_axis_b_tvalid = Input(Bool())
    val m_axis_result_tdata = Output(UInt(8.W))
    val m_axis_result_tvalid = Output(Bool())
  })
}
