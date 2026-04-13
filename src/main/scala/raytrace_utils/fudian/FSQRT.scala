package raytrace_utils.fudian

import chisel3._
import chisel3.util._
import raytrace_utils._


class FSQRT(cfg: FloatConfig = FloatConfig.FP32) extends Module {
  val expWidth = cfg.expWidth
  val precision = cfg.precision
  val totalWidth = cfg.totalWidth

  val io = IO(new Bundle {
    val in  = Input(UInt(totalWidth.W))
    val out = Output(UInt(totalWidth.W))
  })

  if (cfg.useFloatIP) {
    val bb = Module(new Frsqrt)
    bb.io.aclk := clock
    bb.io.s_axis_a_tdata := io.in
    bb.io.s_axis_a_tvalid := true.B
    io.out := bb.io.m_axis_result_tdata
  } else {
    val rsqrtSim = Module(new RsqrtSimBlackBox(expWidth, precision))
    rsqrtSim.io.a := io.in
    io.out := PipeUtils.pipeData(rsqrtSim.io.result, cfg.fsqrtLatency)
  }
}

class RsqrtSimBlackBox(val expWidth: Int = 8, val precision: Int = 24)
    extends BlackBox(
      Map("WIDTH" -> (expWidth + precision))
    ) with HasBlackBoxResource {

  override val desiredName = "rsqrt_sim_model"

  val totalWidth = expWidth + precision

  val io = IO(new Bundle {
    val a      = Input(UInt(totalWidth.W))
    val result = Output(UInt(totalWidth.W))
  })

  addResource("/rsqrt_sim_model.sv")
}

// ---------------------------------------------------------------------------
// 外部浮点 IP 核接口 (综合用, 需用户提供实际实现)
// ---------------------------------------------------------------------------
class Frsqrt extends BlackBox with HasBlackBoxResource {
  val io = IO(new Bundle {
    val aclk = Input(Clock())
    val s_axis_a_tdata = Input(UInt(32.W))
    val s_axis_a_tvalid = Input(Bool())
    val m_axis_result_tdata = Output(UInt(32.W))
    val m_axis_result_tvalid = Output(Bool())
  })
}
