package raytrace_utils.fudian
import chisel3._
import chisel3.util._
import raytrace_utils.FloatConfig
import raytrace_utils.GlobalConfig
import raytrace_utils.PipeUtils
import raytrace_utils.fudian.utils.ShiftRightJam


class FPToInt  (
  val expWidth: Int = 8,
  val precision: Int = 24,
  val latency: Int = 1)
  extends Module {
  val io = IO(new Bundle {
    val a      = Input(UInt(32.W))
    val op = Input(Bool())
    val result = Output(UInt(32.W))
  })

  val sign  = io.a(31)
  val exp   = io.a(30, 23)
  val frac  = Cat(1.U(1.W), io.a(22, 0))

  val shift  = exp - 127.U
  val raw    = Mux(shift >= 23.U, frac << (shift - 23.U), frac >> (23.U - shift))
  val result = Mux(sign, (-raw.asSInt).asUInt, raw)

  io.result := PipeUtils.pipeData(result(31, 0), latency)
}




class Fptoint extends BlackBox with HasBlackBoxResource {
  val io = IO(new Bundle() {
    val aclk = Input(Clock())
    val s_axis_a_tdata = Input(UInt(32.W))
    val s_axis_a_tvalid = Input(Bool())
    val m_axis_result_tdata = Output(UInt(32.W))
    val m_axis_result_tvalid = Output(Bool())
  })
}
