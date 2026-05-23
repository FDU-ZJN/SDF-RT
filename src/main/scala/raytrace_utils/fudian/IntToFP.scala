package raytrace_utils.fudian

import chisel3._
import chisel3.util._
import raytrace_utils._

class IntToFP(val expWidth: Int, val precision: Int) extends Module {
  val io = IO(new Bundle {
    val int = Input(UInt(32.W))
    val result = Output(UInt((expWidth + precision).W))
  })

  private val cfg = FloatConfig(expWidth, precision)
  private val value = io.int
  private val isZero = value === 0.U
  private val lzc = PriorityEncoder(Reverse(value))
  private val msbIdx = 31.U - lzc
  private val bias = FloatPoint.expBias(expWidth).U(expWidth.W)
  private val exp = (bias + msbIdx)(expWidth - 1, 0)

  private val shifted = Wire(UInt(32.W))
  when(msbIdx >= (precision - 1).U) {
    shifted := value >> (msbIdx - (precision - 1).U)
  }.otherwise {
    shifted := value << ((precision - 1).U - msbIdx)
  }

  private val frac = shifted(precision - 2, 0)
  io.result := Mux(isZero, 0.U((expWidth + precision).W), Cat(0.U(1.W), exp, frac))
}
