package raytrace_utils.fudian

import chisel3._
import chisel3.util._
import raytrace_utils._
// Cascade FMA (a * b + c)
class FCMA(cfg: FloatConfig = FloatConfig.FP32) extends Module {
  val expWidth=cfg.expWidth
  val precision=cfg.precision
  val io = IO(new Bundle() {
    val a, b, c = Input(UInt((expWidth + precision).W))
    val rm = Input(UInt(3.W))
    val result = Output(UInt((expWidth + precision).W))
    val fflags = Output(UInt(5.W))
  })

  val fmul = Module(new FMUL(cfg))
  val fadd = Module(new FCMA_ADD(expWidth, 2 * precision, precision))

  fmul.io.a := io.a
  fmul.io.b := io.b
  fmul.io.rm := io.rm

  val mul_to_fadd = fmul.io.to_fadd
  fadd.io.a := Cat(io.c, 0.U(precision.W))
  fadd.io.b := mul_to_fadd.fp_prod.asUInt
  fadd.io.b_inter_valid := true.B
  fadd.io.b_inter_flags := mul_to_fadd.inter_flags
  fadd.io.rm := io.rm

  io.result := fadd.io.result
  io.fflags := fadd.io.fflags
}

