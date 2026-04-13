package raytrace_utils

import chisel3._
import chisel3.util._

class FRQ(cfg: FloatConfig = FloatConfig.FP32) extends Module {
  val totalWidth = cfg.totalWidth

  val io = IO(new Bundle {
    val in        = Input(UInt(totalWidth.W))
    val result    = Output(UInt(totalWidth.W))
    val out_valid = Output(Bool())
    val in_valid  = Input(Bool())
  })

  if (cfg.useFloatIP) {
    val bb = Module(new Fdiv)
    bb.io.aclk := clock
    bb.io.s_axis_a_tdata := io.in
    bb.io.s_axis_a_tvalid := io.in_valid

    io.result    := bb.io.m_axis_result_tdata
    io.out_valid := bb.io.m_axis_result_tvalid
  } else {
    val fullOne = BigInt(0x3F800000L).U(totalWidth.W) // 1.0f

    val fdiv = Module(new FDIV(cfg))
    fdiv.io.a := fullOne
    fdiv.io.b := io.in
    fdiv.io.in_valid := io.in_valid

    io.result    := fdiv.io.result
    io.out_valid := fdiv.io.out_valid
  }
}
