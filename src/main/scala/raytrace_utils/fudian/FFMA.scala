package raytrace_utils.fudian

import chisel3._
import chisel3.util._
import raytrace_utils._

class FFMA(cfg: FloatConfig = FloatConfig.FP32) extends Module {
  val io = IO(new Bundle {
    val a = Input(UInt(cfg.totalWidth.W))
    val b = Input(UInt(cfg.totalWidth.W))
    val c = Input(UInt(cfg.totalWidth.W))
    val res = Output(UInt(cfg.totalWidth.W))
  })

  if (cfg.useFloatIP) {
    val bb = Module(new Fma)
    bb.io.aclk := clock
    bb.io.s_axis_a_tdata := io.a
    bb.io.s_axis_a_tvalid := true.B
    bb.io.s_axis_b_tdata := io.b
    bb.io.s_axis_b_tvalid := true.B
    bb.io.s_axis_c_tdata := io.c
    bb.io.s_axis_c_tvalid := true.B
    io.res := Mux(bb.io.m_axis_result_tvalid, bb.io.m_axis_result_tdata, 0.U(cfg.totalWidth.W))
  } else {
    val mul = Module(new FMUL(cfg))
    val add = Module(new FADD(cfg))
    mul.io.a := io.a
    mul.io.b := io.b
    add.io.a := mul.io.result
    add.io.b := PipeUtils.pipeData(io.c, cfg.fmulLatency)
    io.res := add.io.res
  }
}

class Fma extends BlackBox with HasBlackBoxResource {
  val io = IO(new Bundle {
    val aclk = Input(Clock())
    val s_axis_a_tdata = Input(UInt(32.W))
    val s_axis_a_tvalid = Input(Bool())
    val s_axis_b_tdata = Input(UInt(32.W))
    val s_axis_b_tvalid = Input(Bool())
    val s_axis_c_tdata = Input(UInt(32.W))
    val s_axis_c_tvalid = Input(Bool())
    val m_axis_result_tdata = Output(UInt(32.W))
    val m_axis_result_tvalid = Output(Bool())
  })
}
