package raytrace_utils.fudian

import chisel3._
import chisel3.util._
import raytrace_utils.GlobalConfig

class IMUL(useFloatIP: Boolean = GlobalConfig.useFloatIP) extends Module {
  val io = IO(new Bundle {
    val a = Input(UInt(16.W))
    val b = Input(UInt(16.W))
    val p = Output(UInt(32.W))
  })

  if (useFloatIP) {
    val bb = Module(new Imul)
    bb.io.CLK := clock
    bb.io.A := io.a
    bb.io.B := io.b
    io.p := bb.io.P
  } else {
    io.p := RegNext(io.a * io.b)
  }
}

class Imul extends BlackBox with HasBlackBoxResource {
  val io = IO(new Bundle {
    val CLK = Input(Clock())
    val A = Input(UInt(16.W))
    val B = Input(UInt(16.W))
    val P = Output(UInt(32.W))
  })
}
