package raytrace_utils
import chisel3._
import chisel3.util._
class BvhStack(addrWidth: Int, depth: Int) extends Module {
  val ptrW = log2Ceil(depth + 1)

  val io = IO(new Bundle {
    val pushLeft  = Input(Bool())
    val pushRight = Input(Bool())
    val pop       = Input(Bool())
    val leftData  = Input(UInt(addrWidth.W))
    val rightData = Input(UInt(addrWidth.W))
    val topData   = Output(UInt(addrWidth.W))
    val level     = Output(UInt(ptrW.W))
    val empty     = Output(Bool())
    val full      = Output(Bool())
  })

  val mem = Reg(Vec(depth, UInt(addrWidth.W)))
  val sp  = RegInit(0.U(ptrW.W))

  val pushCount = io.pushLeft.asUInt +& io.pushRight.asUInt
  val popCount  = io.pop.asUInt
  val baseSp    = sp - popCount

  // 先 pop，再 push（与原代码逻辑对齐）
  when(io.pushLeft) {
    mem(baseSp(log2Ceil(depth) - 1, 0)) := io.leftData
  }
  when(io.pushRight) {
    mem((baseSp + io.pushLeft.asUInt)(log2Ceil(depth) - 1, 0)) := io.rightData
  }
  when(io.pop || io.pushLeft || io.pushRight) {
    sp := baseSp + pushCount
  }

  io.topData := mem((sp - 1.U)(log2Ceil(depth) - 1, 0))
  io.level   := sp
  io.empty   := sp === 0.U
  io.full    := sp === depth.U
}

