package raytrace_utils

import chisel3._
import chisel3.util._

class RingBuffIO[T <: Bundle](dataT: () => T, rchannel: Int) extends Bundle {
  val full    = Output(Bool())
  val empty   = Output(Bool())
  val push1   = Input(Bool())
  val wdata1  = Input(dataT())
  val push2   = Input(Bool())
  val wdata2  = Input(dataT())
  
  val pop     = Input(Bool())
  val popN    = Input(UInt(1.W))
  val rdatas  = Output(Vec(rchannel, dataT()))
  val rvalids = Output(Vec(rchannel, Bool()))
  val clear   = Input(Bool())
}

class RingBuff[T <: Bundle](dataT: ()=> T, len: Int, rchannel: Int, debug_id: Int) extends Module{
    require((len&(len-1))==0)
    val io = IO(new RingBuffIO(dataT, rchannel))
    val buff = Reg(Vec(len, dataT()))
    val buff_head = Reg(UInt(log2Up(len).W))
    val buff_tail = Reg(UInt(log2Up(len).W))
    val buff_count = Reg(UInt(log2Ceil(len + 1).W))
    val empty = buff_count === 0.U
    val full = buff_count === len.U

    io.empty := empty
    io.full := full

    for(i <- 0 until rchannel){
        io.rdatas(i) := buff(buff_head+i.U)
        io.rvalids(i) := buff_count > i.U
    }

    when(io.clear){
        buff_head := 0.U
        buff_tail := 0.U
        buff_count := 0.U
    }.otherwise{
        val popCount = Wire(UInt(log2Ceil(rchannel + 1).W))
        popCount := 0.U
        when(io.pop) {
            popCount := io.popN + 1.U
        }

        val countAfterPop = buff_count - popCount
        val pushReqCount = PopCount(Seq(io.push1 || io.push2, io.push1 && io.push2))
        val pushAcceptCount = Wire(UInt(log2Ceil(3).W))
        pushAcceptCount := 0.U

        when(io.push1 && io.push2 && (countAfterPop <= (len - 2).U)) {
            pushAcceptCount := 2.U
            buff(buff_tail) := io.wdata1
            buff(buff_tail + 1.U) := io.wdata2
        }.elsewhen((io.push1 || io.push2) && (countAfterPop <= (len - 1).U)) {
            pushAcceptCount := 1.U
            buff(buff_tail) := io.wdata1
        }

        when(io.pop) {
            buff_head := buff_head + popCount
        }
        when(pushAcceptCount =/= 0.U) {
            buff_tail := buff_tail + pushAcceptCount
        }
        buff_count := countAfterPop + pushAcceptCount

        when(pushReqCount > pushAcceptCount) {
            assert(false.B, s"RingBuff[$debug_id] overflow")
        }
  }
}
