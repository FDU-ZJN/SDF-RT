package raytrace_utils

import chisel3._
import chisel3.util._

class TwoLaneDeq[T <: Data](gen: T) extends Bundle {
  val validMask = UInt(2.W)
  val bits = Vec(2, gen)
}

class FourToTwoArbiter[T <: Data](gen: T, fifoDepth: Int = 2) extends Module {
  require(fifoDepth > 0, "fifoDepth must be positive")
  val io = IO(new Bundle {
    val enq = Vec(4, Flipped(Decoupled(gen)))
    val deq = Decoupled(new TwoLaneDeq(gen))
    val occupancy = Output(UInt(log2Ceil(4 * fifoDepth + 1).W))
  })

  val fifos = Seq.fill(4)(Module(new Queue(gen, fifoDepth, pipe = true)))
  val fifoDeq = VecInit(fifos.map(_.io.deq))
  for (i <- 0 until 4) {
    fifos(i).io.enq <> io.enq(i)
  }

  val fifoValid = VecInit(fifos.map(_.io.deq.valid))
  val fifoBits = VecInit(fifos.map(_.io.deq.bits))
  io.occupancy := fifos.map(_.io.count).foldLeft(0.U)(_ + _)

  val rrPtr = RegInit(0.U(2.W))

  val rotate0 = VecInit((0 until 4).map { i =>
    val idx = (i.U + rrPtr)(1, 0)
    fifoValid(idx)
  })
  val pick0OH = PriorityEncoderOH(rotate0)
  val pick0Valid = rotate0.asUInt.orR
  val pick0Idx = (OHToUInt(pick0OH) + rrPtr)(1, 0)

  val base1 = Mux(pick0Valid, pick0Idx + 1.U, rrPtr)
  val rotate1 = VecInit((0 until 4).map { i =>
    val idx = (i.U + base1)(1, 0)
    fifoValid(idx) && (!pick0Valid || idx =/= pick0Idx)
  })
  val pick1OH = PriorityEncoderOH(rotate1)
  val pick1Valid = rotate1.asUInt.orR
  val pick1Idx = (OHToUInt(pick1OH) + base1)(1, 0)

  when(io.deq.fire) {
    rrPtr := Mux(pick1Valid, pick1Idx + 1.U, pick0Idx + 1.U)
  }

  io.deq.valid := pick0Valid
  io.deq.bits.validMask := Cat(pick1Valid, pick0Valid)
  io.deq.bits.bits(0) := Mux(pick0Valid, fifoBits(pick0Idx), 0.U.asTypeOf(gen))
  io.deq.bits.bits(1) := Mux(pick1Valid, fifoBits(pick1Idx), 0.U.asTypeOf(gen))

  for (i <- 0 until 4) {
    fifoDeq(i).ready := false.B
  }
  when(io.deq.fire) {
    when(pick0Valid) { fifoDeq(pick0Idx).ready := true.B }
    when(pick1Valid) { fifoDeq(pick1Idx).ready := true.B }
  }
}
