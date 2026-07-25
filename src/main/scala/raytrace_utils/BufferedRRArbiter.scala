package raytrace_utils

import chisel3._
import chisel3.util._

class BufferedRRArbiter[T <: Data](gen: T, numInputs: Int, fifoDepth: Int = 2) extends Module {
  require(numInputs > 0, "numInputs must be positive")
  require(fifoDepth > 0, "fifoDepth must be positive")
  private val occupancyWidth = log2Ceil(numInputs * fifoDepth + 1)
  val io = IO(new Bundle {
    val enq = Vec(numInputs, Flipped(Decoupled(gen)))
    val deq = Decoupled(gen)
    val occupancy = Output(UInt(occupancyWidth.W))
  })

  val fifos = Seq.fill(numInputs)(Module(new Queue(gen, fifoDepth, pipe = true)))
  val arbiter = Module(new RRArbiter(gen, numInputs))
  for (i <- 0 until numInputs) {
    fifos(i).io.enq <> io.enq(i)
    arbiter.io.in(i) <> fifos(i).io.deq
  }
  io.occupancy := fifos.map(_.io.count).foldLeft(0.U(occupancyWidth.W))(_ +& _)
  io.deq <> arbiter.io.out
}

