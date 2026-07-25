import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec
import raytrace_utils.BufferedRRArbiter

class BufferedRRArbiterTest extends AnyFlatSpec with ChiselScalatestTester {
  behavior of "BufferedRRArbiter"

  it should "serialize simultaneous inputs without loss under backpressure" in {
    test(new BufferedRRArbiter(UInt(8.W), numInputs = 4, fifoDepth = 2)) { dut =>
      dut.io.deq.ready.poke(false.B)
      for (i <- 0 until 4) {
        dut.io.enq(i).valid.poke(true.B)
        dut.io.enq(i).bits.poke((10 + i).U)
      }
      dut.clock.step()
      for (i <- 0 until 4) dut.io.enq(i).valid.poke(false.B)
      dut.io.occupancy.expect(4.U)

      dut.clock.step(3)
      dut.io.occupancy.expect(4.U)
      dut.io.deq.ready.poke(true.B)
      val seen = collection.mutable.Set.empty[Int]
      while (seen.size < 4) {
        if (dut.io.deq.valid.peek().litToBoolean) {
          seen += dut.io.deq.bits.peek().litValue.toInt
        }
        dut.clock.step()
      }
      assert(seen == Set(10, 11, 12, 13))
      dut.io.occupancy.expect(0.U)
    }
  }
}
