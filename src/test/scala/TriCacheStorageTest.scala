import chisel3._
import chiseltest._
import chisel3.util._
import org.scalatest.flatspec.AnyFlatSpec
import Trace.TriMemRefillLineArray

class TriCacheStorageTest extends AnyFlatSpec with ChiselScalatestTester {
  behavior of "Triangle cache BRAM storage models"

  it should "preserve ordering across a full 4-entry response queue" in {
    test(new Queue(UInt(16.W), entries = 4, useSyncReadMem = true)) { dut =>
      dut.io.enq.valid.poke(false.B)
      dut.io.deq.ready.poke(false.B)
      dut.clock.step(2)
      dut.io.deq.valid.expect(false.B)

      for (value <- 0 until 4) {
        dut.io.enq.bits.poke(value.U)
        dut.io.enq.valid.poke(true.B)
        dut.io.enq.ready.expect(true.B)
        dut.clock.step()
      }
      dut.io.enq.valid.poke(false.B)
      dut.io.enq.ready.expect(false.B)
      dut.io.count.expect(4.U)

      for (value <- 0 until 4) {
        dut.io.deq.valid.expect(true.B)
        dut.io.deq.bits.expect(value.U)
        dut.io.deq.ready.poke(true.B)
        dut.clock.step()
        dut.io.deq.ready.poke(false.B)
      }
      dut.io.deq.valid.expect(false.B)
      dut.io.count.expect(0.U)
    }
  }

  it should "route four refill lines by MSHR ID" in {
    test(new TriMemRefillLineArray(dataWidth = 32, depth = 4, useXpm = false)) { dut =>
      dut.io.wr_en.poke(false.B)
      dut.io.rd_en.poke(false.B)
      dut.clock.step()

      for (id <- 0 until 4) {
        dut.io.wr_en.poke(true.B)
        dut.io.wr_addr.poke(id.U)
        dut.io.wr_data.poke((0x12340000L + id).U)
        dut.clock.step()
      }
      dut.io.wr_en.poke(false.B)

      for (id <- 0 until 4) {
        dut.io.rd_en.poke(true.B)
        dut.io.rd_addr.poke(id.U)
        dut.clock.step()
        dut.io.rd_data.expect((0x12340000L + id).U)
      }
    }
  }
}
