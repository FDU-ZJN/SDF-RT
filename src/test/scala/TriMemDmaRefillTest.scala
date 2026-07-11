import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec
import raytrace_utils.{FloatConfig, GlobalConfig, TriPeConfig}
import Trace.TriangleMemRefillBackend

class TriMemDmaRefillTest extends AnyFlatSpec with ChiselScalatestTester {
  behavior of "TriangleMemRefillBackend DataMover path"

  private val cfg = TriPeConfig(cfg = FloatConfig.FP32.copy(), numPEs = 1)

  it should "issue a 64-byte command and return one triangle after data and status" in {
    test(GlobalConfig.withMemImplMode(2)(new TriangleMemRefillBackend(cfg, numBanks = 8, idWidth = 2))) { dut =>
      dut.io.triangleBaseAddress.poke(BigInt("10000000", 16).U)
      dut.io.req.valid.poke(false.B)
      dut.io.resp.ready.poke(true.B)
      dut.io.dmaCmd.ready.poke(true.B)
      dut.io.dmaData.valid.poke(false.B)
      dut.io.dmaStatus.valid.poke(false.B)
      dut.clock.step()

      dut.io.req.bits.bank.poke(3.U)
      dut.io.req.bits.addr.poke(2.U)
      dut.io.req.bits.id.poke(1.U)
      dut.io.req.valid.poke(true.B)
      dut.io.dmaCmd.valid.expect(true.B)
      val expectedAddress = BigInt("10000000", 16) + ((2 * 8 + 3) * 64)
      val command = dut.io.dmaCmd.bits.peek().litValue
      assert(((command >> 32) & ((BigInt(1) << 40) - 1)) == expectedAddress)
      assert(((command >> 72) & 0xf) == 1)
      assert(((command >> 30) & 1) == 1)
      assert(((command >> 23) & 1) == 1)
      assert((command & ((BigInt(1) << 23) - 1)) == 64)
      dut.clock.step()
      dut.io.req.valid.poke(false.B)

      val beats = Seq(
        BigInt("00112233445566778899aabbccddeeff", 16),
        BigInt("102132435465768798a9bacbdcedfe0f", 16),
        BigInt("2031425364758697a8b9cadbecfd0e1f", 16),
        BigInt("30415263748596a7b8c9daebfc0d1e2f", 16)
      )
      beats.zipWithIndex.foreach { case (data, index) =>
        dut.io.dmaData.bits.data.poke(data.U)
        dut.io.dmaData.bits.last.poke((index == 3).B)
        dut.io.dmaData.valid.poke(true.B)
        dut.io.dmaData.ready.expect(true.B)
        dut.clock.step()
      }
      dut.io.dmaData.valid.poke(false.B)

      dut.io.dmaStatus.bits.poke("h81".U) // OKAY plus tag 1
      dut.io.dmaStatus.valid.poke(true.B)
      dut.clock.step()
      dut.io.dmaStatus.valid.poke(false.B)

      dut.io.resp.valid.expect(true.B)
      dut.io.resp.bits.id.expect(1.U)
      val line = beats.zipWithIndex.map { case (beat, i) => beat << (128 * i) }.reduce(_ | _)
      dut.io.resp.bits.data.expect((line & ((BigInt(1) << 288) - 1)).U)
      dut.clock.step()
      dut.io.issuedCount.expect(1.U)
      dut.io.completedCount.expect(1.U)
      dut.io.outstandingCount.expect(0.U)
      dut.io.dmaReadError.expect(false.B)
    }
  }

  it should "complete with zero data and record a DataMover status error" in {
    test(GlobalConfig.withMemImplMode(2)(new TriangleMemRefillBackend(cfg, numBanks = 8, idWidth = 2))) { dut =>
      dut.io.triangleBaseAddress.poke(0.U)
      dut.io.resp.ready.poke(true.B)
      dut.io.dmaCmd.ready.poke(true.B)
      dut.io.dmaData.valid.poke(false.B)
      dut.io.dmaStatus.valid.poke(false.B)
      dut.io.req.bits.bank.poke(0.U)
      dut.io.req.bits.addr.poke(0.U)
      dut.io.req.bits.id.poke(2.U)
      dut.io.req.valid.poke(true.B)
      dut.clock.step()
      dut.io.req.valid.poke(false.B)

      for (i <- 0 until 4) {
        dut.io.dmaData.bits.data.poke((i + 1).U)
        dut.io.dmaData.bits.last.poke((i == 3).B)
        dut.io.dmaData.valid.poke(true.B)
        dut.clock.step()
      }
      dut.io.dmaData.valid.poke(false.B)
      dut.io.dmaStatus.bits.poke("h42".U) // SLVERR plus tag 2
      dut.io.dmaStatus.valid.poke(true.B)
      dut.clock.step()
      dut.io.dmaStatus.valid.poke(false.B)

      dut.io.resp.valid.expect(true.B)
      dut.io.resp.bits.id.expect(2.U)
      dut.io.resp.bits.data.expect(0.U)
      dut.io.dmaReadError.expect(true.B)
      dut.io.dmaStatusErrorCount.expect(1.U)
    }
  }
}
