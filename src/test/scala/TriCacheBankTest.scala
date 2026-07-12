import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec
import raytrace_utils._
import Trace._

class TriCacheBankTest extends AnyFlatSpec with ChiselScalatestTester {
  behavior of "TriangleMemCachedBank"

  it should "reset valid bits, refill a miss, and serve the following hit" in {
    val c = TriPeConfig(cacheLineTriangles = 2)
    test(new TriangleMemCachedBank(c, srcWidth = 1, tagWidth = 4, bankId = 0, numBanks = 1,
      numSets = 4, ways = 1, reqQueueDepth = 2, mergeQueueDepth = 2, mshrEntries = 1,
      useXpmCacheOverride = Some(false))) { dut =>
      dut.io.req.valid.poke(false.B)
      dut.io.resp.ready.poke(true.B)
      dut.io.allocResp.valid.poke(false.B)
      dut.io.refillDone.valid.poke(false.B)
      dut.clock.step(2)
      dut.clock.step(5)

      dut.io.req.bits.addr.poke(3.U)
      dut.io.req.bits.mask.poke(3.U)
      dut.io.req.bits.src.poke(0.U)
      dut.io.req.bits.tag.poke(9.U)
      dut.io.req.valid.poke(true.B)
      while (!dut.io.req.ready.peekBoolean()) {
        dut.clock.step()
      }
      dut.clock.step()
      dut.io.req.valid.poke(false.B)

      while (!dut.io.allocReq.valid.peekBoolean()) {
        dut.clock.step()
      }
      dut.io.allocResp.bits.id.poke(0.U)
      dut.io.allocResp.bits.merged.poke(false.B)
      dut.io.allocResp.valid.poke(true.B)
      dut.clock.step()
      dut.io.allocResp.valid.poke(false.B)

      val line = BigInt("3f8000004000000040400000", 16) | (BigInt("4080000040a0000040c00000", 16) << 288)
      dut.io.refillDone.bits.id.poke(0.U)
      dut.io.refillDone.bits.data.poke(line.U)
      dut.io.refillDone.valid.poke(true.B)
      while (!dut.io.refillDone.ready.peekBoolean()) {
        dut.clock.step()
      }
      dut.clock.step()
      dut.io.refillDone.valid.poke(false.B)

      while (!dut.io.resp.valid.peekBoolean()) {
        dut.clock.step()
      }
      dut.io.resp.bits.resp.tag.expect(9.U)
      dut.io.resp.bits.resp.block.tris(0).v0.x.expect("h40400000".U)
      dut.clock.step()

      dut.io.req.bits.addr.poke(3.U)
      dut.io.req.bits.mask.poke(3.U)
      dut.io.req.bits.src.poke(0.U)
      dut.io.req.bits.tag.poke(6.U)
      dut.io.req.valid.poke(true.B)
      while (!dut.io.req.ready.peekBoolean()) {
        dut.clock.step()
      }
      dut.clock.step()
      dut.io.req.valid.poke(false.B)

      var hitSeen = false
      for (_ <- 0 until 8 if !hitSeen) {
        assert(!dut.io.allocReq.valid.peekBoolean())
        if (dut.io.resp.valid.peekBoolean()) {
          dut.io.resp.bits.resp.tag.expect(6.U)
          dut.io.resp.bits.resp.block.tris(0).v0.x.expect("h40400000".U)
          hitSeen = true
        } else {
          dut.clock.step()
        }
      }
      assert(hitSeen, "cache hit did not produce a response")
    }
  }

  it should "return all merged waiters from one refill" in {
    val c = TriPeConfig(cacheLineTriangles = 2)
    test(new TriangleMemCachedBank(c, srcWidth = 1, tagWidth = 4, bankId = 0, numBanks = 1,
      numSets = 4, ways = 1, reqQueueDepth = 4, mergeQueueDepth = 4, mshrEntries = 1,
      useXpmCacheOverride = Some(false))) { dut =>
      dut.io.req.valid.poke(false.B)
      dut.io.resp.ready.poke(true.B)
      dut.io.allocResp.valid.poke(false.B)
      dut.io.refillDone.valid.poke(false.B)
      dut.clock.step(5)

      dut.io.req.bits.addr.poke(2.U)
      dut.io.req.bits.mask.poke(3.U)
      dut.io.req.bits.src.poke(0.U)
      dut.io.req.bits.tag.poke(10.U)
      dut.io.req.valid.poke(true.B)
      while (!dut.io.req.ready.peekBoolean()) {
        dut.clock.step()
      }
      dut.clock.step()

      for (tag <- 11 until 14) {
        dut.io.req.bits.tag.poke(tag.U)
        while (!dut.io.req.ready.peekBoolean()) {
          dut.clock.step()
        }
        dut.clock.step()
      }
      dut.io.req.valid.poke(false.B)

      for (index <- 0 until 4) {
        while (!dut.io.allocReq.valid.peekBoolean()) {
          dut.clock.step()
        }
        dut.io.allocReq.bits.addr.expect(2.U)
        dut.io.allocResp.bits.id.poke(0.U)
        dut.io.allocResp.bits.merged.poke((index != 0).B)
        dut.io.allocResp.valid.poke(true.B)
        dut.clock.step()
        dut.io.allocResp.valid.poke(false.B)
      }

      val line = BigInt("3f8000004000000040400000", 16) | (BigInt("4080000040a0000040c00000", 16) << 288)
      dut.io.refillDone.bits.id.poke(0.U)
      dut.io.refillDone.bits.data.poke(line.U)
      dut.io.refillDone.valid.poke(true.B)
      while (!dut.io.refillDone.ready.peekBoolean()) {
        dut.clock.step()
      }
      dut.clock.step()
      dut.io.refillDone.valid.poke(false.B)

      var responseTags = Set.empty[BigInt]
      for (_ <- 0 until 16 if responseTags.size < 4) {
        if (dut.io.resp.valid.peekBoolean()) {
          responseTags += dut.io.resp.bits.resp.tag.peek().litValue
        }
        dut.clock.step()
      }
      assert(responseTags == Set(BigInt(10), BigInt(11), BigInt(12), BigInt(13)),
        s"one refill did not wake all merged waiters: $responseTags")
    }
  }
}
