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

  it should "recheck a parked hit when a refill replaces the same direct-mapped set" in {
    val c = TriPeConfig(cacheLineTriangles = 1)
    test(new TriangleMemCachedBank(c, srcWidth = 1, tagWidth = 4, bankId = 0, numBanks = 1,
      numSets = 2, ways = 1, reqQueueDepth = 4, mergeQueueDepth = 2, mshrEntries = 2,
      useXpmCacheOverride = Some(false))) { dut =>
      dut.io.req.valid.poke(false.B)
      dut.io.resp.ready.poke(true.B)
      dut.io.allocResp.valid.poke(false.B)
      dut.io.refillDone.valid.poke(false.B)
      dut.clock.step(5)

      def request(addr: Int, tag: Int): Unit = {
        dut.io.req.bits.addr.poke(addr.U)
        dut.io.req.bits.mask.poke(1.U)
        dut.io.req.bits.src.poke(0.U)
        dut.io.req.bits.tag.poke(tag.U)
        dut.io.req.valid.poke(true.B)
        while (!dut.io.req.ready.peekBoolean()) dut.clock.step()
        dut.clock.step()
        dut.io.req.valid.poke(false.B)
      }

      def allocate(expectedAddr: Int, id: Int): Unit = {
        while (!dut.io.allocReq.valid.peekBoolean()) dut.clock.step()
        dut.io.allocReq.bits.addr.expect(expectedAddr.U)
        dut.io.allocResp.bits.id.poke(id.U)
        dut.io.allocResp.bits.merged.poke(false.B)
        dut.io.allocResp.valid.poke(true.B)
        dut.clock.step()
        dut.io.allocResp.valid.poke(false.B)
      }

      def refill(id: Int, data: BigInt): Unit = {
        dut.io.refillDone.bits.id.poke(id.U)
        dut.io.refillDone.bits.data.poke(data.U)
        dut.io.refillDone.valid.poke(true.B)
        while (!dut.io.refillDone.ready.peekBoolean()) dut.clock.step()
        dut.clock.step()
        dut.io.refillDone.valid.poke(false.B)
      }

      val lineA = BigInt("3f800000", 16)
      val lineB = BigInt("40000000", 16)

      request(addr = 0, tag = 1)
      allocate(expectedAddr = 0, id = 0)
      refill(id = 0, data = lineA)
      while (!dut.io.resp.valid.peekBoolean()) dut.clock.step()
      dut.io.resp.bits.resp.block.tris(0).v0.x.expect(lineA.U)
      dut.clock.step()

      // Start a miss that will replace addr 0, but hold its refill until an
      // addr-0 hit has completed tag lookup and is parked before data read.
      request(addr = 2, tag = 2)
      allocate(expectedAddr = 2, id = 1)
      request(addr = 0, tag = 3)
      while (!(dut.io.stat.valid.peekBoolean() && dut.io.stat.bits.peekBoolean())) {
        dut.clock.step()
      }

      dut.io.refillDone.bits.id.poke(1.U)
      dut.io.refillDone.bits.data.poke(lineB.U)
      dut.io.refillDone.valid.poke(true.B)
      dut.clock.step()
      dut.io.refillDone.valid.poke(false.B)

      // The parked request must be replayed as an addr-0 miss, never returned
      // with addr-1's newly written data.
      allocate(expectedAddr = 0, id = 0)
      refill(id = 0, data = lineA)

      var sawTag3 = false
      for (_ <- 0 until 20 if !sawTag3) {
        if (dut.io.resp.valid.peekBoolean() && dut.io.resp.bits.resp.tag.peek().litValue == 3) {
          dut.io.resp.bits.resp.block.tris(0).v0.x.expect(lineA.U)
          sawTag3 = true
        }
        dut.clock.step()
      }
      assert(sawTag3, "parked hit was not replayed after the conflicting refill")
    }
  }

  it should "not apply an advancing lookup refill bypass to the next request" in {
    val c = TriPeConfig(cacheLineTriangles = 1)
    test(new TriangleMemCachedBank(c, srcWidth = 1, tagWidth = 4, bankId = 0, numBanks = 1,
      numSets = 2, ways = 1, reqQueueDepth = 4, mergeQueueDepth = 2, mshrEntries = 3,
      useXpmCacheOverride = Some(false))) { dut =>
      dut.io.req.valid.poke(false.B)
      dut.io.resp.ready.poke(true.B)
      dut.io.allocResp.valid.poke(false.B)
      dut.io.refillDone.valid.poke(false.B)
      dut.clock.step(5)

      def request(addr: Int, tag: Int): Unit = {
        dut.io.req.bits.addr.poke(addr.U)
        dut.io.req.bits.mask.poke(1.U)
        dut.io.req.bits.src.poke(0.U)
        dut.io.req.bits.tag.poke(tag.U)
        dut.io.req.valid.poke(true.B)
        while (!dut.io.req.ready.peekBoolean()) dut.clock.step()
        dut.clock.step()
        dut.io.req.valid.poke(false.B)
      }

      def allocate(expectedAddr: Int, id: Int): Unit = {
        while (!dut.io.allocReq.valid.peekBoolean()) dut.clock.step()
        dut.io.allocReq.bits.addr.expect(expectedAddr.U)
        dut.io.allocResp.bits.id.poke(id.U)
        dut.io.allocResp.bits.merged.poke(false.B)
        dut.io.allocResp.valid.poke(true.B)
        dut.clock.step()
        dut.io.allocResp.valid.poke(false.B)
      }

      def refill(id: Int, data: BigInt): Unit = {
        dut.io.refillDone.bits.id.poke(id.U)
        dut.io.refillDone.bits.data.poke(data.U)
        dut.io.refillDone.valid.poke(true.B)
        while (!dut.io.refillDone.ready.peekBoolean()) dut.clock.step()
        dut.clock.step()
        dut.io.refillDone.valid.poke(false.B)
      }

      val lineSet1 = BigInt("3f800000", 16)
      val lineSet0 = BigInt("40000000", 16)

      // Cache addr 1 (set 1, tag 0), then hold a refill for addr 2
      // (set 0, tag 1).
      request(addr = 1, tag = 1)
      allocate(expectedAddr = 1, id = 0)
      refill(id = 0, data = lineSet1)
      while (!dut.io.resp.valid.peekBoolean()) dut.clock.step()
      dut.clock.step()

      request(addr = 2, tag = 2)
      allocate(expectedAddr = 2, id = 1)

      // addr 4 is an old miss in set 0.  addr 3 follows it and must miss in
      // set 1; its requested tag happens to equal addr 2's refill tag, which
      // exposes an incorrectly carried refill bypass as a false hit.
      dut.io.req.bits.addr.poke(4.U)
      dut.io.req.bits.mask.poke(1.U)
      dut.io.req.bits.src.poke(0.U)
      dut.io.req.bits.tag.poke(3.U)
      dut.io.req.valid.poke(true.B)
      while (!dut.io.req.ready.peekBoolean()) dut.clock.step()
      dut.clock.step()
      dut.io.req.bits.addr.poke(3.U)
      dut.io.req.bits.tag.poke(4.U)
      while (!dut.io.req.ready.peekBoolean()) dut.clock.step()
      dut.clock.step()
      dut.io.req.valid.poke(false.B)

      while (!(dut.io.stat.valid.peekBoolean() && !dut.io.stat.bits.peekBoolean())) {
        dut.clock.step()
      }
      dut.io.refillDone.bits.id.poke(1.U)
      dut.io.refillDone.bits.data.poke(lineSet0.U)
      dut.io.refillDone.valid.poke(true.B)
      dut.clock.step()
      dut.io.refillDone.valid.poke(false.B)

      allocate(expectedAddr = 4, id = 2)
      // The following addr-3 request must also allocate.  A false cache hit
      // here is the tag-snapshot corruption that caused wrong PPM colors.
      allocate(expectedAddr = 3, id = 0)
    }
  }

}
