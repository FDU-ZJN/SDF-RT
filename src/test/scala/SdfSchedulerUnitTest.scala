import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec
import scala.collection.mutable
import raytrace_utils._
import SDF.SdfSchedulerUnit

class SdfSchedulerUnitTest extends AnyFlatSpec with ChiselScalatestTester {
  behavior of "SdfSchedulerUnit"

  private val cfg = FloatConfig.FP32

  private def clearInputs(dut: SdfSchedulerUnit): Unit = {
    dut.io.issue_in.valid.poke(false.B)
    dut.io.pe_out.valid.poke(false.B)
    dut.io.pe_in.ready.poke(true.B)
    dut.io.out.ready.poke(true.B)
    dut.io.immutable.origin.x.poke(0.U); dut.io.immutable.origin.y.poke(0.U); dut.io.immutable.origin.z.poke(0.U)
    dut.io.immutable.dir.x.poke(0.U); dut.io.immutable.dir.y.poke(0.U); dut.io.immutable.dir.z.poke(0.U)
    dut.io.immutable.meta.rayId.poke(0.U)
    dut.io.sdf_state.dist.poke(0.U)
    dut.io.sdf_state.iter.poke(0.U)
    dut.io.sdf_state.prevSdf.poke(0.U)
    dut.io.final_state.dist.poke(0.U)
    dut.io.final_state.iter.poke(0.U)
    dut.io.final_state.prevSdf.poke(0.U)
  }

  private def pokeResponse(
    dut: SdfSchedulerUnit,
    id: Int,
    iter: Int,
    done: Boolean,
    hit: Boolean,
    dist: BigInt = 0
  ): Unit = {
    dut.io.pe_out.valid.poke(true.B)
    dut.io.pe_out.bits.contextId.poke(id.U)
    dut.io.pe_out.bits.state.dist.poke(dist.U)
    dut.io.pe_out.bits.state.iter.poke(iter.U)
    dut.io.pe_out.bits.state.prevSdf.poke(0.U)
    dut.io.pe_out.bits.done.poke(done.B)
    dut.io.pe_out.bits.hit.poke(hit.B)
  }

  it should "overwrite one context step at a time and hold terminal contexts" in {
    test(new SdfSchedulerUnit(cfg, 32)) { dut =>
      clearInputs(dut)
      dut.clock.step()

      dut.io.issue_in.valid.poke(true.B)
      dut.io.issue_in.bits.poke(7.U)
      dut.io.issue_in.ready.expect(true.B)
      dut.io.pe_in.valid.expect(true.B)
      dut.io.pe_in.bits.contextId.expect(7.U)
      dut.clock.step()
      clearInputs(dut)

      pokeResponse(dut, id = 7, iter = 1, done = false, hit = false, dist = 0x3f000000L)
      dut.io.state_write.valid.expect(true.B)
      dut.io.state_write.bits.contextId.expect(7.U)
      dut.io.state_write.bits.state.iter.expect(1.U)
      dut.clock.step()
      clearInputs(dut)

      dut.io.sdf_state.dist.poke("h3f000000".U)
      dut.io.sdf_state.iter.poke(1.U)
      dut.io.pe_in.valid.expect(true.B)
      dut.io.pe_in.bits.contextId.expect(7.U)
      dut.io.pe_in.bits.state.iter.expect(1.U)
      dut.clock.step()
      clearInputs(dut)

      pokeResponse(dut, id = 7, iter = 2, done = true, hit = false, dist = 0x3f400000L)
      dut.clock.step()
      clearInputs(dut)
      dut.io.final_state.dist.poke("h3f400000".U)
      dut.io.out.valid.expect(true.B)
      dut.io.out.bits.contextId.expect(7.U)
      dut.io.out.bits.hit.expect(false.B)
      dut.io.out.bits.dist.expect("h3f400000".U)
      dut.clock.step()
      clearInputs(dut)
      dut.io.out.valid.expect(false.B)
    }
  }

  it should "accept a new context while another context is written back" in {
    test(new SdfSchedulerUnit(cfg, 32)) { dut =>
      clearInputs(dut)
      dut.io.issue_in.valid.poke(true.B)
      dut.io.issue_in.bits.poke(9.U)
      dut.clock.step()
      clearInputs(dut)

      pokeResponse(dut, id = 9, iter = 1, done = false, hit = false)
      dut.io.issue_in.valid.poke(true.B)
      dut.io.issue_in.bits.poke(10.U)
      dut.io.issue_in.ready.expect(true.B)
      dut.io.pe_in.valid.expect(true.B)
      dut.io.pe_in.bits.contextId.expect(10.U)
      dut.clock.step()
      clearInputs(dut)

      dut.io.sdf_state.iter.poke(1.U)
      dut.io.pe_in.valid.expect(true.B)
      dut.io.pe_in.bits.contextId.expect(9.U)
    }
  }

  it should "hold terminal contexts across result and backend backpressure" in {
    test(new SdfSchedulerUnit(cfg, 32)) { dut =>
      clearInputs(dut)
      dut.io.issue_in.valid.poke(true.B)
      dut.io.issue_in.bits.poke(1.U)
      dut.clock.step()
      dut.io.issue_in.bits.poke(2.U)
      dut.clock.step()
      clearInputs(dut)

      dut.io.out.ready.poke(false.B)
      pokeResponse(dut, id = 1, iter = 1, done = true, hit = true)
      dut.clock.step()
      clearInputs(dut)
      dut.io.out.ready.poke(false.B)
      pokeResponse(dut, id = 2, iter = 1, done = true, hit = false)
      dut.clock.step()
      clearInputs(dut)

      dut.io.out.ready.poke(false.B)
      for (_ <- 0 until 5) {
        dut.io.out.valid.expect(true.B)
        dut.io.out.bits.contextId.expect(1.U)
        dut.io.out.bits.hit.expect(true.B)
        dut.clock.step()
      }
      dut.io.out.ready.poke(true.B)
      dut.clock.step()
      dut.io.out.valid.expect(true.B)
      dut.io.out.bits.contextId.expect(2.U)
      dut.io.out.bits.hit.expect(false.B)
    }
  }

  it should "sustain one interleaved step per cycle with 32 contexts" in {
    test(new SdfSchedulerUnit(cfg, 32)) { dut =>
      clearInputs(dut)
      val stateIter = Array.fill(GlobalConfig.frontendContextCount)(0)
      val returns = mutable.Map.empty[Int, (Int, Int)]
      // Non-IP FP32 uses the longer supported SDF recurrence latency.
      val responseLatency = 28
      var issuedSteps = 0

      for (cycle <- 0 until 128) {
        clearInputs(dut)
        returns.get(cycle).foreach { case (id, nextIter) =>
          pokeResponse(dut, id, nextIter, done = false, hit = false)
        }
        if (cycle < GlobalConfig.frontendContextCount) {
          dut.io.issue_in.valid.poke(true.B)
          dut.io.issue_in.bits.poke(cycle.U)
          dut.io.issue_in.ready.expect(true.B)
        }

        dut.io.pe_in.valid.expect(true.B)
        val selected = dut.io.pe_in.bits.contextId.peek().litValue.toInt
        dut.io.sdf_state.iter.poke(stateIter(selected).U)
        dut.io.pe_in.bits.state.iter.expect(stateIter(selected).U)
        returns(cycle + responseLatency) = (selected, stateIter(selected) + 1)
        issuedSteps += 1

        dut.clock.step()
        returns.get(cycle).foreach { case (id, nextIter) => stateIter(id) = nextIter }
      }
      assert(issuedSteps == 128)
    }
  }
}
