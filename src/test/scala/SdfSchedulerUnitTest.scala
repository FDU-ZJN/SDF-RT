import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec
import raytrace_utils.{FloatConfig, GlobalConfig}
import SDF.SdfSchedulerUnit

class SdfSchedulerUnitTest extends AnyFlatSpec with ChiselScalatestTester {
  behavior of "SdfSchedulerUnit"

  private val cfg = FloatConfig.FP32
  private val maxSteps = 4

  private def clearInputs(dut: SdfSchedulerUnit): Unit = {
    for (lane <- 0 until GlobalConfig.sdfStepNumWorkers) {
      dut.io.issue_in(lane).valid.poke(false.B)
      dut.io.pe_out_miss(lane).valid.poke(false.B)
      dut.io.pe_out_hit(lane).valid.poke(false.B)
      dut.io.pe_in(lane).ready.poke(true.B)
      dut.io.out(lane).ready.poke(true.B)
    }
  }

  private def pokeIssue(dut: SdfSchedulerUnit, lane: Int, rayId: Int): Unit = {
    val bits = dut.io.issue_in(lane).bits
    bits.meta.rayId.poke(rayId.U)
    bits.ray.origin.x.poke(0.U); bits.ray.origin.y.poke(0.U); bits.ray.origin.z.poke(0.U)
    bits.ray.dir.x.poke(0.U); bits.ray.dir.y.poke(0.U); bits.ray.dir.z.poke(0.U)
    bits.ray.dist.poke(0.U)
    dut.io.issue_in(lane).valid.poke(true.B)
  }

  private def pokeResp(dut: SdfSchedulerUnit, lane: Int, rayId: Int, iter: Int, hit: Boolean): Unit = {
    val bits = if (hit) dut.io.pe_out_hit(lane).bits else dut.io.pe_out_miss(lane).bits
    bits.meta.rayId.poke(rayId.U)
    bits.iter.poke(iter.U)
    bits.hit.poke(hit.B)
    bits.prevSdf.poke(0.U)
    bits.ray.origin.x.poke(0.U); bits.ray.origin.y.poke(0.U); bits.ray.origin.z.poke(0.U)
    bits.ray.dir.x.poke(0.U); bits.ray.dir.y.poke(0.U); bits.ray.dir.z.poke(0.U)
    bits.ray.dist.poke(0.U)
    if (hit) dut.io.pe_out_hit(lane).valid.poke(true.B) else dut.io.pe_out_miss(lane).valid.poke(true.B)
  }

  it should "emit one final after retrying misses and terminally complete hits or out-of-bounds" in {
    test(new SdfSchedulerUnit(cfg, 32, maxSteps)) { dut =>
      clearInputs(dut)
      dut.clock.step()

      pokeIssue(dut, 0, rayId = 7)
      dut.io.issue_in(0).ready.expect(true.B)
      dut.clock.step()
      clearInputs(dut)

      // A non-terminal miss enters retryQ, then is the next PE input.
      pokeResp(dut, 0, rayId = 7, iter = 1, hit = false)
      dut.clock.step()
      clearInputs(dut)
      dut.io.pe_in(0).valid.expect(true.B)
      dut.io.pe_in(0).bits.meta.rayId.expect(7.U)
      dut.io.pe_in(0).bits.iter.expect(1.U)
      dut.clock.step()
      clearInputs(dut)

      // maxSteps is terminal: it must not enter retryQ and produces one final.
      pokeResp(dut, 0, rayId = 7, iter = maxSteps, hit = false)
      dut.clock.step()
      clearInputs(dut)
      dut.io.out(0).valid.expect(true.B)
      dut.io.out(0).bits.meta.rayId.expect(7.U)
      dut.clock.step()
      clearInputs(dut)
      dut.io.out(0).valid.expect(false.B)

      // A hit is terminal on its first PE response and never appears in retryQ.
      pokeIssue(dut, 1, rayId = 8)
      dut.clock.step()
      clearInputs(dut)
      pokeResp(dut, 1, rayId = 8, iter = 1, hit = true)
      dut.clock.step()
      clearInputs(dut)
      dut.io.out(1).valid.expect(true.B)
      dut.io.out(1).bits.meta.rayId.expect(8.U)
      dut.clock.step()
      clearInputs(dut)
      dut.io.pe_in(1).valid.expect(false.B)
    }
  }

  it should "not resend an issue while the upstream holds valid with ready low" in {
    test(new SdfSchedulerUnit(cfg, 32, maxSteps)) { dut =>
      clearInputs(dut)
      dut.clock.step()

      pokeIssue(dut, 1, rayId = 9)
      dut.io.issue_in(1).ready.expect(true.B)
      dut.io.pe_in(1).valid.expect(true.B)
      dut.clock.step()

      dut.io.issue_in(1).ready.expect(false.B)
      dut.io.pe_in(1).valid.expect(false.B)
      dut.clock.step(4)
      dut.io.issue_in(1).ready.expect(false.B)
      dut.io.pe_in(1).valid.expect(false.B)

      dut.io.issue_in(1).valid.poke(false.B)
      pokeResp(dut, 1, rayId = 9, iter = 1, hit = true)
      dut.clock.step()
      clearInputs(dut)
      dut.io.out(1).valid.expect(true.B)
      dut.io.out(1).bits.meta.rayId.expect(9.U)
    }
  }
}
