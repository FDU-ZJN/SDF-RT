import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec
import raytrace_utils.FloatConfig
import SDF.SdfSchedulerUnit

class SdfSchedulerUnitTest extends AnyFlatSpec with ChiselScalatestTester {
  behavior of "SdfSchedulerUnit"

  private val cfg = FloatConfig.FP32
  private val maxSteps = 4

  private def clearInputs(dut: SdfSchedulerUnit): Unit = {
    dut.io.issue_in.valid.poke(false.B)
    dut.io.pe_out_miss.valid.poke(false.B)
    dut.io.pe_out_hit.valid.poke(false.B)
    dut.io.pe_in.ready.poke(true.B)
    dut.io.out.ready.poke(true.B)
  }

  private def pokeIssue(dut: SdfSchedulerUnit, rayId: Int): Unit = {
    val bits = dut.io.issue_in.bits
    bits.meta.rayId.poke(rayId.U)
    bits.ray.origin.x.poke(0.U); bits.ray.origin.y.poke(0.U); bits.ray.origin.z.poke(0.U)
    bits.ray.dir.x.poke(0.U); bits.ray.dir.y.poke(0.U); bits.ray.dir.z.poke(0.U)
    bits.ray.dist.poke(0.U)
    dut.io.issue_in.valid.poke(true.B)
  }

  private def pokeResp(dut: SdfSchedulerUnit, rayId: Int, iter: Int, hit: Boolean): Unit = {
    val bits = if (hit) dut.io.pe_out_hit.bits else dut.io.pe_out_miss.bits
    bits.meta.rayId.poke(rayId.U)
    bits.iter.poke(iter.U)
    bits.hit.poke(hit.B)
    bits.prevSdf.poke(0.U)
    bits.ray.origin.x.poke(0.U); bits.ray.origin.y.poke(0.U); bits.ray.origin.z.poke(0.U)
    bits.ray.dir.x.poke(0.U); bits.ray.dir.y.poke(0.U); bits.ray.dir.z.poke(0.U)
    bits.ray.dist.poke(0.U)
    if (hit) dut.io.pe_out_hit.valid.poke(true.B) else dut.io.pe_out_miss.valid.poke(true.B)
  }

  it should "emit one final after retrying misses and terminally complete hits or out-of-bounds" in {
    test(new SdfSchedulerUnit(cfg, 32, maxSteps)) { dut =>
      clearInputs(dut)
      dut.clock.step()

      pokeIssue(dut, rayId = 7)
      dut.io.issue_in.ready.expect(true.B)
      dut.clock.step()
      clearInputs(dut)

      // A non-terminal miss enters retryQ, then is the next PE input.
      pokeResp(dut, rayId = 7, iter = 1, hit = false)
      dut.clock.step()
      clearInputs(dut)
      dut.io.pe_in.valid.expect(true.B)
      dut.io.pe_in.bits.meta.rayId.expect(7.U)
      dut.io.pe_in.bits.iter.expect(1.U)
      dut.clock.step()
      clearInputs(dut)

      // maxSteps is terminal: it must not enter retryQ and produces one final.
      pokeResp(dut, rayId = 7, iter = maxSteps, hit = false)
      dut.clock.step()
      clearInputs(dut)
      dut.io.out.valid.expect(true.B)
      dut.io.out.bits.meta.rayId.expect(7.U)
      dut.clock.step()
      clearInputs(dut)
      dut.io.out.valid.expect(false.B)

      // A hit is terminal on its first PE response and never appears in retryQ.
      pokeIssue(dut, rayId = 8)
      dut.clock.step()
      clearInputs(dut)
      pokeResp(dut, rayId = 8, iter = 1, hit = true)
      dut.clock.step()
      clearInputs(dut)
      dut.io.out.valid.expect(true.B)
      dut.io.out.bits.meta.rayId.expect(8.U)
      dut.clock.step()
      clearInputs(dut)
      dut.io.pe_in.valid.expect(false.B)
    }
  }

  it should "prioritize a retry without accepting a new issue twice" in {
    test(new SdfSchedulerUnit(cfg, 32, maxSteps)) { dut =>
      clearInputs(dut)
      dut.clock.step()

      pokeIssue(dut, rayId = 9)
      dut.io.issue_in.ready.expect(true.B)
      dut.io.pe_in.valid.expect(true.B)
      dut.clock.step()
      clearInputs(dut)

      pokeResp(dut, rayId = 9, iter = 1, hit = false)
      dut.clock.step()
      clearInputs(dut)

      pokeIssue(dut, rayId = 10)
      dut.io.issue_in.ready.expect(false.B)
      dut.io.pe_in.valid.expect(true.B)
      dut.io.pe_in.bits.meta.rayId.expect(9.U)
      dut.clock.step()

      dut.io.issue_in.ready.expect(true.B)
      dut.io.pe_in.valid.expect(true.B)
      dut.io.pe_in.bits.meta.rayId.expect(10.U)
    }
  }
}
