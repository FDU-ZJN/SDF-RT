import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

import SDF.{InitStage, SdfPE}
import raytrace_utils.{FloatConfig, SdfPeConfig}

class InitSdfCModelParityTest extends AnyFlatSpec with ChiselScalatestTester {
  private val cfg = FloatConfig.FP32
  private def bits(value: Float): BigInt = BigInt(java.lang.Float.floatToRawIntBits(value) & 0xffffffffL)

  behavior of "InitStage C model parity vectors"

  it should "advance a positive AABB entry by the same raw FP32 fraction step" in {
    test(new InitStage(cfg, 32)) { dut =>
      dut.io.setup_origin.x.poke(bits(0.0f).U)
      dut.io.setup_origin.y.poke(bits(0.0f).U)
      dut.io.setup_origin.z.poke(bits(2.0f).U)
      dut.io.setup_grid_min.x.poke(bits(-1.0f).U)
      dut.io.setup_grid_min.y.poke(bits(-1.0f).U)
      dut.io.setup_grid_min.z.poke(bits(-1.0f).U)
      dut.io.setup_grid_max.x.poke(bits(1.0f).U)
      dut.io.setup_grid_max.y.poke(bits(1.0f).U)
      dut.io.setup_grid_max.z.poke(bits(1.0f).U)
      dut.io.out.ready.poke(true.B)
      dut.io.in.valid.poke(false.B)
      dut.clock.step()

      dut.io.in.bits.ray.origin.x.poke(bits(0.0f).U)
      dut.io.in.bits.ray.origin.y.poke(bits(0.0f).U)
      dut.io.in.bits.ray.origin.z.poke(bits(2.0f).U)
      dut.io.in.bits.ray.dir.x.poke(bits(0.0f).U)
      dut.io.in.bits.ray.dir.y.poke(bits(0.0f).U)
      dut.io.in.bits.ray.dir.z.poke(bits(-1.0f).U)
      dut.io.in.bits.meta.rayId.poke(7.U)
      dut.io.in.valid.poke(true.B)
      while (!dut.io.in.ready.peek().litToBoolean) dut.clock.step()
      dut.clock.step()
      dut.io.in.valid.poke(false.B)

      var waited = 0
      while (!dut.io.out.valid.peek().litToBoolean && waited < 100) {
        dut.clock.step()
        waited += 1
      }
      assert(waited < 100, "InitStage response timed out")
      dut.io.out.bits.hit.expect(true.B)
      dut.io.out.bits.ray.dist.expect(bits(1.0f + 1.0f / 16384.0f).U)
      dut.io.out.bits.meta.rayId.expect(7.U)
    }
  }

  behavior of "SdfPE C model parity vectors"

  it should "accept positive hits and treat a final-step negative as a hit at the current t" in {
    val peCfg = SdfPeConfig(
      cfg = cfg,
      GlobalResX = 2, GlobalResY = 2, GlobalResZ = 2,
      LocalResX = 1, LocalResY = 1, LocalResZ = 1,
      maxSteps = 4,
      threshold1 = 0.3f, threshold2 = 0.3f, threshold3 = 0.3f)
    test(new SdfPE(peCfg)) { dut =>
      dut.io.grid_min.x.poke(bits(-1.0f).U)
      dut.io.grid_min.y.poke(bits(-1.0f).U)
      dut.io.grid_min.z.poke(bits(-1.0f).U)
      dut.io.inv_voxel.x.poke(bits(1.0f).U)
      dut.io.inv_voxel.y.poke(bits(1.0f).U)
      dut.io.inv_voxel.z.poke(bits(1.0f).U)
      dut.io.sdf_mem_req.ready.poke(true.B)
      dut.io.sdf_mem_resp.valid.poke(true.B)
      dut.io.sdf_mem_resp.bits.poke(bits(0.25f).U)
      dut.io.out.ready.poke(true.B)
      dut.io.out_hit.ready.poke(true.B)
      dut.io.in.valid.poke(false.B)
      dut.clock.step()

      def issue(dist: Float, prevSdf: Float, rayId: Int, iter: Int = 0): Unit = {
        dut.io.in.bits.ray.origin.x.poke(bits(0.0f).U)
        dut.io.in.bits.ray.origin.y.poke(bits(0.0f).U)
        dut.io.in.bits.ray.origin.z.poke(bits(0.0f).U)
        dut.io.in.bits.ray.dir.x.poke(bits(0.0f).U)
        dut.io.in.bits.ray.dir.y.poke(bits(0.0f).U)
        dut.io.in.bits.ray.dir.z.poke(bits(0.0f).U)
        dut.io.in.bits.ray.dist.poke(bits(dist).U)
        dut.io.in.bits.meta.rayId.poke(rayId.U)
        dut.io.in.bits.iter.poke(iter.U)
        dut.io.in.bits.prevSdf.poke(bits(prevSdf).U)
        dut.io.in.valid.poke(true.B)
        dut.clock.step()
        dut.io.in.valid.poke(false.B)
      }

      issue(0.5f, 0.0f, 11)
      var waited = 0
      while (!dut.io.out_hit.valid.peek().litToBoolean && waited < 100) {
        dut.clock.step()
        waited += 1
      }
      assert(waited < 100, "positive SDF hit timed out")
      dut.io.out_hit.bits.ray.dist.expect(bits(0.5f).U)
      dut.io.out_hit.bits.iter.expect(1.U)
      dut.clock.step()

      dut.io.sdf_mem_resp.bits.poke(bits(-0.25f).U)
      issue(0.5f, 0.5f, 12)
      waited = 0
      while (!dut.io.out.valid.peek().litToBoolean && waited < 100) {
        assert(!dut.io.out_hit.valid.peek().litToBoolean, "negative SDF unexpectedly counted as a hit")
        dut.clock.step()
        waited += 1
      }
      assert(waited < 100, "negative SDF miss timed out")
      dut.io.out.bits.hit.expect(false.B)
      dut.io.out.bits.ray.dist.expect(bits(0.25f).U)
      dut.io.out.bits.iter.expect(1.U)
      dut.io.out.bits.prevSdf.expect(bits(0.25f).U)
      dut.clock.step()

      // A negative value on the last allowed sample terminates as an SDF hit.
      // Its output t is the current positive-to-negative crossing point; the
      // speculative prevSdf/2 backoff is not committed.
      issue(0.5f, 0.5f, 13, iter = 3)
      waited = 0
      while (!dut.io.out_hit.valid.peek().litToBoolean && waited < 100) {
        dut.clock.step()
        waited += 1
      }
      assert(waited < 100, "final-step negative SDF hit timed out")
      dut.io.out_hit.bits.ray.dist.expect(bits(0.5f).U)
      dut.io.out_hit.bits.iter.expect(4.U)
    }
  }
}
