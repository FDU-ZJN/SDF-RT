import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec
import DDA.DdaInitPE
import raytrace_utils._

class DdaWidthOptimizationTest extends AnyFlatSpec with ChiselScalatestTester {
  behavior of "narrow DDA coordinates"

  private def fp32(value: Float): UInt =
    java.lang.Float.floatToRawIntBits(value).U(32.W)

  it should "use seven bits and saturate a positive out-of-bounds coordinate" in {
    test(new DdaInitPE(FloatConfig.FP32, 32)) { dut =>
      assert(dut.io.out.bits.subX.getWidth == 7)
      assert(dut.io.out.bits.subY.getWidth == 7)
      assert(dut.io.out.bits.subZ.getWidth == 7)

      dut.io.out.ready.poke(true.B)
      dut.io.in.valid.poke(false.B)
      dut.io.grid_min.x.poke(fp32(0.0f))
      dut.io.grid_min.y.poke(fp32(0.0f))
      dut.io.grid_min.z.poke(fp32(0.0f))
      dut.io.inv_sub_voxel.x.poke(fp32(1.0f))
      dut.io.inv_sub_voxel.y.poke(fp32(1.0f))
      dut.io.inv_sub_voxel.z.poke(fp32(1.0f))

      dut.io.in.bits.ray.origin.x.poke(fp32(100.0f))
      dut.io.in.bits.ray.origin.y.poke(fp32(0.25f))
      dut.io.in.bits.ray.origin.z.poke(fp32(0.25f))
      dut.io.in.bits.ray.dir.x.poke(fp32(1.0f))
      dut.io.in.bits.ray.dir.y.poke(fp32(1.0f))
      dut.io.in.bits.ray.dir.z.poke(fp32(1.0f))
      dut.io.in.bits.ray.dist.poke(fp32(0.0f))
      dut.io.in.bits.meta.rayId.poke(7.U)
      dut.io.in.bits.traceSlot.poke(3.U)
      dut.io.in.valid.poke(true.B)
      dut.clock.step()
      dut.io.in.valid.poke(false.B)

      var waited = 0
      while (!dut.io.out.valid.peekBoolean() && waited < 64) {
        dut.clock.step()
        waited += 1
      }
      assert(dut.io.out.valid.peekBoolean(), "DDA init result did not arrive")
      dut.io.out.bits.subX.expect(GlobalConfig.DdaRes.S)
      dut.io.out.bits.subY.expect(0.S)
      dut.io.out.bits.subZ.expect(0.S)
    }
  }
}
