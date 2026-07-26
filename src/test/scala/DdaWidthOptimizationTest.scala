import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec
import DDA.DdaInitPE
import raytrace_utils._
import scala.collection.mutable.ArrayBuffer

class DdaWidthOptimizationTest extends AnyFlatSpec with ChiselScalatestTester {
  behavior of "narrow DDA coordinates"

  private def fp32(value: Float): UInt =
    java.lang.Float.floatToRawIntBits(value).U(32.W)

  private case class InitVector(
    origin: (Float, Float, Float),
    dir: (Float, Float, Float),
    rayId: Int,
    frontendId: Int,
    traceSlot: Int,
    expectedSub: (Int, Int, Int),
    expectedTMax: (Float, Float, Float)
  )

  private def pokeVector(dut: DdaInitPE, v: InitVector): Unit = {
    dut.io.in.bits.ray.origin.x.poke(fp32(v.origin._1))
    dut.io.in.bits.ray.origin.y.poke(fp32(v.origin._2))
    dut.io.in.bits.ray.origin.z.poke(fp32(v.origin._3))
    dut.io.in.bits.ray.dir.x.poke(fp32(v.dir._1))
    dut.io.in.bits.ray.dir.y.poke(fp32(v.dir._2))
    dut.io.in.bits.ray.dir.z.poke(fp32(v.dir._3))
    dut.io.in.bits.ray.dist.poke(fp32(0.0f))
    dut.io.in.bits.meta.rayId.poke(v.rayId.U)
    dut.io.in.bits.frontendContextId.poke(v.frontendId.U)
    dut.io.in.bits.traceSlot.poke(v.traceSlot.U)
  }

  it should "use the configured narrow width and saturate a positive out-of-bounds coordinate" in {
    test(new DdaInitPE(FloatConfig.FP32, 32)) { dut =>
      assert(dut.io.out.bits.traversal.subX.getWidth == GlobalConfig.ddaCoordWidth)
      assert(dut.io.out.bits.traversal.subY.getWidth == GlobalConfig.ddaCoordWidth)
      assert(dut.io.out.bits.traversal.subZ.getWidth == GlobalConfig.ddaCoordWidth)

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
      dut.io.in.bits.frontendContextId.poke(5.U)
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
      dut.io.out.bits.traversal.subX.expect(GlobalConfig.DdaRes.S)
      dut.io.out.bits.traversal.subY.expect(0.S)
      dut.io.out.bits.traversal.subZ.expect(0.S)
    }
  }

  it should "share one axis pipeline and accept one ray every three cycles" in {
    test(new DdaInitPE(FloatConfig.FP32, 32)) { dut =>
      val vectors = Seq(
        InitVector(
          origin = (0.25f, 1.5f, 2.75f),
          dir = (1.0f, 1.0f, 1.0f),
          rayId = 101,
          frontendId = 1,
          traceSlot = 3,
          expectedSub = (0, 1, 2),
          expectedTMax = (0.75f, 0.5f, 0.25f)
        ),
        InitVector(
          origin = (4.25f, 5.5f, 6.75f),
          dir = (-1.0f, -1.0f, -1.0f),
          rayId = 202,
          frontendId = 2,
          traceSlot = 4,
          expectedSub = (4, 5, 6),
          expectedTMax = (0.25f, 0.5f, 0.75f)
        ),
        InitVector(
          origin = (7.5f, 8.25f, 9.75f),
          dir = (1.0f, -1.0f, 1.0f),
          rayId = 303,
          frontendId = 3,
          traceSlot = 5,
          expectedSub = (7, 8, 9),
          expectedTMax = (0.5f, 0.25f, 0.25f)
        )
      )

      dut.io.out.ready.poke(true.B)
      dut.io.in.valid.poke(false.B)
      dut.io.grid_min.x.poke(fp32(0.0f))
      dut.io.grid_min.y.poke(fp32(0.0f))
      dut.io.grid_min.z.poke(fp32(0.0f))
      dut.io.inv_sub_voxel.x.poke(fp32(1.0f))
      dut.io.inv_sub_voxel.y.poke(fp32(1.0f))
      dut.io.inv_sub_voxel.z.poke(fp32(1.0f))
      dut.clock.step()

      val fireCycles = ArrayBuffer.empty[Int]
      var nextInput = 0
      var nextOutput = 0
      var cycle = 0
      while (nextOutput < vectors.length && cycle < 128) {
        if (nextInput < vectors.length) {
          dut.io.in.valid.poke(true.B)
          pokeVector(dut, vectors(nextInput))
        } else {
          dut.io.in.valid.poke(false.B)
        }

        if (nextInput < vectors.length && dut.io.in.ready.peekBoolean()) {
          fireCycles += cycle
          nextInput += 1
        }

        if (dut.io.out.valid.peekBoolean()) {
          val expected = vectors(nextOutput)
          dut.io.out.bits.sourceFrontendId.expect(expected.frontendId.U)
          dut.io.out.bits.contextId.expect(expected.traceSlot.U)
          dut.io.out.bits.immutable.meta.rayId.expect(expected.rayId.U)
          dut.io.out.bits.immutable.origin.x.expect(fp32(expected.origin._1))
          dut.io.out.bits.immutable.origin.y.expect(fp32(expected.origin._2))
          dut.io.out.bits.immutable.origin.z.expect(fp32(expected.origin._3))
          dut.io.out.bits.immutable.dir.x.expect(fp32(expected.dir._1))
          dut.io.out.bits.immutable.dir.y.expect(fp32(expected.dir._2))
          dut.io.out.bits.immutable.dir.z.expect(fp32(expected.dir._3))
          dut.io.out.bits.traversal.contextId.expect(expected.traceSlot.U)
          dut.io.out.bits.traversal.subX.expect(expected.expectedSub._1.S)
          dut.io.out.bits.traversal.subY.expect(expected.expectedSub._2.S)
          dut.io.out.bits.traversal.subZ.expect(expected.expectedSub._3.S)
          dut.io.out.bits.traversal.tMaxX.expect(fp32(expected.expectedTMax._1))
          dut.io.out.bits.traversal.tMaxY.expect(fp32(expected.expectedTMax._2))
          dut.io.out.bits.traversal.tMaxZ.expect(fp32(expected.expectedTMax._3))
          dut.io.out.bits.traversal.tDeltaX.expect(fp32(1.0f))
          dut.io.out.bits.traversal.tDeltaY.expect(fp32(1.0f))
          dut.io.out.bits.traversal.tDeltaZ.expect(fp32(1.0f))
          dut.io.out.bits.traversal.dirNegX.expect((expected.dir._1 < 0.0f).B)
          dut.io.out.bits.traversal.dirNegY.expect((expected.dir._2 < 0.0f).B)
          dut.io.out.bits.traversal.dirNegZ.expect((expected.dir._3 < 0.0f).B)
          nextOutput += 1
        }

        dut.clock.step()
        cycle += 1
      }

      assert(nextOutput == vectors.length, s"received $nextOutput/${vectors.length} DDA init results")
      assert(fireCycles.sliding(2).forall(pair => pair(1) - pair(0) == 3),
        s"DDA init input handshakes were not three cycles apart: ${fireCycles.mkString(",")}")
    }
  }
}
