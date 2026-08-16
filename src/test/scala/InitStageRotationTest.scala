import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec
import SDF.InitStage
import raytrace_utils._

class InitStageRotationTest extends AnyFlatSpec with ChiselScalatestTester {
  behavior of "InitStage direction rotation"

  private def bits(value: Float): BigInt =
    BigInt(java.lang.Float.floatToRawIntBits(value) & 0xffffffffL)

  private def floatValue(value: BigInt): Float =
    java.lang.Float.intBitsToFloat(value.toInt)

  it should "rotate back-to-back directions around Y and preserve context order" in {
    val cfg = FloatConfig.FP32.copy(useFloatIP = false)
    test(new InitStage(cfg, GlobalConfig.addrWidth)) { dut =>
      dut.io.setup_origin.x.poke(0.U)
      dut.io.setup_origin.y.poke(0.U)
      dut.io.setup_origin.z.poke(0.U)
      for (axis <- Seq(dut.io.setup_grid_min.x, dut.io.setup_grid_min.y, dut.io.setup_grid_min.z)) {
        axis.poke(bits(-2.0f).U)
      }
      for (axis <- Seq(dut.io.setup_grid_max.x, dut.io.setup_grid_max.y, dut.io.setup_grid_max.z)) {
        axis.poke(bits(2.0f).U)
      }
      dut.io.rotation_cos.poke(bits(0.0f).U)
      dut.io.rotation_sin.poke(bits(1.0f).U)
      dut.io.immutable.origin.x.poke(0.U)
      dut.io.immutable.origin.y.poke(0.U)
      dut.io.immutable.origin.z.poke(0.U)
      dut.io.immutable.meta.rayId.poke(0.U)

      val inputs = Seq(
        (3, (1.0f, 0.25f, 0.0f), (0.0f, 0.25f, -1.0f)),
        (7, (0.0f, -0.5f, 1.0f), (1.0f, -0.5f, 0.0f))
      )
      for ((context, direction, _) <- inputs) {
        dut.io.in.valid.poke(true.B)
        dut.io.in.bits.poke(context.U)
        dut.io.immutable.dir.x.poke(bits(direction._1).U)
        dut.io.immutable.dir.y.poke(bits(direction._2).U)
        dut.io.immutable.dir.z.poke(bits(direction._3).U)
        dut.clock.step()
      }
      dut.io.in.valid.poke(false.B)

      var received = 0
      var cycles = 0
      while (received < inputs.size && cycles < 100) {
        if (dut.io.out.valid.peek().litToBoolean) {
          val (expectedContext, _, expectedDirection) = inputs(received)
          dut.io.out.bits.contextId.expect(expectedContext.U)
          val actual = Seq(
            floatValue(dut.io.out.bits.transformedDir.x.peek().litValue),
            floatValue(dut.io.out.bits.transformedDir.y.peek().litValue),
            floatValue(dut.io.out.bits.transformedDir.z.peek().litValue)
          )
          val expected = Seq(expectedDirection._1, expectedDirection._2, expectedDirection._3)
          actual.zip(expected).foreach { case (got, want) =>
            assert(math.abs(got - want) < 1.0e-5f, s"direction mismatch: got=$got want=$want")
          }
          received += 1
        }
        dut.clock.step()
        cycles += 1
      }
      assert(received == inputs.size, s"received $received/${inputs.size} Init responses")
    }
  }
}
