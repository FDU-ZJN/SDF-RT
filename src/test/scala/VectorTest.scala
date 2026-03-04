package raytrace_utils

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec
import raytrace_utils.fudian._

class DotProductPipelineTest extends AnyFlatSpec with ChiselScalatestTester {

  def floatToUint(f: Float): BigInt = {
    val bits = java.lang.Float.floatToIntBits(f)
    BigInt(Integer.toUnsignedString(bits))
  }

  "DotProductUnit" should "handle pipelined inputs every cycle" in {
    // 使用预设延时：FMUL=3, FADD=2 => FCMA=5
    val cfg = FloatConfig.FP32
    test(new DotProductUnit(cfg)) { dut =>

      // 定义三组测试向量 (A*B)
      val inputs = Seq(
        (1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 32.0f),  // Vec0: result 32.0
        (0.5f, 0.5f, 0.5f, 2.0f, 2.0f, 2.0f, 3.0f),   // Vec1: result 3.0
        (10.0f, 0.0f, 0.0f, 1.0f, 5.0f, 5.0f, 10.0f)  // Vec2: result 10.0
      )

      dut.io.rm.poke(0.U) // RNE

      // --- 第一阶段：填满流水线 ---
      println("--- Pushing inputs into pipeline ---")
      for (i <- 0 until inputs.length) {
        val (ax, ay, az, bx, by, bz, _) = inputs(i)
        dut.io.a.x.poke(floatToUint(ax).U); dut.io.a.y.poke(floatToUint(ay).U); dut.io.a.z.poke(floatToUint(az).U)
        dut.io.b.x.poke(floatToUint(bx).U); dut.io.b.y.poke(floatToUint(by).U); dut.io.b.z.poke(floatToUint(bz).U)
        dut.clock.step(1)
      }

      // --- 第二阶段：等待第一组结果到达 (7 - 3 = 4 拍) ---
      // 之前已经 step 了 3 拍输入，还需要 4 拍
      println(s"--- Waiting for first result (Latency) ---")
      dut.clock.step(4)

      // --- 第三阶段：连续检查吞吐量 ---
      for (i <- 0 until inputs.length) {
        val expected = inputs(i)._7
        val actualBits = dut.io.res.peek().litValue.toInt
        val actualFloat = java.lang.Float.intBitsToFloat(actualBits)

        println(s"Cycle ${17 + i}: Expected $expected, Got $actualFloat")
        dut.io.res.expect(floatToUint(expected).U)

        // 理论上，一旦流水线填满，之后每个周期（或特定周期）都会出一个结果
        dut.clock.step(1)
      }
    }
  }
}

class CrossProductPipelineTest extends AnyFlatSpec with ChiselScalatestTester {

  def floatToUint(f: Float): BigInt = {
    val bits = java.lang.Float.floatToIntBits(f)
    BigInt(Integer.toUnsignedString(bits))
  }

  "CrossProductUnit" should "compute cross product with 5-cycle latency and full throughput" in {
    val cfg = FloatConfig.FP32 // FMUL=3, FADD=2
    test(new CrossProductUnit(cfg)) { dut =>

      // 测试数据准备：A x B = C
      // 1. (1,0,0) x (0,1,0) = (0,0,1)
      // 2. (1,2,3) x (4,5,6) = (-3, 6, -3)
      val inputs = Seq(
        (1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f),
        (1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, -3.0f, 6.0f, -3.0f)
      )

      dut.io.rm.poke(0.U) // RNE

      // --- 第一阶段：连续喂入数据 ---
      println("--- Pushing Vec0 ---")
      dut.io.a.x.poke(floatToUint(inputs(0)._1).U); dut.io.a.y.poke(floatToUint(inputs(0)._2).U); dut.io.a.z.poke(floatToUint(inputs(0)._3).U)
      dut.io.b.x.poke(floatToUint(inputs(0)._4).U); dut.io.b.y.poke(floatToUint(inputs(0)._5).U); dut.io.b.z.poke(floatToUint(inputs(0)._6).U)
      dut.clock.step(1)

      println("--- Pushing Vec1 ---")
      dut.io.a.x.poke(floatToUint(inputs(1)._1).U); dut.io.a.y.poke(floatToUint(inputs(1)._2).U); dut.io.a.z.poke(floatToUint(inputs(1)._3).U)
      dut.io.b.x.poke(floatToUint(inputs(1)._4).U); dut.io.b.y.poke(floatToUint(inputs(1)._5).U); dut.io.b.z.poke(floatToUint(inputs(1)._6).U)
      dut.clock.step(1)

      // --- 第二阶段：等待流水线排空 ---
      // 已经 step(2) 了，距离第一组结果(5拍)还剩 3 拍
      dut.clock.step(3)

      // --- 第三阶段：验证结果 ---
      // 此时是第 5 拍结束，应看到 Vec0 的结果
      val res0_x = java.lang.Float.intBitsToFloat(dut.io.res.x.peek().litValue.toInt)
      val res0_y = java.lang.Float.intBitsToFloat(dut.io.res.y.peek().litValue.toInt)
      val res0_z = java.lang.Float.intBitsToFloat(dut.io.res.z.peek().litValue.toInt)
      println(s"Cycle 5 (Vec0): ($res0_x, $res0_y, $res0_z)")
      dut.io.res.x.expect(floatToUint(inputs(0)._7).U)
      dut.io.res.y.expect(floatToUint(inputs(0)._8).U)
      dut.io.res.z.expect(floatToUint(inputs(0)._9).U)

      dut.clock.step(1) // 到第 6 拍

      // 应看到 Vec1 的结果
      val res1_x = java.lang.Float.intBitsToFloat(dut.io.res.x.peek().litValue.toInt)
      val res1_y = java.lang.Float.intBitsToFloat(dut.io.res.y.peek().litValue.toInt)
      val res1_z = java.lang.Float.intBitsToFloat(dut.io.res.z.peek().litValue.toInt)
      println(s"Cycle 6 (Vec1): ($res1_x, $res1_y, $res1_z)")
      dut.io.res.x.expect(floatToUint(inputs(1)._7).U)
      dut.io.res.y.expect(floatToUint(inputs(1)._8).U)
      dut.io.res.z.expect(floatToUint(inputs(1)._9).U)
    }
  }
}