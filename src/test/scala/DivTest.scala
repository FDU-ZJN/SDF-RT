package raytrace_utils

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec
class FDIVTest extends AnyFlatSpec with ChiselScalatestTester {

  // 辅助函数：将 Java Float 转换为 BigInt 对应的位模式
  def fToHex(f: Float): BigInt = {
    val bits = java.lang.Float.floatToRawIntBits(f)
    (BigInt(bits) & 0xFFFFFFFFL)
  }

  "FDIV" should "match precision and latency" in {
    // 使用默认配置，假设 fdivLatency = 6
    test(new FDIV(FloatConfig.FP32)) { c =>

      // 测试用例 1: 4.0 / 2.0 = 2.0
      val valA = 4.0f
      val valB = 2.0f
      val expected = 2.0f

      c.io.in_valid.poke(true.B)
      c.io.a.poke(fToHex(valA).U)
      c.io.b.poke(fToHex(valB).U)

      c.clock.step(1)
      c.io.in_valid.poke(false.B) // 只打入一拍

      // 等待剩余的延迟 (6 - 1 = 5 拍)
      c.clock.step(5)

      // 验证输出
      c.io.out_valid.expect(true.B)
      c.io.result.expect(fToHex(expected).U)

      // 测试用例 2: 1.0 / 3.0 (验证 RNE 舍入)
      // 1/3 在 FP32 下是 0.33333334 (0x3eaaaaab)
      val valA2 = 1.0f
      val valB2 = 3.0f
      val expected2 = 1.0f / 3.0f

      c.io.in_valid.poke(true.B)
      c.io.a.poke(fToHex(valA2).U)
      c.io.b.poke(fToHex(valB2).U)
      c.clock.step(1)
      c.io.in_valid.poke(false.B)

      c.clock.step(5)
      c.io.out_valid.expect(true.B)
      c.io.result.expect(fToHex(expected2).U)
    }
  }

  it should "handle continuous pipeline" in {
    test(new FDIV(FloatConfig.FP32)) { c =>
      // 测试流水线满载情况
      val inputs = Seq((10.0f, 2.0f), (9.0f, 3.0f), (8.0f, 4.0f))
      val outputs = inputs.map(x => x._1 / x._2)

      // 连续输入 3 拍
      inputs.foreach { case (a, b) =>
        c.io.in_valid.poke(true.B)
        c.io.a.poke(fToHex(a).U)
        c.io.b.poke(fToHex(b).U)
        c.clock.step(1)
      }
      c.io.in_valid.poke(false.B)

      // 在第 6 拍（从第一个数据输入算起）开始观察输出
      c.clock.step(6 - inputs.length)

      outputs.foreach { exp =>
        c.io.out_valid.expect(true.B)
        c.io.result.expect(fToHex(exp).U)
        c.clock.step(1)
      }
    }
  }
}