import sdf_rt.RayTriangleIntersection
import raytrace_utils._
import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec
import java.lang.Float.floatToRawIntBits

class TriangleIntersectTest extends AnyFlatSpec with ChiselScalatestTester {

  // 辅助函数：将 Scala Float 转换为 Chisel UInt
  def fp32(f: Float): UInt = {
    (floatToRawIntBits(f).toLong & 0xFFFFFFFFL).U(32.W)
  }

  "RayTriangleIntersection" should "correctly detect an intersection in the center of a triangle" in {
    val cfg = FloatConfig.FP32
    test(new RayTriangleIntersection(cfg)) { du =>
      // --- 设置输入数据 ---
      // 射线：起点 (0, 0, -5), 方向 (0, 0, 1) -> 指向正前方
      du.io.orig.x.poke(fp32(0.0f)); du.io.orig.y.poke(fp32(0.0f)); du.io.orig.z.poke(fp32(-5.0f))
      du.io.dir.x.poke(fp32(0.0f));  du.io.dir.y.poke(fp32(0.0f));  du.io.dir.z.poke(fp32(1.0f))

      // 三角形：V0(-1, -1, 0), V1(1, -1, 0), V2(0, 1, 0) -> 位于原点附近的 XY 平面
      du.io.v0.x.poke(fp32(-1.0f)); du.io.v0.y.poke(fp32(-1.0f)); du.io.v0.z.poke(fp32(0.0f))
      du.io.v1.x.poke(fp32(1.0f));  du.io.v1.y.poke(fp32(-1.0f)); du.io.v1.z.poke(fp32(0.0f))
      du.io.v2.x.poke(fp32(0.0f));  du.io.v2.y.poke(fp32(1.0f));  du.io.v2.z.poke(fp32(0.0f))

      du.io.rm.poke(0.U) // RNE 模式
      du.io.in_valid.poke(true.B)

      // --- 计算并等待流水线 ---
      // 根据之前的设计，总延迟为 23 拍
      val latency = 23

      // 运行一个周期使输入生效
      du.clock.step(1)
      du.io.in_valid.poke(false.B)

      // 等待剩余的流水线步长
      du.clock.step(latency - 1)

      // --- 验证结果 ---
      // 预期：射线在 t=5.0 处击中三角形中心 (u=0.25, v=0.5 左右，取决于具体 E1, E2 定义)
      du.io.out_valid.expect(true.B)
      du.io.hit.expect(true.B)

      // 打印输出值（十六进制）方便调试
      println(s"Result T: ${du.io.t.peek().litValue.toString(16)}")
      println(s"Result U: ${du.io.u.peek().litValue.toString(16)}")
      println(s"Result V: ${du.io.v.peek().litValue.toString(16)}")

      // 检查 T 是否接近 5.0 (0x40A00000)
      du.io.t.expect(fp32(5.0f))
    }
  }

  "RayTriangleIntersection" should "not hit when ray points away" in {
    val cfg = FloatConfig.FP32
    test(new RayTriangleIntersection(cfg)) { du =>
      // 射线方向背向三角形 (0, 0, -1)
      du.io.orig.x.poke(fp32(0.0f)); du.io.orig.y.poke(fp32(0.0f)); du.io.orig.z.poke(fp32(-5.0f))
      du.io.dir.x.poke(fp32(0.0f));  du.io.dir.y.poke(fp32(0.0f));  du.io.dir.z.poke(fp32(-1.0f))

      du.io.v0.x.poke(fp32(-1.0f)); du.io.v0.y.poke(fp32(-1.0f)); du.io.v0.z.poke(fp32(0.0f))
      du.io.v1.x.poke(fp32(1.0f));  du.io.v1.y.poke(fp32(-1.0f)); du.io.v1.z.poke(fp32(0.0f))
      du.io.v2.x.poke(fp32(0.0f));  du.io.v2.y.poke(fp32(1.0f));  du.io.v2.z.poke(fp32(0.0f))

      du.io.in_valid.poke(true.B)
      du.clock.step(24) // 步进足够长的距离

      du.io.hit.expect(false.B)
    }
  }
}