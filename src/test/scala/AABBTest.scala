import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec
import raytrace_utils._
import raytrace_utils.fudian._
import sdf_rt._

import scala.collection.mutable.ListBuffer

class RayAABBIntersectionPipelineTest extends AnyFlatSpec with ChiselScalatestTester {

  def f2u(f: Float): UInt = (java.lang.Float.floatToIntBits(f).toLong & 0xFFFFFFFFL).U(32.W)
  def u2f(u: UInt): Float = java.lang.Float.intBitsToFloat(u.peek().litValue.toInt)

  // 复用你的软件模型
  def intersectAABBSw(orig: (Float, Float, Float), dir: (Float, Float, Float),
                      min: (Float, Float, Float), max: (Float, Float, Float)): (Boolean, Float, Float) = {
    val eps = 1e-9f
    val invDir = (1.0f/(dir._1+eps), 1.0f/(dir._2+eps), 1.0f/(dir._3+eps))
    val t0 = ((min._1-orig._1)*invDir._1, (min._2-orig._2)*invDir._2, (min._3-orig._3)*invDir._3)
    val t1 = ((max._1-orig._1)*invDir._1, (max._2-orig._2)*invDir._2, (max._3-orig._3)*invDir._3)
    val tMin = math.max(math.min(t0._1, t1._1), math.max(math.min(t0._2, t1._2), math.min(t0._3, t1._3)))
    val tMax = math.min(math.max(t0._1, t1._1), math.min(math.max(t0._2, t1._2), math.max(t0._3, t1._3)))
    val hit = (tMax >= tMin) && (tMax >= 0)
    val tNear = if (tMin > 0) tMin else tMax
    (hit, tNear, tMax)
  }

  it should "handle a continuous stream of ray intersections" in {
    val cfg = FloatConfig.FP32
    test(new RayAABBIntersection(cfg)).withAnnotations(Seq(WriteVcdAnnotation)) { dut =>

      // 1. 准备 10 组不同的测试数据
      val testCases = for (i <- 0 until 10) yield {
        val zOrig = -5.0f - i  // 每一组射线起始点越来越远
        val rOrig = (0.0f, 0.0f, zOrig)
        val rDir  = (0.0f, 0.0f, 1.0f)
        val bMin  = (-1.0f, -1.0f, -1.0f)
        val bMax  = (1.0f, 1.0f, 1.0f)
        val expected = intersectAABBSw(rOrig, rDir, bMin, bMax)
        (rOrig, rDir, bMin, bMax, expected)
      }

      val latency = 4 + cfg.faddLatency + cfg.fdivLatency + cfg.fmulLatency
      val results = new ListBuffer[(Boolean, Float, Float)]()

      println(s"Starting pipeline test with latency: $latency")

      // 2. 输入阶段 (Input Phase): 连续塞入 10 拍数据
      for (i <- 0 until 10) {
        val (orig, dir, min, max, _) = testCases(i)
        dut.io.ray.origin.x.poke(f2u(orig._1)); dut.io.ray.origin.y.poke(f2u(orig._2)); dut.io.ray.origin.z.poke(f2u(orig._3))
        dut.io.ray.dir.x.poke(f2u(dir._1));     dut.io.ray.dir.y.poke(f2u(dir._2));     dut.io.ray.dir.z.poke(f2u(dir._3))
        dut.io.aabb.min.x.poke(f2u(min._1));    dut.io.aabb.min.y.poke(f2u(min._2));    dut.io.aabb.min.z.poke(f2u(min._3))
        dut.io.aabb.max.x.poke(f2u(max._1));    dut.io.aabb.max.y.poke(f2u(max._2));    dut.io.aabb.max.z.poke(f2u(max._3))
        dut.io.in_valid.poke(true.B)

        // 每塞一笔，检查一下输出（前 latency 拍应该是无效的）
        if (dut.io.out_valid.peek().litToBoolean) {
          results += ((dut.io.hit.peek().litToBoolean, u2f(dut.io.tNear), u2f(dut.io.tFar)))
        }
        dut.clock.step(1)
      }

      // 3. 排水阶段 (Drain Phase): 停止输入，等待剩余结果吐完
      dut.io.in_valid.poke(false.B)
      for (_ <- 0 until latency) {
        if (dut.io.out_valid.peek().litToBoolean) {
          results += ((dut.io.hit.peek().litToBoolean, u2f(dut.io.tNear), u2f(dut.io.tFar)))
        }
        dut.clock.step(1)
      }

      // 4. 最终验证
      println(s"Received ${results.size} valid outputs.")
      assert(results.size == 10, "Did not receive all 10 results!")

      for (i <- 0 until 10) {
        val (expHit, expNear, expFar) = testCases(i)._5
        val (hwHit, hwNear, hwFar) = results(i)

        println(f"Case $i -> HW: ($hwHit, $hwNear%.4f), SW: ($expHit, $expNear%.4f)")
        assert(hwHit == expHit, s"Hit mismatch at case $i")
        assert(math.abs(hwNear - expNear) < 1e-4, s"tNear mismatch at case $i")
      }
      println("Pipeline test PASSED!")
    }
  }
}