import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec
import Trace.TriPE
import raytrace_utils._

class TriPeBackpressureTest extends AnyFlatSpec with ChiselScalatestTester {
  behavior of "TriPE batch accounting"

  it should "not complete while the final triangle request is queued behind memory backpressure" in {
    val cfg = TriPeConfig(cfg = FloatConfig.FP32)
    test(new TriPE(cfg)) { dut =>
      dut.io.ray_valid.poke(false.B)
      dut.io.ray_ctx.poke(0.U)
      dut.io.tri_batch_valid.poke(false.B)
      dut.io.tri_batch_ctx.poke(0.U)
      dut.io.mem_req.ready.poke(false.B)
      dut.io.mem_resp.valid.poke(false.B)
      dut.io.mem_resp.bits.tag.poke(0.U)
      for (lane <- 0 until cfg.cacheLineTriangles) {
        dut.io.mem_resp.bits.block.mask(lane).poke(false.B)
        dut.io.mem_resp.bits.block.tris(lane).v0.x.poke(0.U)
        dut.io.mem_resp.bits.block.tris(lane).v0.y.poke(0.U)
        dut.io.mem_resp.bits.block.tris(lane).v0.z.poke(0.U)
        dut.io.mem_resp.bits.block.tris(lane).e1.x.poke(0.U)
        dut.io.mem_resp.bits.block.tris(lane).e1.y.poke(0.U)
        dut.io.mem_resp.bits.block.tris(lane).e1.z.poke(0.U)
        dut.io.mem_resp.bits.block.tris(lane).e2.x.poke(0.U)
        dut.io.mem_resp.bits.block.tris(lane).e2.y.poke(0.U)
        dut.io.mem_resp.bits.block.tris(lane).e2.z.poke(0.U)
        dut.io.mem_resp.bits.block.tris(lane).id.poke(0.U)
      }
      for (ctx <- 0 until 2) {
        dut.io.dda_done(ctx).poke(false.B)
        dut.io.clear_ctx(ctx).poke(false.B)
      }

      dut.io.start_ready(0).expect(true.B)
      dut.io.ray_in.origin.x.poke(0.U)
      dut.io.ray_in.origin.y.poke(0.U)
      dut.io.ray_in.origin.z.poke(0.U)
      dut.io.ray_in.dir.x.poke(java.lang.Float.floatToRawIntBits(1.0f).U)
      dut.io.ray_in.dir.y.poke(0.U)
      dut.io.ray_in.dir.z.poke(0.U)
      dut.io.ray_in.dist.poke(0.U)
      dut.io.ray_meta.rayId.poke(9.U)
      dut.io.ray_valid.poke(true.B)
      dut.clock.step()
      dut.io.ray_valid.poke(false.B)

      dut.io.output_ready(0).expect(true.B)
      dut.io.tri_batch_in.tri.base_addr.poke(0.U)
      dut.io.tri_batch_in.tri.count.poke(cfg.cacheLineTriangles.U)
      dut.io.tri_batch_in.cellExitT.poke(java.lang.Float.floatToRawIntBits(1.0f).U)
      dut.io.tri_batch_valid.poke(true.B)
      dut.clock.step()
      dut.io.tri_batch_valid.poke(false.B)
      dut.io.dda_done(0).poke(true.B)

      for (_ <- 0 until 20) {
        dut.io.out_done.expect(false.B)
        dut.clock.step()
      }
      dut.io.mem_req.valid.expect(true.B)
    }
  }
}
