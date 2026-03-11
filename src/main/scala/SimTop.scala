package sdf_rt
import raytrace_utils._
import chisel3._
import chisel3.util._

class SimTop extends Module {
  val c = TriPeConfig(cfg = FloatConfig.FP32, numPEs = 4, addrWidth = 32)
  val bvhC = BvhPeConfig(addrWidth = 32, stackDepth = 64, reqQueueDepth = 16, leafQueueDepth = 16, cfg = FloatConfig.FP32)
  val io = IO(new Bundle {
    val ray_in = Input(new Ray(c.cfg))
    val ray_valid = Input(Bool())
    val out_ready = Output(Bool())
    val out_rgb = Output(new Vec3(c.cfg))
    val out_valid = Output(Bool())
    val out_id = Output(UInt(c.addrWidth.W))
  })

  // BVHStage: ray input, root node, hit update
  val bvhStage = Module(new BVHStage(bvhC))
  val traceStage = Module(new TraceStage())
  bvhStage.io.start := io.ray_valid
  bvhStage.io.rootNode := 0.U // root node index, can be parameterized
  bvhStage.io.ray_in := io.ray_in
  bvhStage.io.hit_update_valid := traceStage.io.out_best_hit
  bvhStage.io.hit_update_t := traceStage.io.best_t

  // TraceStage: connect leaf_out from BVHStage

  traceStage.io.ray_in := io.ray_in
  traceStage.io.ray_valid := io.ray_valid
  traceStage.io.tri_batch_in := bvhStage.io.leaf_out.bits
  traceStage.io.tri_batch_valid := bvhStage.io.leaf_out.valid
  traceStage.io.end_exec := bvhStage.io.done

  bvhStage.io.leaf_out.ready := traceStage.io.output_ready

  io.out_ready := traceStage.io.output_ready

  // RenderStage: connect hit info from TraceStage
  val renderStage = Module(new RenderStage(c.cfg))
  renderStage.io.hit_id := traceStage.io.out_id
  renderStage.io.in_valid := traceStage.io.out_valid
  renderStage.io.in_hit := traceStage.io.out_best_hit

  io.out_rgb := renderStage.io.out_rgb
  io.out_valid := renderStage.io.out_valid
  io.out_id := renderStage.io.out_id
}

object SimTopGen extends App {
  emitVerilog(new SimTop(), Array("--target-dir", "build"))
}
