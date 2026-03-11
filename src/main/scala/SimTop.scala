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

  val seqCounter = RegInit(0.U(c.addrWidth.W))
  val commitQueue = Module(new CommitQueue(c.cfg, depth = 8))

  val rayMeta = Wire(new RayMeta(c.addrWidth))
  rayMeta.slotId := commitQueue.io.allocSlot
  rayMeta.seqId := seqCounter
  rayMeta.pixelX := 0.U
  rayMeta.pixelY := 0.U

  // BVHStage: ray input, root node, hit update
  val bvhStage = Module(new BVHStage(bvhC))
  val traceStage = Module(new TraceStage())

  val inputReady = bvhStage.io.start_ready && traceStage.io.start_ready && commitQueue.io.allocReady
  val inputFire = io.ray_valid && inputReady

  commitQueue.io.allocValid := inputFire

  bvhStage.io.start := inputFire
  bvhStage.io.rootNode := 0.U // root node index, can be parameterized
  bvhStage.io.ray_in := io.ray_in
  bvhStage.io.hit_update_valid := traceStage.io.out_best_hit
  bvhStage.io.hit_update_t := traceStage.io.best_t

  // TraceStage: connect leaf_out from BVHStage

  traceStage.io.ray_in := io.ray_in
  traceStage.io.ray_meta := rayMeta
  traceStage.io.ray_valid := inputFire
  traceStage.io.tri_batch_in := bvhStage.io.leaf_out.bits
  traceStage.io.tri_batch_valid := bvhStage.io.leaf_out.valid
  traceStage.io.end_exec := bvhStage.io.done

  bvhStage.io.leaf_out.ready := traceStage.io.output_ready

  io.out_ready := inputReady

  // RenderStage: connect hit info from TraceStage
  val renderStage = Module(new RenderStage(c.cfg))
  renderStage.io.in_result := traceStage.io.out_result
  renderStage.io.in_valid := traceStage.io.out_valid

  commitQueue.io.writebackValid := renderStage.io.out_valid
  commitQueue.io.writeback := renderStage.io.out_result

  when(inputFire) {
    seqCounter := seqCounter + 1.U
  }

  io.out_rgb := commitQueue.io.outResult.rgb
  io.out_valid := commitQueue.io.outValid
  io.out_id := commitQueue.io.outResult.hitId
}

object SimTopGen extends App {
  emitVerilog(new SimTop(), Array("--target-dir", "build"))
}
