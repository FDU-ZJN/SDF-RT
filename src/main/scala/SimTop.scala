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
  val commitQueue = Module(new CommitQueue(c.cfg))

  val rayMeta = Wire(new RayMeta(c.addrWidth))
  rayMeta.slotId := commitQueue.io.allocSlot
  rayMeta.seqId := seqCounter
  rayMeta.pixelX := 0.U
  rayMeta.pixelY := 0.U

  // BVHStage: ray input, root node, hit update
  val bvhStage = Module(new BVHStage(bvhC))
  val traceStage = Module(new TraceStage())

  val inputReady = bvhStage.io.start_in.ready && commitQueue.io.alloc.ready
  val inputFire = io.ray_valid && inputReady

  commitQueue.io.alloc.valid := inputFire
  commitQueue.io.alloc.bits := seqCounter

  bvhStage.io.start_in.valid := inputFire
  bvhStage.io.start_in.bits.ray := io.ray_in
  bvhStage.io.start_in.bits.meta := rayMeta
  bvhStage.io.start_in.bits.rootNode := 0.U // root node index, can be parameterized
  bvhStage.io.hit_update := traceStage.io.hit_update

  traceStage.io.issue_in.valid := bvhStage.io.first_leaf_pulse
  traceStage.io.issue_in.bits.ray := bvhStage.io.ray_passthrough
  traceStage.io.issue_in.bits.meta := bvhStage.io.done_meta
  traceStage.io.end_exec := bvhStage.io.done

  bvhStage.io.leaf_out <> traceStage.io.tri_batch_in

  io.out_ready := inputReady

  // RenderStage: connect hit info from TraceStage
  val renderStage = Module(new RenderStage(c.cfg))
  renderStage.io.in <> traceStage.io.result_out

  commitQueue.io.writeback <> renderStage.io.out

  val missQueue = Module(new Queue(new RenderResult(c.cfg, c.addrWidth), entries = 8))
  val missResult = Wire(new RenderResult(c.cfg, c.addrWidth))
  missResult.meta  := bvhStage.io.done_meta
  missResult.hit   := false.B
  missResult.hitId := 0.U
  missResult.rgb.x := 0.U
  missResult.rgb.y := 0.U
  missResult.rgb.z := 0.U
  missQueue.io.enq.valid := bvhStage.io.no_leaf_done
  missQueue.io.enq.bits  := missResult
  assert(!(bvhStage.io.no_leaf_done && !missQueue.io.enq.ready), "Miss Queue Overflow!")
  commitQueue.io.writeback2 <> missQueue.io.deq

  when(inputFire) {
    seqCounter := seqCounter + 1.U
  }

  commitQueue.io.out.ready := true.B
  io.out_rgb := commitQueue.io.out.bits.rgb
  io.out_valid := commitQueue.io.out.valid
  io.out_id := commitQueue.io.out.bits.hitId
}

object SimTopGen extends App {
  emitVerilog(new SimTop(), Array("--target-dir", "build"))
}
