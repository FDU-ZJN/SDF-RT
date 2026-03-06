package sdf_rt
import raytrace_utils._
import chisel3._
import chisel3.util._

class SimTop {
  val c = TriPeConfig(cfg = FloatConfig.FP32, numPEs = 4, addrWidth = 32)
  val io = IO(new Bundle {
    val ray_in = Input(new Ray(c.cfg))
    val ray_valid = Input(Bool())
    val tri_batch_in = Input(new TriBatch(c.addrWidth))
    val tri_batch_valid = Input(Bool())
    val end_exec = Input(Bool())

    val out_rgb = Output(new Vec3(c.cfg))
    val out_valid = Output(Bool())
  })
  val traceStage = Module(new TraceStage())
  traceStage.io.ray_in := io.ray_in
  traceStage.io.ray_valid := io.ray_valid
  traceStage.io.tri_batch_in := io.tri_batch_in
  traceStage.io.tri_batch_valid := io.tri_batch_valid
  traceStage.io.end_exec := io.end_exec
  val renderStage = Module(new RenderPE(c.cfg))
  renderStage.io.hit_id := traceStage.io.out_id
  renderStage.io.hit_valid := traceStage.io.out_valid
  renderStage.io.in_hit := traceStage.io.out_best_hit
  io.out_rgb:= renderStage.io.out_rgb
  io.out_valid := renderStage.io.out_valid

}
object SimTopGen extends App {
  emitVerilog(new TraceStage(), Array("--target-dir", "build"))
}
