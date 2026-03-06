package sdf_rt
import raytrace_utils._
import raytrace_utils.fudian._
import chisel3._
import chisel3.util._


class TraceStage() extends Module {
  val c = TriPeConfig(cfg = FloatConfig.FP32, numPEs = 4, addrWidth = 32)
  val io = IO(new Bundle {
    // 暴露给 Verilator 的顶层接口
    val ray_in          = Input(new Ray(c.cfg))
    val ray_valid       = Input(Bool())
    val tri_batch_in    = Input(new TriBatch(c.addrWidth))
    val tri_batch_valid = Input(Bool())
    val end_exec        = Input(Bool())

    val out_best_hit    = Output(Bool())
    val out_id           = Output(UInt(c.addrWidth.W))
    val out_valid        = Output(Bool())
    val output_ready    = Output(Bool())
  })

  val pe  = Module(new TriPE(c))
  val mem = Module(new TriangleMemWrapper(c))

  // --- 直接互联 (Direct Interconnect) ---
  // 1. 内存接口对接
  mem.io.req  <> pe.io.mem_req
  pe.io.mem_resp <> mem.io.resp

  // 2. 外部 IO 对接
  pe.io.ray_in          := io.ray_in
  pe.io.ray_valid       := io.ray_valid
  pe.io.tri_batch_in    := io.tri_batch_in
  pe.io.tri_batch_valid := io.tri_batch_valid
  pe.io.end_exec        := io.end_exec

  io.out_best_hit       := pe.io.out_best_hit
  io.out_id              := pe.io.hit_id
  io.out_valid           := pe.io.out_done
  io.output_ready       := pe.io.output_ready
}
