package DDA.Trace

import chisel3._
import chisel3.util._
import raytrace_utils._


class TraceStage(c: TriPeConfig = TriPeConfig()) extends Module {
  val io = IO(new Bundle {
    val issue_in = Flipped(Decoupled(new RayIssue(c.cfg, c.addrWidth)))
    val tri_batch_in = Flipped(Decoupled(new TriBatch(c.addrWidth)))
    val end_exec = Input(Bool())

    val hit_update = Valid(new TraceHitUpdate(c.cfg, c.addrWidth))
    val result_out = Decoupled(new TraceResult(c.cfg, c.addrWidth))
  })

  val pe  = Module(new TriPE(c))
  val mem = Module(new TriangleMemWrapper(c))

  // --- 直接互联 (Direct Interconnect) ---
  // 1. 内存接口对接
  mem.io.req  <> pe.io.mem_req
  pe.io.mem_resp <> mem.io.resp

  pe.io.ray_in := io.issue_in.bits.ray
  pe.io.ray_meta := io.issue_in.bits.meta
  pe.io.ray_valid := io.issue_in.fire
  // Stall ray issue when a previous result is still back-pressured.
  val resultPending = RegInit(false.B)
  io.issue_in.ready := pe.io.start_ready && !resultPending

  pe.io.tri_batch_in := io.tri_batch_in.bits
  pe.io.tri_batch_valid := io.tri_batch_in.valid
  io.tri_batch_in.ready := pe.io.output_ready
  pe.io.end_exec := io.end_exec

  io.hit_update.valid := pe.io.out_best_hit
  io.hit_update.bits.hit := pe.io.out_best_hit
  io.hit_update.bits.hitId := pe.io.hit_id
  io.hit_update.bits.hitT := pe.io.t_best
  val peResult = Wire(new TraceResult(c.cfg, c.addrWidth))
  peResult.meta := pe.io.out_meta
  peResult.hit := pe.io.out_best_hit
  peResult.hitId := pe.io.hit_id
  peResult.hitT := pe.io.t_best

  io.result_out.valid := pe.io.out_done
  io.result_out.bits := peResult


}
