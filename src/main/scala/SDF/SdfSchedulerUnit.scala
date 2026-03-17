package SDF

import chisel3._
import chisel3.util._
import raytrace_utils._

class SdfSchedulerUnit(cfg: FloatConfig, addrWidth: Int, maxSteps: Int) extends Module {
  val io = IO(new Bundle {
    val issue_in = Flipped(Decoupled(new RayIssue(cfg, addrWidth)))

    val pe_in = Decoupled(new SdfRayReq(cfg, addrWidth))
    val pe_out = Flipped(Decoupled(new SdfRayResp(cfg, addrWidth)))

    val out_rgb = Output(new Vec3(cfg))
    val out_meta = Output(new RayMeta(addrWidth))
    val out_valid = Output(Bool())
  })

  val workQ = Module(new Queue(new SdfRayReq(cfg, addrWidth), 16))
  val retryQ = Module(new Queue(new SdfRayReq(cfg, addrWidth), 16))
  val finalQ = Module(new Queue(new SdfRayResp(cfg, addrWidth), 8))

  val oneFp = "h3F800000".U(cfg.totalWidth.W)
  val zeroFp = 0.U(cfg.totalWidth.W)

  val newReq = Wire(new SdfRayReq(cfg, addrWidth))
  newReq.ray := io.issue_in.bits.ray
  newReq.meta := io.issue_in.bits.meta
  newReq.tNear := 0.U
  newReq.tFar := 0.U
  newReq.iter := 0.U

  val inArb = Module(new RRArbiter(new SdfRayReq(cfg, addrWidth), 2))
  inArb.io.in(0).valid := retryQ.io.deq.valid
  inArb.io.in(0).bits := retryQ.io.deq.bits
  retryQ.io.deq.ready := inArb.io.in(0).ready

  inArb.io.in(1).valid := io.issue_in.valid
  inArb.io.in(1).bits := newReq
  io.issue_in.ready := inArb.io.in(1).ready

  workQ.io.enq <> inArb.io.out
  io.pe_in <> workQ.io.deq

  io.pe_out.ready := true.B

  val needRetry = io.pe_out.valid && !io.pe_out.bits.hit && (io.pe_out.bits.iter < maxSteps.U)
  val terminalOut = io.pe_out.valid && !needRetry

  retryQ.io.enq.valid := needRetry
  retryQ.io.enq.bits.ray := io.pe_out.bits.ray
  retryQ.io.enq.bits.meta := io.pe_out.bits.meta
  retryQ.io.enq.bits.tNear := 0.U
  retryQ.io.enq.bits.tFar := 0.U
  retryQ.io.enq.bits.iter := io.pe_out.bits.iter
  when(needRetry) {
    assert(retryQ.io.enq.ready, "SdfStage retryQ overflow")
  }

  finalQ.io.enq.valid := terminalOut
  finalQ.io.enq.bits := io.pe_out.bits
  when(terminalOut) {
    assert(finalQ.io.enq.ready, "SdfStage finalQ overflow")
  }

  finalQ.io.deq.ready := true.B
  io.out_valid := finalQ.io.deq.valid
  io.out_meta := finalQ.io.deq.bits.meta
  io.out_rgb.x := Mux(finalQ.io.deq.bits.hit, oneFp, zeroFp)
  io.out_rgb.y := Mux(finalQ.io.deq.bits.hit, oneFp, zeroFp)
  io.out_rgb.z := Mux(finalQ.io.deq.bits.hit, oneFp, zeroFp)
}
