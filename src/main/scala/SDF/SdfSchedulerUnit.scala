package SDF

import chisel3._
import chisel3.util._
import raytrace_utils._

class SdfSchedulerUnit(cfg: FloatConfig, addrWidth: Int, maxSteps: Int) extends Module {
  val io = IO(new Bundle {
    val issue_in = Flipped(Decoupled(new RayIssue(cfg, addrWidth)))

    val pe_in = Decoupled(new SdfRayReq(cfg, addrWidth))
    val pe_out_miss = Flipped(Decoupled(new SdfRayResp(cfg, addrWidth)))
    val pe_out_hit = Flipped(Decoupled(new SdfRayResp(cfg, addrWidth)))

    val out = Decoupled(new SdfRayResp(cfg, addrWidth))
  })

  val retryQ = Module(new Queue(new SdfRayReq(cfg, addrWidth), GlobalConfig.sdfRetryQueueDepth))

  val newReq = Wire(new SdfRayReq(cfg, addrWidth))
  newReq.ray := io.issue_in.bits.ray
  newReq.meta := io.issue_in.bits.meta
  newReq.iter := 0.U
  newReq.prevSdf := 0.U

  val inArb = Module(new RRArbiter(new SdfRayReq(cfg, addrWidth), 2))
  inArb.io.in(0).valid := retryQ.io.deq.valid
  inArb.io.in(0).bits := retryQ.io.deq.bits
  retryQ.io.deq.ready := inArb.io.in(0).ready

  inArb.io.in(1).valid := io.issue_in.valid
  inArb.io.in(1).bits := newReq
  io.issue_in.ready := inArb.io.in(1).ready

  val arbOutQ = Module(new Queue(new SdfRayReq(cfg, addrWidth), 1, pipe = true))
  arbOutQ.io.enq <> inArb.io.out

  val arbReq = arbOutQ.io.deq.bits
  val arbIsDeferredTerminalMiss = arbOutQ.io.deq.valid && (arbReq.iter >= maxSteps.U)

  io.pe_in.valid := arbOutQ.io.deq.valid && !arbIsDeferredTerminalMiss
  io.pe_in.bits := arbReq

  io.pe_out_miss.ready := true.B
  io.pe_out_hit.ready := true.B

  val missNeedRetry = io.pe_out_miss.valid && (io.pe_out_miss.bits.iter < maxSteps.U)
  val missTerminal = io.pe_out_miss.valid && !missNeedRetry
  val hitTerminal = io.pe_out_hit.valid

  // If a terminal miss and a hit arrive together, defer the miss by one extra scheduler cycle.
  val missTerminalConflict = missTerminal && hitTerminal
  val missTerminalDirect = missTerminal && !hitTerminal

  val retryFromMiss = missNeedRetry
  val retryFromConflictMiss = missTerminalConflict
  val retryPush = retryFromMiss || retryFromConflictMiss

  retryQ.io.enq.valid := retryPush
  retryQ.io.enq.bits.ray := io.pe_out_miss.bits.ray
  retryQ.io.enq.bits.meta := io.pe_out_miss.bits.meta
  // Mark conflict-deferred terminal miss so it bypasses PE next time and commits directly.
  retryQ.io.enq.bits.iter := Mux(retryFromConflictMiss, maxSteps.U, io.pe_out_miss.bits.iter)
  retryQ.io.enq.bits.prevSdf := io.pe_out_miss.bits.prevSdf
  when(retryPush) {
    assert(retryQ.io.enq.ready, "SdfStage retryQ overflow")
  }

  val deferredResp = Wire(new SdfRayResp(cfg, addrWidth))
  deferredResp.ray := arbReq.ray
  deferredResp.meta := arbReq.meta
  deferredResp.hit := false.B
  deferredResp.iter := arbReq.iter
  deferredResp.prevSdf := arbReq.prevSdf

  val selHit = hitTerminal
  val selMiss = !selHit && missTerminalDirect
  val selDeferred = !selHit && !selMiss && arbIsDeferredTerminalMiss

  io.out.valid := selHit || selMiss || selDeferred
  io.out.bits := Mux(selHit, io.pe_out_hit.bits, Mux(selMiss, io.pe_out_miss.bits, deferredResp))
  when(io.out.valid) {
    assert(io.out.ready, "SdfStage output must not be backpressured")
  }

  arbOutQ.io.deq.ready := Mux(arbIsDeferredTerminalMiss, selDeferred, io.pe_in.ready)
}
