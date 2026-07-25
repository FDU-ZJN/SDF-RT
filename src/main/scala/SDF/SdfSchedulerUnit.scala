package SDF

import chisel3._
import chisel3.util._
import raytrace_utils._

class SdfSchedulerUnit(
  cfg: FloatConfig,
  addrWidth: Int,
  maxSteps: Int
) extends Module {
  val io = IO(new Bundle {
    val issue_in = Flipped(Decoupled(new RayIssue(cfg, addrWidth)))
    val pe_in = Decoupled(new SdfRayReq(cfg, addrWidth))
    val pe_out_miss = Flipped(Decoupled(new SdfRayResp(cfg, addrWidth)))
    val pe_out_hit = Flipped(Decoupled(new SdfRayResp(cfg, addrWidth)))
    val out = Decoupled(new SdfRayResp(cfg, addrWidth))
  })

  val retryQ = Module(new Queue(new SdfRayReq(cfg, addrWidth), GlobalConfig.sdfRetryQueueDepth, pipe = true))
  val finalQ = Module(new Queue(new SdfRayResp(cfg, addrWidth), GlobalConfig.sdfFinalQueueDepth, pipe = true))
  val finalCreditWidth = log2Ceil(GlobalConfig.sdfFinalQueueDepth + 1)
  val finalCredits = RegInit(GlobalConfig.sdfFinalQueueDepth.U(finalCreditWidth.W))

  val newReq = Wire(new SdfRayReq(cfg, addrWidth))
  newReq.ray := io.issue_in.bits.ray
  newReq.meta := io.issue_in.bits.meta
  newReq.iter := 0.U
  newReq.prevSdf := 0.U

  val useRetry = retryQ.io.deq.valid
  val finalDeqFire = finalQ.io.deq.fire
  val hasFinalCredit = finalCredits =/= 0.U || finalDeqFire
  io.pe_in.valid := retryQ.io.deq.valid || (io.issue_in.valid && hasFinalCredit)
  io.pe_in.bits := Mux(useRetry, retryQ.io.deq.bits, newReq)
  retryQ.io.deq.ready := io.pe_in.ready && useRetry
  io.issue_in.ready := io.pe_in.ready && !useRetry && hasFinalCredit

  io.pe_out_miss.ready := true.B
  io.pe_out_hit.ready := true.B

  val missNeedRetry = io.pe_out_miss.valid && (io.pe_out_miss.bits.iter < maxSteps.U)
  val missTerminal = io.pe_out_miss.valid && !missNeedRetry
  val hitTerminal = io.pe_out_hit.valid
  val peMissFire = io.pe_out_miss.fire
  val peHitFire = io.pe_out_hit.fire

  when(missTerminal && hitTerminal) {
    assert(false.B, "SdfStage unexpectedly produced hit and terminal miss together")
  }

  retryQ.io.enq.valid := missNeedRetry
  retryQ.io.enq.bits.ray := io.pe_out_miss.bits.ray
  retryQ.io.enq.bits.meta := io.pe_out_miss.bits.meta
  retryQ.io.enq.bits.iter := io.pe_out_miss.bits.iter
  retryQ.io.enq.bits.prevSdf := io.pe_out_miss.bits.prevSdf
  when(missNeedRetry) {
    assert(retryQ.io.enq.ready, "SdfStage retryQ overflow")
  }

  finalQ.io.enq.valid := hitTerminal || missTerminal
  finalQ.io.enq.bits := Mux(hitTerminal, io.pe_out_hit.bits, io.pe_out_miss.bits)
  when(finalQ.io.enq.valid) {
    assert(finalQ.io.enq.ready, "SdfStage finalQ overflow")
  }

  io.out <> finalQ.io.deq
  val retryEnqFire = retryQ.io.enq.fire
  val finalEnqFire = finalQ.io.enq.fire
  val issueFire = io.issue_in.fire

  when(issueFire && !finalDeqFire) {
    finalCredits := finalCredits - 1.U
  }.elsewhen(!issueFire && finalDeqFire) {
    finalCredits := finalCredits + 1.U
  }
  when(issueFire) {
    assert(hasFinalCredit, "SdfStage issued without finalQ credit")
  }

  when(peMissFire) {
    assert(retryEnqFire ^ finalEnqFire,
      "SDF miss must choose exactly one of retryQ or finalQ")
  }
  when(peHitFire) {
    assert(finalEnqFire && !retryEnqFire,
      "SDF hit must enter finalQ only")
  }
  when(finalEnqFire) {
    assert(peMissFire || peHitFire,
      "SDF finalQ entry must originate at a PE output")
  }
}
