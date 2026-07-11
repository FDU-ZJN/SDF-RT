package SDF

import chisel3._
import chisel3.util._
import raytrace_utils._

class SdfSchedulerUnit(
  cfg: FloatConfig,
  addrWidth: Int,
  maxSteps: Int,
  numWorkers: Int = GlobalConfig.sdfStepNumWorkers
) extends Module {
  require(numWorkers == 2, s"SdfSchedulerUnit currently supports 2 workers, got $numWorkers")

  val io = IO(new Bundle {
    val issue_in = Vec(numWorkers, Flipped(Decoupled(new RayIssue(cfg, addrWidth))))

    val pe_in = Vec(numWorkers, Decoupled(new SdfRayReq(cfg, addrWidth)))
    val pe_out_miss = Vec(numWorkers, Flipped(Decoupled(new SdfRayResp(cfg, addrWidth))))
    val pe_out_hit = Vec(numWorkers, Flipped(Decoupled(new SdfRayResp(cfg, addrWidth))))

    val out = Vec(numWorkers, Decoupled(new SdfRayResp(cfg, addrWidth)))
  })

  val retryQs = Seq.fill(numWorkers)(Module(new Queue(new SdfRayReq(cfg, addrWidth), GlobalConfig.sdfRetryQueueDepth / numWorkers, pipe = true)))
  val finalQs = Seq.fill(numWorkers)(Module(new Queue(new SdfRayResp(cfg, addrWidth), GlobalConfig.sdfFinalQueueDepth, pipe = true)))
  val finalCreditWidth = log2Ceil(GlobalConfig.sdfFinalQueueDepth + 1)
  val finalCredits = RegInit(VecInit(Seq.fill(numWorkers)(GlobalConfig.sdfFinalQueueDepth.U(finalCreditWidth.W))))

  for (lane <- 0 until numWorkers) {
    val newReq = Wire(new SdfRayReq(cfg, addrWidth))
    newReq.ray := io.issue_in(lane).bits.ray
    newReq.meta := io.issue_in(lane).bits.meta
    newReq.iter := 0.U
    newReq.prevSdf := 0.U

    val useRetry = retryQs(lane).io.deq.valid
    val finalDeqFire = finalQs(lane).io.deq.fire
    val hasFinalCredit = finalCredits(lane) =/= 0.U || finalDeqFire
    io.pe_in(lane).valid := retryQs(lane).io.deq.valid || (io.issue_in(lane).valid && hasFinalCredit)
    io.pe_in(lane).bits := Mux(useRetry, retryQs(lane).io.deq.bits, newReq)
    retryQs(lane).io.deq.ready := io.pe_in(lane).ready && useRetry
    io.issue_in(lane).ready := io.pe_in(lane).ready && !useRetry && hasFinalCredit

    io.pe_out_miss(lane).ready := true.B
    io.pe_out_hit(lane).ready := true.B

    val missNeedRetry = io.pe_out_miss(lane).valid && (io.pe_out_miss(lane).bits.iter < maxSteps.U)
    val missTerminal = io.pe_out_miss(lane).valid && !missNeedRetry
    val hitTerminal = io.pe_out_hit(lane).valid
    val peMissFire = io.pe_out_miss(lane).fire
    val peHitFire = io.pe_out_hit(lane).fire

    when(missTerminal && hitTerminal) {
      assert(false.B, s"SdfStage lane $lane unexpectedly produced hit and terminal miss together")
    }

    retryQs(lane).io.enq.valid := missNeedRetry
    retryQs(lane).io.enq.bits.ray := io.pe_out_miss(lane).bits.ray
    retryQs(lane).io.enq.bits.meta := io.pe_out_miss(lane).bits.meta
    retryQs(lane).io.enq.bits.iter := io.pe_out_miss(lane).bits.iter
    retryQs(lane).io.enq.bits.prevSdf := io.pe_out_miss(lane).bits.prevSdf
    when(missNeedRetry) {
      assert(retryQs(lane).io.enq.ready, s"SdfStage lane $lane retryQ overflow")
    }

    finalQs(lane).io.enq.valid := hitTerminal || missTerminal
    finalQs(lane).io.enq.bits := Mux(hitTerminal, io.pe_out_hit(lane).bits, io.pe_out_miss(lane).bits)
    when(finalQs(lane).io.enq.valid) {
      assert(finalQs(lane).io.enq.ready, s"SdfStage lane $lane finalQ overflow")
    }

    io.out(lane) <> finalQs(lane).io.deq
    val retryEnqFire = retryQs(lane).io.enq.fire
    val finalEnqFire = finalQs(lane).io.enq.fire
    val issueFire = io.issue_in(lane).fire

    // Reserve one finalQ entry when a new ray enters the SDF transaction.
    // Retries reuse that reservation; it is returned only after the terminal
    // response leaves finalQ. This bounds all in-flight work by finalQ depth.
    when(issueFire && !finalDeqFire) {
      finalCredits(lane) := finalCredits(lane) - 1.U
    }.elsewhen(!issueFire && finalDeqFire) {
      finalCredits(lane) := finalCredits(lane) + 1.U
    }
    when(issueFire) {
      assert(hasFinalCredit, s"SdfStage lane $lane issued without finalQ credit")
    }

    // A PE result is a transaction boundary: a miss can either loop through
    // retryQ or terminate into finalQ, but never both.  Likewise every final
    // must have been produced by a PE result in this cycle.
    when(peMissFire) {
      assert(retryEnqFire ^ finalEnqFire,
        s"SDF lane $lane miss must choose exactly one of retryQ or finalQ")
    }
    when(peHitFire) {
      assert(finalEnqFire && !retryEnqFire,
        s"SDF lane $lane hit must enter finalQ only")
    }
    when(finalEnqFire) {
      assert(peMissFire || peHitFire,
        s"SDF lane $lane finalQ entry must originate at a PE output")
    }

  }
}
