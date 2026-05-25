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
  val finalQs = Seq.fill(numWorkers)(Module(new Queue(new SdfRayResp(cfg, addrWidth), GlobalConfig.sdfFinalQueueDepth)))

  for (lane <- 0 until numWorkers) {
    val newReq = Wire(new SdfRayReq(cfg, addrWidth))
    newReq.ray := io.issue_in(lane).bits.ray
    newReq.meta := io.issue_in(lane).bits.meta
    newReq.iter := 0.U
    newReq.prevSdf := 0.U

    val useRetry = retryQs(lane).io.deq.valid
    io.pe_in(lane).valid := retryQs(lane).io.deq.valid || io.issue_in(lane).valid
    io.pe_in(lane).bits := Mux(useRetry, retryQs(lane).io.deq.bits, newReq)
    retryQs(lane).io.deq.ready := io.pe_in(lane).ready && useRetry
    io.issue_in(lane).ready := io.pe_in(lane).ready && !useRetry

    io.pe_out_miss(lane).ready := true.B
    io.pe_out_hit(lane).ready := true.B

    val missNeedRetry = io.pe_out_miss(lane).valid && (io.pe_out_miss(lane).bits.iter < maxSteps.U)
    val missTerminal = io.pe_out_miss(lane).valid && !missNeedRetry
    val hitTerminal = io.pe_out_hit(lane).valid

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
  }
}
