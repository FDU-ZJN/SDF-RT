package SDF

import chisel3._
import chisel3.util._
import raytrace_utils._

class SdfSchedulerUnit(cfg: FloatConfig, addrWidth: Int, maxSteps: Int, numWorkers: Int = 1) extends Module {
  require(numWorkers == 1 || numWorkers == 2, s"SdfSchedulerUnit currently supports 1 or 2 workers, got $numWorkers")

  val io = IO(new Bundle {
    val issue_in = Vec(numWorkers, Flipped(Decoupled(new RayIssue(cfg, addrWidth))))

    val pe_in = Vec(numWorkers, Decoupled(new SdfRayReq(cfg, addrWidth)))
    val pe_out_miss = Vec(numWorkers, Flipped(Decoupled(new SdfRayResp(cfg, addrWidth))))
    val pe_out_hit = Vec(numWorkers, Flipped(Decoupled(new SdfRayResp(cfg, addrWidth))))

    val out_rgb = Output(new Vec3(cfg))
    val out_meta = Output(new RayMeta(addrWidth))
    val out_hit = Output(Bool())
    val out_ray = Output(new Ray(cfg))
    val out_valid = Output(Bool())
  })

  private val outputQueueDepth = 64
  private val outputPipelineSlack = 40

  val finalQ = Module(new Queue(new SdfRayResp(cfg, addrWidth), GlobalConfig.sdfFinalQueueDepth))
  val retryQs = Seq.fill(numWorkers)(Module(new Queue(new SdfRayReq(cfg, addrWidth), GlobalConfig.sdfRetryQueueDepth / numWorkers, pipe = true)))
  val terminalMissQs = Seq.fill(numWorkers)(Module(new Queue(new SdfRayResp(cfg, addrWidth), outputQueueDepth)))
  val hitQs = Seq.fill(numWorkers)(Module(new Queue(new SdfRayResp(cfg, addrWidth), outputQueueDepth)))

  val oneFp = "h3F800000".U(cfg.totalWidth.W)
  val zeroFp = 0.U(cfg.totalWidth.W)

  val newReqs = Wire(Vec(numWorkers, new SdfRayReq(cfg, addrWidth)))
  for (i <- 0 until numWorkers) {
    newReqs(i).ray := io.issue_in(i).bits.ray
    newReqs(i).meta := io.issue_in(i).bits.meta
    newReqs(i).iter := 0.U
    newReqs(i).prevSdf := 0.U
  }

  for (i <- 0 until numWorkers) {
    io.pe_in(i).valid := false.B
    io.pe_in(i).bits := 0.U.asTypeOf(new SdfRayReq(cfg, addrWidth))
  }

  val laneHasRetry = VecInit(retryQs.map(_.io.deq.valid))
  val outputBackpressure = VecInit((terminalMissQs ++ hitQs).map(_.io.count > (outputQueueDepth - outputPipelineSlack).U)).asUInt.orR
  val issueCanEnter = !outputBackpressure

  for (i <- 0 until numWorkers) {
    val useNew = issueCanEnter && !laneHasRetry(i) && io.issue_in(i).valid
    io.pe_in(i).valid := !outputBackpressure && (retryQs(i).io.deq.valid || useNew)
    io.pe_in(i).bits := Mux(retryQs(i).io.deq.valid, retryQs(i).io.deq.bits, newReqs(i))
    retryQs(i).io.deq.ready := !outputBackpressure && retryQs(i).io.deq.valid && io.pe_in(i).ready
    io.issue_in(i).ready := issueCanEnter && !laneHasRetry(i) && io.pe_in(i).ready
  }

  for (i <- 0 until numWorkers) {
    val missNeedRetry = io.pe_out_miss(i).valid && (io.pe_out_miss(i).bits.iter < maxSteps.U)
    val missTerminal = io.pe_out_miss(i).valid && !missNeedRetry

    retryQs(i).io.enq.valid := missNeedRetry
    retryQs(i).io.enq.bits.ray := io.pe_out_miss(i).bits.ray
    retryQs(i).io.enq.bits.meta := io.pe_out_miss(i).bits.meta
    retryQs(i).io.enq.bits.iter := io.pe_out_miss(i).bits.iter
    retryQs(i).io.enq.bits.prevSdf := io.pe_out_miss(i).bits.prevSdf
    when(missNeedRetry) {
      assert(retryQs(i).io.enq.ready, "SdfStage per-lane retryQ overflow")
    }

    terminalMissQs(i).io.enq.valid := missTerminal
    terminalMissQs(i).io.enq.bits := io.pe_out_miss(i).bits
    when(missTerminal) {
      assert(terminalMissQs(i).io.enq.ready, "SdfStage terminal miss queue overflow")
    }
    io.pe_out_miss(i).ready := true.B

    hitQs(i).io.enq.valid := io.pe_out_hit(i).valid
    hitQs(i).io.enq.bits := io.pe_out_hit(i).bits
    io.pe_out_hit(i).ready := true.B
    when(io.pe_out_hit(i).valid) {
      assert(hitQs(i).io.enq.ready, "SdfStage hit output queue overflow")
    }
  }

  val missArb = Module(new RRArbiter(new SdfRayResp(cfg, addrWidth), numWorkers))
  val hitArb = Module(new RRArbiter(new SdfRayResp(cfg, addrWidth), numWorkers))
  for (i <- 0 until numWorkers) {
    missArb.io.in(i) <> terminalMissQs(i).io.deq
    hitArb.io.in(i) <> hitQs(i).io.deq
  }

  val missTerminal = missArb.io.out.valid
  val hitTerminal = hitArb.io.out.valid

  // If a terminal miss and a hit arrive together, defer the miss by one extra scheduler cycle.
  val missTerminalConflict = missTerminal && hitTerminal
  val missTerminalDirect = missTerminal && !hitTerminal

  val selHit = hitTerminal
  val selMiss = !selHit && missTerminalDirect

  finalQ.io.enq.valid := selHit || selMiss
  finalQ.io.enq.bits := Mux(selHit, hitArb.io.out.bits, missArb.io.out.bits)
  when(finalQ.io.enq.valid) {
    assert(finalQ.io.enq.ready, "SdfStage finalQ overflow")
  }

  hitArb.io.out.ready := selHit && finalQ.io.enq.ready
  missArb.io.out.ready := selMiss && finalQ.io.enq.ready

  finalQ.io.deq.ready := true.B
  io.out_valid := finalQ.io.deq.valid
  io.out_meta := finalQ.io.deq.bits.meta
  io.out_hit := finalQ.io.deq.bits.hit
  io.out_ray := finalQ.io.deq.bits.ray
  io.out_rgb.x := Mux(finalQ.io.deq.bits.hit, oneFp, zeroFp)
  io.out_rgb.y := Mux(finalQ.io.deq.bits.hit, oneFp, zeroFp)
  io.out_rgb.z := Mux(finalQ.io.deq.bits.hit, oneFp, zeroFp)
}
