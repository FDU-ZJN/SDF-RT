package Trace

import chisel3._
import chisel3.util._
import raytrace_utils._

class TraceController(
  c: TriPeConfig = TriPeConfig(),
  maxCmds: Int = GlobalConfig.ddaMaxSteps
) extends Module {
  private val numWorkers = GlobalConfig.traceNumWorkers
  private val ctxCount = 2
  private val ctxW = 1
  private val slotCount = GlobalConfig.ddaTraceSlotCount
  private val traceSlotBits = GlobalConfig.ddaTraceSlotBits
  private val cmdIdxW = math.max(1, log2Ceil(maxCmds))
  private val workerCtxCount = numWorkers * ctxCount

  val io = IO(new Bundle {
    val job_in = Flipped(Decoupled(new DdaTraceJobDesc(c.cfg, c.addrWidth, maxCmds)))
    val cmd_write = Vec(2, Flipped(Valid(new DdaTraceCmdWrite(c.addrWidth, maxCmds))))
    val slot_release = Valid(UInt(traceSlotBits.W))
    val result_out = Decoupled(new TraceResult(c.cfg, c.addrWidth))
  })

  require(numWorkers > 0, "TraceController requires at least one worker")

  val workers = Seq.fill(numWorkers)(Module(new TriPE(c)))
  val cmdQueues = Seq.fill(slotCount)(Module(new Queue(new TriBatch(c.addrWidth), maxCmds, hasFlush = true)))
  val mem = Module(new TriangleMemMultiPort(c, numWorkers))

  val sEmpty :: sIssueRay :: sReadyFirst :: sWaitBatch :: sReadyCont :: sResultPending :: Nil = Enum(6)
  val ctxState = RegInit(VecInit(Seq.fill(numWorkers)(VecInit(Seq.fill(ctxCount)(sEmpty)))))
  val ctxJob = Reg(Vec(numWorkers, Vec(ctxCount, new DdaTraceJobDesc(c.cfg, c.addrWidth, maxCmds))))
  val ctxCmdIdx = RegInit(VecInit(Seq.fill(numWorkers)(VecInit(Seq.fill(ctxCount)(0.U(cmdIdxW.W))))))
  val ctxResult = Reg(Vec(numWorkers, Vec(ctxCount, new TraceResult(c.cfg, c.addrWidth))))
  val slotFlushPending = RegInit(VecInit(Seq.fill(slotCount)(false.B)))
  val jobQ = Module(new Queue(new DdaTraceJobDesc(c.cfg, c.addrWidth, maxCmds), 2, pipe = true))

  val missT = "h7F7FFFFF".U(c.cfg.totalWidth.W)
  val slotCmdCountW = log2Ceil(maxCmds + 1)
  val slotCmdCount = RegInit(VecInit(Seq.fill(slotCount)(0.U(slotCmdCountW.W))))
  val cmdQueueReady = Wire(Vec(slotCount, Bool()))
  val cmdQueueValid = Wire(Vec(slotCount, Bool()))
  val cmdQueueBits = Wire(Vec(slotCount, new TriBatch(c.addrWidth)))
  val cmdWriteReady = Wire(Vec(2, Bool()))
  val clearCtx = Wire(Vec(numWorkers, Vec(ctxCount, Bool())))

  for (s <- 0 until slotCount) {
    cmdQueueReady(s) := false.B
  }
  for (lane <- 0 until 2) {
    cmdWriteReady(lane) := false.B
  }
  for (w <- 0 until numWorkers) {
    for (ctx <- 0 until ctxCount) {
      clearCtx(w)(ctx) := false.B
    }
  }

  jobQ.io.enq <> io.job_in
  io.slot_release.valid := false.B
  io.slot_release.bits := 0.U

  for (s <- 0 until slotCount) {
    val laneHits = Wire(Vec(2, Bool()))
    for (lane <- 0 until 2) {
      laneHits(lane) := io.cmd_write(lane).valid && (io.cmd_write(lane).bits.slotIdx === s.U)
    }
    assert(PopCount(laneHits) <= 1.U, s"TraceController does not allow same-slot dual cmd writes for slot $s")
    cmdQueues(s).io.enq.valid := laneHits.asUInt.orR
    cmdQueues(s).io.enq.bits := Mux(laneHits(0), io.cmd_write(0).bits.tri, io.cmd_write(1).bits.tri)
    cmdQueues(s).io.flush.get := slotFlushPending(s)
    cmdQueues(s).io.deq.ready := cmdQueueReady(s)
    cmdQueueValid(s) := slotCmdCount(s) =/= 0.U
    cmdQueueBits(s) := cmdQueues(s).io.deq.bits
    for (lane <- 0 until 2) {
      when(laneHits(lane)) {
        cmdWriteReady(lane) := cmdQueues(s).io.enq.ready
      }
    }

    val cmdEnqFire = cmdQueues(s).io.enq.fire
    val cmdDeqFire = cmdQueues(s).io.deq.fire
    when(slotFlushPending(s)) {
      slotCmdCount(s) := 0.U
    }.elsewhen(cmdEnqFire || cmdDeqFire) {
      slotCmdCount(s) := slotCmdCount(s) + cmdEnqFire.asUInt - cmdDeqFire.asUInt
    }

    when(cmdQueueReady(s) && cmdQueueValid(s)) {
      assert(cmdQueues(s).io.deq.valid, "TraceController cmd slot count/queue valid mismatch")
    }
  }

  for (lane <- 0 until 2) {
    when(io.cmd_write(lane).valid) {
      assert(cmdWriteReady(lane), s"TraceController cmd queue overflow on lane $lane")
    }
  }

  for (w <- 0 until numWorkers) {
    workers(w).io.ray_in := 0.U.asTypeOf(new Ray(c.cfg))
    workers(w).io.ray_meta := 0.U.asTypeOf(new RayMeta(c.addrWidth))
    workers(w).io.ray_ctx := 0.U
    workers(w).io.ray_valid := false.B
    workers(w).io.tri_batch_in := 0.U.asTypeOf(new TriBatch(c.addrWidth))
    workers(w).io.tri_batch_ctx := 0.U
    workers(w).io.tri_batch_valid := false.B
    workers(w).io.end_exec := false.B
    workers(w).io.clear_ctx := clearCtx(w)

    mem.io.req(w) <> workers(w).io.mem_req
    mem.io.req_mask(w) <> workers(w).io.mem_req_mask
    workers(w).io.mem_resp <> mem.io.resp(w)
  }

  val allocFree = Wire(Vec(workerCtxCount, Bool()))
  for (idx <- 0 until workerCtxCount) {
    val w = idx / ctxCount
    val ctx = idx % ctxCount
    allocFree(idx) := ctxState(w)(ctx) === sEmpty && workers(w).io.start_ready(ctx)
  }
  val allocOH = PriorityEncoderOH(allocFree)
  jobQ.io.deq.ready := allocFree.asUInt.orR

  when(jobQ.io.deq.fire) {
    for (idx <- 0 until workerCtxCount) {
      val w = idx / ctxCount
      val ctx = idx % ctxCount
      when(allocOH(idx)) {
        ctxJob(w)(ctx) := jobQ.io.deq.bits
        ctxCmdIdx(w)(ctx) := 0.U
        when(jobQ.io.deq.bits.cmdCount === 0.U) {
          ctxResult(w)(ctx).meta := jobQ.io.deq.bits.meta
          ctxResult(w)(ctx).hit := false.B
          ctxResult(w)(ctx).hitId := 0.U
          ctxResult(w)(ctx).hitT := missT
          ctxState(w)(ctx) := sResultPending
        }.otherwise {
          ctxState(w)(ctx) := sIssueRay
        }
      }
    }
  }

  val issueRr = RegInit(VecInit(Seq.fill(numWorkers)(false.B)))
  val issueGrantValid = RegInit(VecInit(Seq.fill(numWorkers)(false.B)))
  val issueGrantCtx = Reg(Vec(numWorkers, UInt(ctxW.W)))
  val issueGrantSlot = Reg(Vec(numWorkers, UInt(traceSlotBits.W)))
  val issueGrantEnd = Reg(Vec(numWorkers, Bool()))
  val batchIssueValid = RegInit(VecInit(Seq.fill(numWorkers)(false.B)))
  val batchIssueBits = Reg(Vec(numWorkers, new TriBatch(c.addrWidth)))
  val batchIssueCtx = Reg(Vec(numWorkers, UInt(ctxW.W)))
  val batchIssueEnd = Reg(Vec(numWorkers, Bool()))
  for (w <- 0 until numWorkers) {
    val issueRay0 = ctxState(w)(0) === sIssueRay && workers(w).io.start_ready(0)
    val issueRay1 = ctxState(w)(1) === sIssueRay && workers(w).io.start_ready(1)
    val issueRayCtx = Mux(issueRay0, 0.U(ctxW.W), 1.U(ctxW.W))
    when(issueRay0 || issueRay1) {
      workers(w).io.ray_in := ctxJob(w)(issueRayCtx).ray
      workers(w).io.ray_meta := ctxJob(w)(issueRayCtx).meta
      workers(w).io.ray_ctx := issueRayCtx
      workers(w).io.ray_valid := true.B
      ctxState(w)(issueRayCtx) := sReadyFirst
    }

    val firstReady = Wire(Vec(ctxCount, Bool()))
    val contReady = Wire(Vec(ctxCount, Bool()))
    for (ctx <- 0 until ctxCount) {
      val slot = ctxJob(w)(ctx).traceSlot
      firstReady(ctx) := ctxState(w)(ctx) === sReadyFirst &&
        workers(w).io.output_ready(ctx) &&
        cmdQueueValid(slot) &&
        !issueGrantValid(w) &&
        !batchIssueValid(w)
      contReady(ctx) := ctxState(w)(ctx) === sReadyCont &&
        workers(w).io.output_ready(ctx) &&
        cmdQueueValid(slot) &&
        !issueGrantValid(w) &&
        !batchIssueValid(w)
    }

    val useFirst = firstReady.asUInt.orR
    val issue0 = Mux(useFirst, firstReady(0) && (!issueRr(w) || !firstReady(1)), contReady(0) && (!issueRr(w) || !contReady(1)))
    val issue1 = Mux(useFirst, firstReady(1) && (issueRr(w) || !firstReady(0)), contReady(1) && (issueRr(w) || !contReady(0)))
    val planBatch = issue0 || issue1
    val planCtx = Mux(issue0, 0.U(ctxW.W), 1.U(ctxW.W))
    val planSlot = ctxJob(w)(planCtx).traceSlot
    val cmdDeqToIssue = issueGrantValid(w) &&
      !batchIssueValid(w) &&
      workers(w).io.output_ready(issueGrantCtx(w)) &&
      cmdQueueValid(issueGrantSlot(w))

    when(planBatch) {
      issueGrantValid(w) := true.B
      issueGrantCtx(w) := planCtx
      issueGrantSlot(w) := planSlot
      issueGrantEnd(w) := ctxCmdIdx(w)(planCtx) === (ctxJob(w)(planCtx).cmdCount - 1.U)
      ctxState(w)(planCtx) := sWaitBatch
      issueRr(w) := !planCtx(0)
    }

    when(issueGrantValid(w)) {
      cmdQueueReady(issueGrantSlot(w)) := !batchIssueValid(w) &&
        workers(w).io.output_ready(issueGrantCtx(w))
      when(cmdDeqToIssue) {
        batchIssueValid(w) := true.B
        batchIssueBits(w) := cmdQueueBits(issueGrantSlot(w))
        batchIssueCtx(w) := issueGrantCtx(w)
        batchIssueEnd(w) := issueGrantEnd(w)
        issueGrantValid(w) := false.B
      }
    }

    when(batchIssueValid(w)) {
      workers(w).io.tri_batch_in := batchIssueBits(w)
      workers(w).io.tri_batch_ctx := batchIssueCtx(w)
      workers(w).io.tri_batch_valid := workers(w).io.output_ready(batchIssueCtx(w))
      workers(w).io.end_exec := batchIssueEnd(w)
      when(workers(w).io.output_ready(batchIssueCtx(w))) {
        batchIssueValid(w) := false.B
        when(batchIssueEnd(w)) {
          ctxState(w)(batchIssueCtx(w)) := sWaitBatch
        }.otherwise {
          ctxCmdIdx(w)(batchIssueCtx(w)) := ctxCmdIdx(w)(batchIssueCtx(w)) + 1.U
          ctxState(w)(batchIssueCtx(w)) := sReadyCont
        }
      }
    }

    when(workers(w).io.out_done) {
      val outCtx = workers(w).io.out_ctx
      when(issueGrantValid(w) && issueGrantCtx(w) === outCtx) {
        issueGrantValid(w) := false.B
      }
      when(batchIssueValid(w) && batchIssueCtx(w) === outCtx) {
        batchIssueValid(w) := false.B
      }
      when(workers(w).io.out_best_hit) {
        ctxResult(w)(outCtx).meta := workers(w).io.out_meta
        ctxResult(w)(outCtx).hit := true.B
        ctxResult(w)(outCtx).hitId := workers(w).io.hit_id
        ctxResult(w)(outCtx).hitT := workers(w).io.t_best
        ctxState(w)(outCtx) := sResultPending
        slotFlushPending(ctxJob(w)(outCtx).traceSlot) := true.B
        clearCtx(w)(outCtx) := true.B
      }.otherwise {
        ctxResult(w)(outCtx).meta := ctxJob(w)(outCtx).meta
        ctxResult(w)(outCtx).hit := false.B
        ctxResult(w)(outCtx).hitId := 0.U
        ctxResult(w)(outCtx).hitT := missT
        ctxState(w)(outCtx) := sResultPending
        clearCtx(w)(outCtx) := true.B
      }
    }
  }

  for (s <- 0 until slotCount) {
    when(slotFlushPending(s)) {
      slotFlushPending(s) := false.B
    }
  }

  val resultArb = Module(new RRArbiter(new TraceResult(c.cfg, c.addrWidth), workerCtxCount))
  for (a <- 0 until workerCtxCount) {
    val aw = a / ctxCount
    val ac = a % ctxCount
    val aLive = ctxState(aw)(ac) =/= sEmpty
    for (b <- (a + 1) until workerCtxCount) {
      val bw = b / ctxCount
      val bc = b % ctxCount
      val bLive = ctxState(bw)(bc) =/= sEmpty
      when(aLive && bLive) {
        assert(
          ctxJob(aw)(ac).traceSlot =/= ctxJob(bw)(bc).traceSlot,
          "TraceController duplicate live traceSlot across contexts"
        )
      }
    }
  }

  for (idx <- 0 until workerCtxCount) {
    val w = idx / ctxCount
    val ctx = idx % ctxCount
    resultArb.io.in(idx).valid := ctxState(w)(ctx) === sResultPending
    resultArb.io.in(idx).bits := ctxResult(w)(ctx)
    when(resultArb.io.in(idx).fire) {
      ctxState(w)(ctx) := sEmpty
      io.slot_release.valid := true.B
      io.slot_release.bits := ctxJob(w)(ctx).traceSlot
    }
  }

  io.result_out <> resultArb.io.out

  for (w <- 0 until numWorkers) {
    when(workers(w).io.out_done) {
      val outCtx = workers(w).io.out_ctx
      assert(
        ctxState(w)(outCtx) === sWaitBatch || ctxState(w)(outCtx) === sReadyCont,
        "TraceController got TriPE out_done for a context that is not waiting for a batch"
      )
    }
  }
}
