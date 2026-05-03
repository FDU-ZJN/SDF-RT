package Trace

import chisel3._
import chisel3.util._
import raytrace_utils._

class TraceController(
  c: TriPeConfig = TriPeConfig(),
  maxCmds: Int = GlobalConfig.ddaMaxSteps
) extends Module {
  private val numWorkers = GlobalConfig.traceNumWorkers
  private val cmdCountW = log2Ceil(maxCmds + 1)
  private val cmdIdxW = math.max(1, log2Ceil(maxCmds))

  val io = IO(new Bundle {
    val job_in = Flipped(Decoupled(new DdaTraceJob(c.cfg, c.addrWidth, maxCmds)))
    val result_out = Decoupled(new TraceResult(c.cfg, c.addrWidth))
  })

  require(numWorkers > 0, "TraceController requires at least one worker")

  val workers = Seq.fill(numWorkers)(Module(new TriPE(c)))
  val mem = Module(new TriangleMemMultiPort(c, numWorkers))

  val sIdle :: sIssueRay :: sIssueBatch :: sWaitBatch :: Nil = Enum(4)
  val workerState = RegInit(VecInit(Seq.fill(numWorkers)(sIdle)))
  val workerJob = Reg(Vec(numWorkers, new DdaTraceJob(c.cfg, c.addrWidth, maxCmds)))
  val workerCmdIdx = RegInit(VecInit(Seq.fill(numWorkers)(0.U(cmdIdxW.W))))
  val workerFlushPending = RegInit(VecInit(Seq.fill(numWorkers)(false.B)))
  val workerResultPending = RegInit(VecInit(Seq.fill(numWorkers)(false.B)))
  val workerResult = Reg(Vec(numWorkers, new TraceResult(c.cfg, c.addrWidth)))

  val missT = "h7F7FFFFF".U(c.cfg.totalWidth.W)
  val workerFree = Wire(Vec(numWorkers, Bool()))
  for (i <- 0 until numWorkers) {
    workerFree(i) := (workerState(i) === sIdle) && !workerFlushPending(i) && !workerResultPending(i) && workers(i).io.start_ready
  }

  val hasFreeWorker = workerFree.asUInt.orR
  val allocOH = PriorityEncoderOH(workerFree)
  io.job_in.ready := hasFreeWorker

  for (i <- 0 until numWorkers) {
    workers(i).io.ray_in := workerJob(i).ray
    workers(i).io.ray_meta := workerJob(i).meta
    workers(i).io.ray_valid := false.B
    workers(i).io.tri_batch_in := workerJob(i).cmds(workerCmdIdx(i))
    workers(i).io.tri_batch_valid := false.B
    workers(i).io.end_exec := false.B
    workers(i).io.flush := workerFlushPending(i)

    mem.io.req(i) <> workers(i).io.mem_req
    mem.io.req_mask(i) <> workers(i).io.mem_req_mask
    workers(i).io.mem_resp <> mem.io.resp(i)
  }

  when(io.job_in.fire) {
    for (i <- 0 until numWorkers) {
      when(allocOH(i)) {
        workerJob(i) := io.job_in.bits
        workerCmdIdx(i) := 0.U
        when(io.job_in.bits.cmdCount === 0.U) {
          workerResult(i).meta := io.job_in.bits.meta
          workerResult(i).hit := false.B
          workerResult(i).hitId := 0.U
          workerResult(i).hitT := missT
          workerResultPending(i) := true.B
          workerState(i) := sIdle
        }.otherwise {
          workerState(i) := sIssueRay
        }
      }
    }
  }

  for (i <- 0 until numWorkers) {
    when(workerFlushPending(i)) {
      workerFlushPending(i) := false.B
    }

    switch(workerState(i)) {
      is(sIdle) {
      }

      is(sIssueRay) {
        workers(i).io.ray_valid := true.B
        when(workers(i).io.start_ready) {
          workerState(i) := sIssueBatch
        }
      }

      is(sIssueBatch) {
        workers(i).io.tri_batch_valid := true.B
        workers(i).io.end_exec := workerCmdIdx(i) === (workerJob(i).cmdCount - 1.U)
        when(workers(i).io.output_ready) {
          workerState(i) := sWaitBatch
        }
      }

      is(sWaitBatch) {
        when(workers(i).io.out_done) {
          when(workers(i).io.out_best_hit) {
            workerResult(i).meta := workers(i).io.out_meta
            workerResult(i).hit := true.B
            workerResult(i).hitId := workers(i).io.hit_id
            workerResult(i).hitT := workers(i).io.t_best
            workerResultPending(i) := true.B
            workerFlushPending(i) := true.B
            workerState(i) := sIdle
          }.elsewhen(workerCmdIdx(i) === (workerJob(i).cmdCount - 1.U)) {
            workerResult(i).meta := workerJob(i).meta
            workerResult(i).hit := false.B
            workerResult(i).hitId := 0.U
            workerResult(i).hitT := missT
            workerResultPending(i) := true.B
            workerState(i) := sIdle
          }.otherwise {
            workerCmdIdx(i) := workerCmdIdx(i) + 1.U
            workerState(i) := sIssueBatch
          }
        }
      }
    }
  }

  val resultArb = Module(new RRArbiter(new TraceResult(c.cfg, c.addrWidth), numWorkers))
  for (i <- 0 until numWorkers) {
    resultArb.io.in(i).valid := workerResultPending(i)
    resultArb.io.in(i).bits := workerResult(i)
    when(resultArb.io.in(i).fire) {
      workerResultPending(i) := false.B
    }
  }

  io.result_out <> resultArb.io.out
}
