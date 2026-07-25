package Backend

import chisel3._
import chisel3.util._
import raytrace_utils._

/** Owns all state and command buffering for one backend traversal context. */
class ContextSlot(
  cfg: FloatConfig = FloatConfig.FP32,
  addrWidth: Int = 32,
  traceCmdDepth: Int = GlobalConfig.ddaTraceCmdDepth
) extends Module {
  val io = IO(new Bundle {
    val reserveInit = Input(Bool())
    val initCommit = Input(Bool())
    val initContext = Input(new DdaContext(cfg, addrWidth))

    val stepIssue = Input(Bool())
    val stepReturn = Input(Bool())
    val stepResult = Input(new DdaStepResult(cfg, addrWidth))

    val claim = Input(Bool())
    val stop = Input(Bool())
    val release = Input(Bool())

    val cmdOut = Decoupled(new TraceCellCmd(cfg, addrWidth))

    val context = Output(new DdaContext(cfg, addrWidth))
    val freeForInit = Output(Bool())
    val stepEligible = Output(Bool())
    val claimEligible = Output(Bool())
    val active = Output(Bool())
    val claimed = Output(Bool())
    val ddaDone = Output(Bool())
    val watchdogPulse = Output(Bool())
  })

  val context = Reg(new DdaContext(cfg, addrWidth))
  val active = RegInit(false.B)
  val initReserved = RegInit(false.B)
  val claimed = RegInit(false.B)
  val ddaDone = RegInit(false.B)
  val traceStopped = RegInit(false.B)
  val resultAccepted = RegInit(false.B)
  val stepInFlight = RegInit(false.B)

  val cmdQueue = Module(new Queue(
    new TraceCellCmd(cfg, addrWidth),
    traceCmdDepth,
    hasFlush = true
  ))
  cmdQueue.io.enq.valid := false.B
  cmdQueue.io.enq.bits := 0.U.asTypeOf(new TraceCellCmd(cfg, addrWidth))
  cmdQueue.io.flush.get := io.stop || io.release || io.initCommit
  io.cmdOut <> cmdQueue.io.deq

  io.context := context
  io.freeForInit := !active && !initReserved
  io.stepEligible := active && !ddaDone && !traceStopped &&
    !resultAccepted && !stepInFlight && cmdQueue.io.enq.ready
  io.claimEligible := active && !claimed && !traceStopped &&
    (cmdQueue.io.deq.valid || ddaDone)
  io.active := active
  io.claimed := claimed
  io.ddaDone := ddaDone
  io.watchdogPulse := false.B

  when(io.reserveInit) {
    initReserved := true.B
  }

  when(io.initCommit) {
    assert(initReserved, "Backend init completed for an unreserved context")
    context := io.initContext
    active := true.B
    initReserved := false.B
    claimed := false.B
    ddaDone := false.B
    traceStopped := false.B
    resultAccepted := false.B
    stepInFlight := false.B
  }

  when(io.stepIssue) {
    assert(io.stepEligible, "Backend issued an ineligible context")
    stepInFlight := true.B
  }

  when(io.stepReturn) {
    assert(stepInFlight, "DDA step returned for a context with no request in flight")
    stepInFlight := false.B
    when(resultAccepted) {
      active := false.B
      claimed := false.B
      ddaDone := false.B
      traceStopped := false.B
      resultAccepted := false.B
    }.elsewhen(!traceStopped && !io.stop && !io.release) {
      context := io.stepResult.ctx
      when(io.stepResult.done) {
        ddaDone := true.B
        io.watchdogPulse := io.stepResult.watchdog
      }.elsewhen(io.stepResult.emitCmd) {
        assert(cmdQueue.io.enq.ready, "Backend trace command reservation overflow")
        cmdQueue.io.enq.valid := true.B
        cmdQueue.io.enq.bits.tri := io.stepResult.tri
        cmdQueue.io.enq.bits.cellExitT := io.stepResult.cellExitT
      }
    }
  }

  when(io.claim) {
    assert(io.claimEligible, "Trace claimed an ineligible backend context")
    claimed := true.B
  }

  when(io.stop) {
    assert(active && claimed, "Trace stopped an inactive or unclaimed context")
    traceStopped := true.B
  }

  when(io.release) {
    assert(active && claimed, "Trace released an inactive or unclaimed context")
    resultAccepted := true.B
    when(!stepInFlight || io.stepReturn) {
      active := false.B
      claimed := false.B
      ddaDone := false.B
      traceStopped := false.B
      resultAccepted := false.B
    }
  }
}
