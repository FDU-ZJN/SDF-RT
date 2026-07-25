package SDF

import chisel3._
import chisel3.util._
import raytrace_utils._

/** Queue-free SDF context scheduler.
  *
  * Step-ready and terminal contexts remain resident in the frontend context
  * table.  This unit tracks only ownership/state bits and selects IDs with
  * round-robin arbitration; no ray or context payload is buffered here.
  */
class SdfSchedulerUnit(
  cfg: FloatConfig,
  addrWidth: Int
) extends Module {
  private val contextCount = GlobalConfig.frontendContextCount
  private val idBits = GlobalConfig.frontendContextBits

  val io = IO(new Bundle {
    val issue_in = Flipped(Decoupled(UInt(idBits.W)))
    val context_id = Output(UInt(idBits.W))
    val immutable = Input(new ImmutableRayRecord(cfg, addrWidth))
    val sdf_state = Input(new FrontendSdfState(cfg))
    val state_write = Valid(new FrontendStateWrite(cfg))
    val final_context_id = Output(UInt(idBits.W))
    val final_state = Input(new FrontendSdfState(cfg))

    val pe_in = Decoupled(new SdfStepReq(cfg))
    val pe_out = Flipped(Decoupled(new SdfStepResp(cfg)))
    val out = Decoupled(new FrontendSdfResult(cfg))
  })

  val stepReady = RegInit(VecInit(Seq.fill(contextCount)(false.B)))
  val stepInFlight = RegInit(VecInit(Seq.fill(contextCount)(false.B)))
  val finalPending = RegInit(VecInit(Seq.fill(contextCount)(false.B)))
  val finalHit = Reg(Vec(contextCount, Bool()))

  val issueId = io.issue_in.bits
  val issueSlotIdle = !stepReady(issueId) && !stepInFlight(issueId) && !finalPending(issueId)
  io.issue_in.ready := issueSlotIdle
  val issueFire = io.issue_in.fire

  // A newly initialized context may bypass the state bitmap and enter the PE
  // on the same cycle.  FrontendContextManager supplies the matching state
  // table write/read bypass for entryT.
  val issueCandidates = Wire(Vec(contextCount, Bool()))
  for (ctx <- 0 until contextCount) {
    issueCandidates(ctx) := stepReady(ctx) || (issueFire && issueId === ctx.U)
  }

  val issueRr = RegInit(0.U(idBits.W))
  val issueRotated = Wire(Vec(contextCount, Bool()))
  for (offset <- 0 until contextCount) {
    val ctx = (issueRr + offset.U)(idBits - 1, 0)
    issueRotated(offset) := issueCandidates(ctx)
  }
  val issuePickOH = PriorityEncoderOH(issueRotated)
  val issuePickId = (OHToUInt(issuePickOH) + issueRr)(idBits - 1, 0)
  val issueValid = issueRotated.asUInt.orR
  io.context_id := issuePickId

  io.pe_in.valid := issueValid
  io.pe_in.bits.contextId := issuePickId
  io.pe_in.bits.origin := io.immutable.origin
  io.pe_in.bits.dir := io.immutable.dir
  io.pe_in.bits.state := io.sdf_state

  when(issueFire) {
    assert(issueSlotIdle, "SDF accepted a context that was already owned")
    stepReady(issueId) := true.B
  }
  when(io.pe_in.fire) {
    assert(issueCandidates(issuePickId), "SDF issued a non-runnable context")
    stepReady(issuePickId) := false.B
    stepInFlight(issuePickId) := true.B
    issueRr := issuePickId + 1.U
  }

  io.pe_out.ready := true.B
  val responseFire = io.pe_out.fire
  val responseId = io.pe_out.bits.contextId
  io.state_write.valid := responseFire
  io.state_write.bits.contextId := responseId
  io.state_write.bits.state := io.pe_out.bits.state

  when(responseFire) {
    assert(stepInFlight(responseId), "SdfPE returned a context with no request in flight")
    stepInFlight(responseId) := false.B
    when(io.pe_out.bits.done) {
      finalPending(responseId) := true.B
      finalHit(responseId) := io.pe_out.bits.hit
    }.otherwise {
      stepReady(responseId) := true.B
    }
  }

  val finalRr = RegInit(0.U(idBits.W))
  val finalRotated = Wire(Vec(contextCount, Bool()))
  for (offset <- 0 until contextCount) {
    val ctx = (finalRr + offset.U)(idBits - 1, 0)
    finalRotated(offset) := finalPending(ctx)
  }
  val finalPickOH = PriorityEncoderOH(finalRotated)
  val finalPickId = (OHToUInt(finalPickOH) + finalRr)(idBits - 1, 0)
  io.final_context_id := finalPickId
  io.out.valid := finalRotated.asUInt.orR
  io.out.bits.contextId := finalPickId
  io.out.bits.hit := finalHit(finalPickId)
  io.out.bits.dist := io.final_state.dist

  when(io.out.fire) {
    assert(finalPending(finalPickId), "SDF completed a non-terminal context")
    finalPending(finalPickId) := false.B
    finalRr := finalPickId + 1.U
  }

  for (ctx <- 0 until contextCount) {
    assert(PopCount(Seq(stepReady(ctx), stepInFlight(ctx), finalPending(ctx))) <= 1.U,
      "SDF context appears in multiple scheduler states")
  }
}
