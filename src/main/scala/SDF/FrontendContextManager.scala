package SDF

import chisel3._
import chisel3.util._
import raytrace_utils._

/** Owns the immutable ray record and mutable SDF state for 32 frontend rays. */
class FrontendContextManager(
  cfg: FloatConfig = FloatConfig.FP32,
  addrWidth: Int = 32,
  readPorts: Int = 3,
  releasePorts: Int = 2,
  stateWritePorts: Int = 2
) extends Module {
  private val count = GlobalConfig.frontendContextCount
  private val idBits = GlobalConfig.frontendContextBits

  val io = IO(new Bundle {
    val ingress = Flipped(Decoupled(new RayIngress(cfg)))
    val init_issue = Decoupled(UInt(idBits.W))
    val init_complete = Input(Valid(new FrontendInitResp(cfg)))
    val sdf_issue = Decoupled(UInt(idBits.W))
    val read_id = Input(Vec(readPorts, UInt(idBits.W)))
    val immutable = Output(Vec(readPorts, new ImmutableRayRecord(cfg, addrWidth)))
    val sdf_state = Output(Vec(readPorts, new FrontendSdfState(cfg)))
    val state_write = Input(Vec(stateWritePorts, Valid(new FrontendStateWrite(cfg))))
    val release = Input(Vec(releasePorts, Valid(UInt(idBits.W))))
    val active_count = Output(UInt(log2Ceil(count + 1).W))
  })

  val active = RegInit(VecInit(Seq.fill(count)(false.B)))
  val sdfReady = RegInit(VecInit(Seq.fill(count)(false.B)))
  val immutableTable = Reg(Vec(count, new ImmutableRayRecord(cfg, addrWidth)))
  val stateTable = Reg(Vec(count, new FrontendSdfState(cfg)))
  val free = VecInit(active.map(!_))
  val allocOH = PriorityEncoderOH(free)
  val allocId = OHToUInt(allocOH)
  // Allocation writes the ray table first and only forwards a registered,
  // narrow context ID to Init.  The slot can be replaced when Init consumes
  // the previous ID, preserving one allocation per cycle in steady state.
  val initIssueValid = RegInit(false.B)
  val initIssueId = Reg(UInt(idBits.W))
  val initIssueReady = !initIssueValid || io.init_issue.ready
  io.init_issue.valid := initIssueValid
  io.init_issue.bits := initIssueId
  io.ingress.ready := free.asUInt.orR && initIssueReady

  when(io.ingress.fire) {
    assert(!active(allocId), "Frontend allocator selected an active context")
    active(allocId) := true.B
    immutableTable(allocId).origin := io.ingress.bits.origin
    immutableTable(allocId).dir := io.ingress.bits.dir
    immutableTable(allocId).meta.rayId := io.ingress.bits.rayId
    stateTable(allocId) := 0.U.asTypeOf(new FrontendSdfState(cfg))
    sdfReady(allocId) := false.B
    initIssueId := allocId
    initIssueValid := true.B
  }.elsewhen(io.init_issue.fire) {
    initIssueValid := false.B
  }

  when(io.init_complete.valid) {
    val completeId = io.init_complete.bits.contextId
    assert(active(completeId), "Init completed an inactive frontend context")
    when(io.init_complete.bits.hit) {
      stateTable(completeId).dist := io.init_complete.bits.entryT
      stateTable(completeId).iter := 0.U
      stateTable(completeId).prevSdf := 0.U
      sdfReady(completeId) := true.B
    }
  }

  for (port <- 0 until stateWritePorts) {
    when(io.state_write(port).valid) {
      assert(active(io.state_write(port).bits.contextId), "Frontend state write targeted an inactive context")
      stateTable(io.state_write(port).bits.contextId) := io.state_write(port).bits.state
    }
  }
  for (a <- 0 until stateWritePorts; b <- a + 1 until stateWritePorts) {
    assert(!(io.state_write(a).valid && io.state_write(b).valid &&
      io.state_write(a).bits.contextId === io.state_write(b).bits.contextId),
      "Frontend context state written by multiple ports in one cycle")
  }
  for (writePort <- 0 until stateWritePorts) {
    assert(!(io.init_complete.valid && io.init_complete.bits.hit &&
      io.state_write(writePort).valid &&
      io.init_complete.bits.contextId === io.state_write(writePort).bits.contextId),
      "Frontend context written by Init and SDF in one cycle")
  }

  val sdfIssueRr = RegInit(0.U(idBits.W))
  val sdfIssueId = RegInit(0.U(idBits.W))
  val sdfIssueValid = RegInit(false.B)
  val sdfRotated = Wire(Vec(count, Bool()))
  for (offset <- 0 until count) {
    val ctx = (sdfIssueRr + offset.U)(idBits - 1, 0)
    sdfRotated(offset) := sdfReady(ctx) &&
      !(sdfIssueValid && sdfIssueId === ctx)
  }
  val sdfPickOH = PriorityEncoderOH(sdfRotated)
  val sdfPickId = (OHToUInt(sdfPickOH) + sdfIssueRr)(idBits - 1, 0)
  val sdfPickValid = sdfRotated.asUInt.orR

  io.sdf_issue.valid := sdfIssueValid
  io.sdf_issue.bits := sdfIssueId
  when(io.sdf_issue.fire) {
    assert(sdfReady(sdfIssueId), "SDF claimed a context that was not initialized")
    sdfReady(sdfIssueId) := false.B
    sdfIssueRr := sdfIssueId + 1.U
    sdfIssueValid := false.B
  }
  when(!sdfIssueValid || io.sdf_issue.fire) {
    when(sdfPickValid) {
      sdfIssueId := sdfPickId
      sdfIssueValid := true.B
    }
  }

  for (port <- 0 until readPorts) {
    io.immutable(port) := immutableTable(io.read_id(port))
    io.sdf_state(port) := stateTable(io.read_id(port))
    for (writePort <- 0 until stateWritePorts) {
      when(io.state_write(writePort).valid &&
        io.state_write(writePort).bits.contextId === io.read_id(port)) {
        io.sdf_state(port) := io.state_write(writePort).bits.state
      }
    }
  }

  for (a <- 0 until releasePorts; b <- a + 1 until releasePorts) {
    assert(!(io.release(a).valid && io.release(b).valid &&
      io.release(a).bits === io.release(b).bits),
      "Frontend context released by multiple ports in one cycle")
  }
  for (port <- 0 until releasePorts) {
    when(io.release(port).valid) {
      assert(active(io.release(port).bits), "Frontend released an inactive context")
      active(io.release(port).bits) := false.B
      sdfReady(io.release(port).bits) := false.B
    }
  }

  io.active_count := PopCount(active)
}
