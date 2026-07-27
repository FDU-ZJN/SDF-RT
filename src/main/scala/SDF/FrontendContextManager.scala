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
  // Keep one already-reserved context ID at the ingress boundary.  The
  // active-vector priority encoder only refills this reservation for a future
  // allocation, so it cannot combine with the incoming ray and Init's AABB
  // reciprocal datapath in the same cycle.
  val allocReserveValid = RegInit(true.B)
  val allocReserveId = RegInit(0.U(idBits.W))
  val releaseWillFree = Wire(Vec(count, Bool()))
  releaseWillFree := VecInit(Seq.fill(count)(false.B))
  for (port <- 0 until releasePorts) {
    when(io.release(port).valid) {
      releaseWillFree(io.release(port).bits) := true.B
    }
  }
  val releaseWillFreeOH = PriorityEncoderOH(releaseWillFree)
  val releaseWillFreeValid = releaseWillFree.asUInt.orR
  val releaseWillFreeId = OHToUInt(releaseWillFreeOH)
  val reserveCandidates = Wire(Vec(count, Bool()))
  for (ctx <- 0 until count) {
    reserveCandidates(ctx) :=
      ((!active(ctx) && (!allocReserveValid || allocReserveId =/= ctx.U)) ||
        releaseWillFree(ctx))
  }
  val reserveCandidateOH = PriorityEncoderOH(reserveCandidates)
  val reserveCandidateValid = reserveCandidates.asUInt.orR
  val reserveCandidateId = OHToUInt(reserveCandidateOH)

  io.init_issue.valid := io.ingress.valid && allocReserveValid
  io.init_issue.bits := allocReserveId
  io.ingress.ready := allocReserveValid && io.init_issue.ready

  when(io.ingress.fire) {
    assert(!active(allocReserveId), "Frontend allocator selected an active context")
    active(allocReserveId) := true.B
    immutableTable(allocReserveId).origin := io.ingress.bits.origin
    immutableTable(allocReserveId).dir := io.ingress.bits.dir
    immutableTable(allocReserveId).meta.rayId := io.ingress.bits.rayId
    stateTable(allocReserveId) := 0.U.asTypeOf(new FrontendSdfState(cfg))
    sdfReady(allocReserveId) := false.B
    allocReserveValid := reserveCandidateValid
    when(reserveCandidateValid) {
      allocReserveId := reserveCandidateId
    }
  }.otherwise {
    when(releaseWillFreeValid) {
      allocReserveValid := true.B
      allocReserveId := releaseWillFreeId
    }.elsewhen(!allocReserveValid && reserveCandidateValid) {
      allocReserveValid := true.B
      allocReserveId := reserveCandidateId
    }
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
    // Only Init consumes a context in its allocation cycle. Keeping ingress
    // off later read ports prevents allocator control and the full ray payload
    // from becoming combinational inputs to the SDF datapath.
    if (port == 0) {
      when(io.ingress.fire && allocReserveId === io.read_id(port)) {
        io.immutable(port).origin := io.ingress.bits.origin
        io.immutable(port).dir := io.ingress.bits.dir
        io.immutable(port).meta.rayId := io.ingress.bits.rayId
        io.sdf_state(port) := 0.U.asTypeOf(new FrontendSdfState(cfg))
      }
    }
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
