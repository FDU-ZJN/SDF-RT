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
    val read_id = Input(Vec(readPorts, UInt(idBits.W)))
    val immutable = Output(Vec(readPorts, new ImmutableRayRecord(cfg, addrWidth)))
    val sdf_state = Output(Vec(readPorts, new FrontendSdfState(cfg)))
    val state_write = Input(Vec(stateWritePorts, Valid(new FrontendStateWrite(cfg))))
    val release = Input(Vec(releasePorts, Valid(UInt(idBits.W))))
    val active_count = Output(UInt(log2Ceil(count + 1).W))
  })

  val active = RegInit(VecInit(Seq.fill(count)(false.B)))
  val immutableTable = Reg(Vec(count, new ImmutableRayRecord(cfg, addrWidth)))
  val stateTable = Reg(Vec(count, new FrontendSdfState(cfg)))
  val free = VecInit(active.map(!_))
  val allocOH = PriorityEncoderOH(free)
  val allocId = OHToUInt(allocOH)
  io.init_issue.valid := io.ingress.valid && free.asUInt.orR
  io.init_issue.bits := allocId
  io.ingress.ready := free.asUInt.orR && io.init_issue.ready

  when(io.ingress.fire) {
    assert(!active(allocId), "Frontend allocator selected an active context")
    active(allocId) := true.B
    immutableTable(allocId).origin := io.ingress.bits.origin
    immutableTable(allocId).dir := io.ingress.bits.dir
    immutableTable(allocId).meta.rayId := io.ingress.bits.rayId
    stateTable(allocId) := 0.U.asTypeOf(new FrontendSdfState(cfg))
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

  for (port <- 0 until readPorts) {
    io.immutable(port) := immutableTable(io.read_id(port))
    io.sdf_state(port) := stateTable(io.read_id(port))
    when(io.ingress.fire && allocId === io.read_id(port)) {
      io.immutable(port).origin := io.ingress.bits.origin
      io.immutable(port).dir := io.ingress.bits.dir
      io.immutable(port).meta.rayId := io.ingress.bits.rayId
      io.sdf_state(port) := 0.U.asTypeOf(new FrontendSdfState(cfg))
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
    }
  }

  io.active_count := PopCount(active)
}
