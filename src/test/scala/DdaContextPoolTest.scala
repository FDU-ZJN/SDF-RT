import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec
import DDA.DdaScheduler
import raytrace_utils._

class DdaContextPoolTest extends AnyFlatSpec with ChiselScalatestTester {
  behavior of "DDA streaming context pool"

  private val cfg = FloatConfig.FP32

  private def clearInputs(dut: DdaScheduler): Unit = {
    dut.io.context_in.valid.poke(false.B)
    dut.io.step_in.ready.poke(false.B)
    dut.io.step_out.valid.poke(false.B)
    dut.io.trace_claim.ready.poke(false.B)
    for (lane <- 0 until GlobalConfig.traceNumWorkers * 2) {
      dut.io.trace_cmd_context(lane).poke(0.U)
      dut.io.trace_cmd_active(lane).poke(false.B)
      dut.io.trace_cmd(lane).ready.poke(false.B)
    }
    for (worker <- 0 until GlobalConfig.traceNumWorkers) {
      dut.io.trace_stop(worker).valid.poke(false.B)
      dut.io.trace_stop(worker).bits.poke(0.U)
      dut.io.trace_release(worker).valid.poke(false.B)
      dut.io.trace_release(worker).bits.poke(0.U)
    }
  }

  private def commitContext(dut: DdaScheduler, id: Int, rayId: Int): Unit = {
    dut.io.context_in.valid.poke(true.B)
    dut.io.context_in.bits.sourceFrontendId.poke(0.U)
    dut.io.context_in.bits.contextId.poke(id.U)
    dut.io.context_in.bits.immutable.origin.x.poke(0.U)
    dut.io.context_in.bits.immutable.origin.y.poke(0.U)
    dut.io.context_in.bits.immutable.origin.z.poke(0.U)
    dut.io.context_in.bits.immutable.dir.x.poke(0.U)
    dut.io.context_in.bits.immutable.dir.y.poke(0.U)
    dut.io.context_in.bits.immutable.dir.z.poke(0.U)
    dut.io.context_in.bits.immutable.meta.rayId.poke(rayId.U)
    pokeState(dut.io.context_in.bits.traversal, id, iter = 0)
    dut.clock.step()
    dut.io.context_in.valid.poke(false.B)
  }

  private def pokeState(state: DdaTraversalState, id: Int, iter: Int): Unit = {
    state.contextId.poke(id.U)
    state.subX.poke(0.S); state.subY.poke(0.S); state.subZ.poke(0.S)
    state.iter.poke(iter.U)
    state.tMaxX.poke(0.U); state.tMaxY.poke(0.U); state.tMaxZ.poke(0.U)
    state.tDeltaX.poke(0.U); state.tDeltaY.poke(0.U); state.tDeltaZ.poke(0.U)
    state.dirNegX.poke(false.B); state.dirNegY.poke(false.B); state.dirNegZ.poke(false.B)
  }

  private def returnCommand(dut: DdaScheduler, id: Int, base: Int): Unit = {
    dut.io.step_out.valid.poke(true.B)
    pokeState(dut.io.step_out.bits.state, id, iter = base + 1)
    dut.io.step_out.bits.done.poke(false.B)
    dut.io.step_out.bits.watchdog.poke(false.B)
    dut.io.step_out.bits.emitCmd.poke(true.B)
    dut.io.step_out.bits.tri.base_addr.poke(base.U)
    dut.io.step_out.bits.tri.count.poke(1.U)
    dut.clock.step()
    dut.io.step_out.valid.poke(false.B)
  }

  it should "reserve all 32 context IDs before applying backpressure" in {
    test(new DdaScheduler(cfg, 32, GlobalConfig.ddaMaxSteps)) { dut =>
      clearInputs(dut)
      dut.io.init_alloc.ready.poke(true.B)
      for (id <- 0 until GlobalConfig.ddaTraceSlotCount) {
        dut.io.init_alloc.valid.expect(true.B)
        dut.io.init_alloc.bits.expect(id.U)
        dut.clock.step()
      }
      dut.io.init_alloc.valid.expect(false.B)
    }
  }

  it should "issue one DDA step per cycle across 32 resident contexts" in {
    test(new DdaScheduler(cfg, 32, GlobalConfig.ddaMaxSteps)) { dut =>
      clearInputs(dut)

      dut.io.init_alloc.ready.poke(true.B)
      for (id <- 0 until GlobalConfig.backendContextCount) {
        dut.io.init_alloc.valid.expect(true.B)
        dut.io.init_alloc.bits.expect(id.U)
        dut.clock.step()
      }
      dut.io.init_alloc.ready.poke(false.B)

      for (id <- 0 until GlobalConfig.backendContextCount) {
        commitContext(dut, id = id, rayId = 100 + id)
      }

      dut.io.step_in.ready.poke(true.B)
      for (id <- 0 until GlobalConfig.backendContextCount) {
        dut.io.step_in.valid.expect(true.B)
        dut.io.step_in.bits.contextId.expect(id.U)
        dut.clock.step()
      }
      dut.io.step_in.valid.expect(false.B)
    }
  }

  it should "fill, stream, stop, and safely recycle a context" in {
    test(new DdaScheduler(cfg, 32, GlobalConfig.ddaMaxSteps)) { dut =>
      clearInputs(dut)
      dut.io.init_alloc.ready.poke(true.B)
      // The first allocator handshake reserves context zero.
      dut.io.init_alloc.valid.expect(true.B)
      dut.io.init_alloc.bits.expect(0.U)
      dut.clock.step()
      dut.io.init_alloc.ready.poke(false.B)
      commitContext(dut, id = 0, rayId = 23)

      // With no consumer, eight DDA returns fill the context-local FIFO.
      for (cmd <- 0 until GlobalConfig.backendContextCmdFifoDepth) {
        dut.io.step_in.ready.poke(true.B)
        dut.io.step_in.valid.expect(true.B)
        dut.io.step_in.bits.contextId.expect(0.U)
        dut.clock.step()
        dut.io.step_in.ready.poke(false.B)
        returnCommand(dut, id = 0, base = cmd)
      }
      dut.io.step_in.ready.poke(true.B)
      dut.io.step_in.valid.expect(false.B)
      dut.io.step_in.ready.poke(false.B)

      // A ready Trace lane claims the context and frees one FIFO entry.
      dut.io.trace_claim.valid.expect(true.B)
      dut.io.trace_claim.bits.contextId.expect(0.U)
      dut.io.trace_claim.ready.poke(true.B)
      dut.clock.step()
      dut.io.trace_claim.ready.poke(false.B)
      dut.io.trace_cmd_context(0).poke(0.U)
      dut.io.trace_cmd_active(0).poke(true.B)
      dut.io.trace_cmd(0).valid.expect(true.B)
      dut.io.trace_cmd(0).bits.base_addr.expect(0.U)
      dut.io.trace_cmd(0).ready.poke(true.B)
      dut.clock.step()
      dut.io.trace_cmd(0).ready.poke(false.B)

      // DDA may run again. Stop arriving while that step is in flight must
      // discard its return instead of appending a ninth command.
      dut.io.step_in.ready.poke(true.B)
      dut.io.step_in.valid.expect(true.B)
      dut.clock.step()
      dut.io.step_in.ready.poke(false.B)
      dut.io.trace_stop(0).valid.poke(true.B)
      dut.io.trace_stop(0).bits.poke(0.U)
      dut.clock.step()
      dut.io.trace_stop(0).valid.poke(false.B)

      // Release and the stale DDA return deliberately share one edge. This
      // must free the context immediately instead of leaking it forever.
      dut.io.trace_release(0).valid.poke(true.B)
      dut.io.trace_release(0).bits.poke(0.U)
      dut.io.step_out.valid.poke(true.B)
      pokeState(dut.io.step_out.bits.state, 0, iter = 0)
      dut.io.step_out.bits.done.poke(false.B)
      dut.io.step_out.bits.watchdog.poke(false.B)
      dut.io.step_out.bits.emitCmd.poke(true.B)
      dut.io.step_out.bits.tri.base_addr.poke(99.U)
      dut.io.step_out.bits.tri.count.poke(1.U)
      dut.clock.step()
      dut.io.trace_release(0).valid.poke(false.B)
      dut.io.step_out.valid.poke(false.B)

      // Recycling is delayed until the stale DDA return has retired.
      dut.io.init_alloc.ready.poke(false.B)
      dut.io.init_alloc.valid.expect(true.B)
      dut.io.init_alloc.bits.expect(0.U)
      dut.io.watchdog_error.expect(false.B)
    }
  }

  it should "withhold trace completion until the final queued command is consumed" in {
    test(new DdaScheduler(cfg, 32, GlobalConfig.ddaMaxSteps)) { dut =>
      clearInputs(dut)
      dut.io.init_alloc.ready.poke(true.B)
      dut.io.init_alloc.bits.expect(0.U)
      dut.clock.step()
      dut.io.init_alloc.ready.poke(false.B)
      commitContext(dut, id = 0, rayId = 31)

      // Produce one command, then bind the context to Trace without consuming it.
      dut.io.step_in.ready.poke(true.B)
      dut.clock.step()
      dut.io.step_in.ready.poke(false.B)
      returnCommand(dut, id = 0, base = 7)
      dut.io.trace_claim.ready.poke(true.B)
      dut.clock.step()
      dut.io.trace_claim.ready.poke(false.B)
      dut.io.trace_cmd_context(0).poke(0.U)
      dut.io.trace_cmd_active(0).poke(true.B)

      // The following DDA step terminates traversal while command 7 remains
      // in the context FIFO.
      dut.io.step_in.ready.poke(true.B)
      dut.clock.step()
      dut.io.step_in.ready.poke(false.B)
      dut.io.step_out.valid.poke(true.B)
      pokeState(dut.io.step_out.bits.state, 0, iter = 0)
      dut.io.step_out.bits.done.poke(true.B)
      dut.io.step_out.bits.watchdog.poke(false.B)
      dut.io.step_out.bits.emitCmd.poke(false.B)
      dut.clock.step()
      dut.io.step_out.valid.poke(false.B)

      dut.io.trace_cmd(0).valid.expect(true.B)
      dut.io.trace_cmd(0).bits.base_addr.expect(7.U)
      dut.io.trace_dda_done(0).expect(false.B)

      dut.io.trace_cmd(0).ready.poke(true.B)
      dut.clock.step()
      dut.io.trace_cmd(0).ready.poke(false.B)
      dut.io.trace_cmd(0).valid.expect(false.B)
      dut.io.trace_dda_done(0).expect(true.B)
    }
  }
}
