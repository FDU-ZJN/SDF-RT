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
    dut.io.context_in.bits.ray.origin.x.poke(0.U)
    dut.io.context_in.bits.ray.origin.y.poke(0.U)
    dut.io.context_in.bits.ray.origin.z.poke(0.U)
    dut.io.context_in.bits.ray.dir.x.poke(0.U)
    dut.io.context_in.bits.ray.dir.y.poke(0.U)
    dut.io.context_in.bits.ray.dir.z.poke(0.U)
    dut.io.context_in.bits.ray.dist.poke(0.U)
    dut.io.context_in.bits.traceSlot.poke(id.U)
    dut.io.context_in.bits.meta.rayId.poke(rayId.U)
    dut.io.context_in.bits.initialized.poke(true.B)
    dut.io.context_in.bits.subX.poke(0.S)
    dut.io.context_in.bits.subY.poke(0.S)
    dut.io.context_in.bits.subZ.poke(0.S)
    dut.io.context_in.bits.iter.poke(0.U)
    dut.io.context_in.bits.tMaxX.poke(0.U)
    dut.io.context_in.bits.tMaxY.poke(0.U)
    dut.io.context_in.bits.tMaxZ.poke(0.U)
    dut.io.context_in.bits.tDeltaX.poke(0.U)
    dut.io.context_in.bits.tDeltaY.poke(0.U)
    dut.io.context_in.bits.tDeltaZ.poke(0.U)
    dut.clock.step()
    dut.io.context_in.valid.poke(false.B)
  }

  private def returnCommand(dut: DdaScheduler, id: Int, base: Int): Unit = {
    dut.io.step_out.valid.poke(true.B)
    dut.io.step_out.bits.ctx.ray.origin.x.poke(0.U)
    dut.io.step_out.bits.ctx.ray.origin.y.poke(0.U)
    dut.io.step_out.bits.ctx.ray.origin.z.poke(0.U)
    dut.io.step_out.bits.ctx.ray.dir.x.poke(0.U)
    dut.io.step_out.bits.ctx.ray.dir.y.poke(0.U)
    dut.io.step_out.bits.ctx.ray.dir.z.poke(0.U)
    dut.io.step_out.bits.ctx.ray.dist.poke(0.U)
    dut.io.step_out.bits.ctx.traceSlot.poke(id.U)
    dut.io.step_out.bits.ctx.meta.rayId.poke(23.U)
    dut.io.step_out.bits.ctx.initialized.poke(true.B)
    dut.io.step_out.bits.ctx.subX.poke(0.S)
    dut.io.step_out.bits.ctx.subY.poke(0.S)
    dut.io.step_out.bits.ctx.subZ.poke(0.S)
    dut.io.step_out.bits.ctx.iter.poke((base + 1).U)
    dut.io.step_out.bits.ctx.tMaxX.poke(0.U)
    dut.io.step_out.bits.ctx.tMaxY.poke(0.U)
    dut.io.step_out.bits.ctx.tMaxZ.poke(0.U)
    dut.io.step_out.bits.ctx.tDeltaX.poke(0.U)
    dut.io.step_out.bits.ctx.tDeltaY.poke(0.U)
    dut.io.step_out.bits.ctx.tDeltaZ.poke(0.U)
    dut.io.step_out.bits.done.poke(false.B)
    dut.io.step_out.bits.watchdog.poke(false.B)
    dut.io.step_out.bits.emitCmd.poke(true.B)
    dut.io.step_out.bits.tri.base_addr.poke(base.U)
    dut.io.step_out.bits.tri.count.poke(1.U)
    dut.io.step_out.bits.cellExitT.poke(java.lang.Float.floatToRawIntBits(base + 1.0f).U)
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
      for (cmd <- 0 until GlobalConfig.ddaTraceCmdDepth) {
        dut.io.step_in.ready.poke(true.B)
        dut.io.step_in.valid.expect(true.B)
        dut.io.step_in.bits.traceSlot.expect(0.U)
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
      dut.io.trace_cmd(0).bits.tri.base_addr.expect(0.U)
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
      dut.io.step_out.bits.ctx.traceSlot.poke(0.U)
      dut.io.step_out.bits.done.poke(false.B)
      dut.io.step_out.bits.watchdog.poke(false.B)
      dut.io.step_out.bits.emitCmd.poke(true.B)
      dut.io.step_out.bits.tri.base_addr.poke(99.U)
      dut.io.step_out.bits.tri.count.poke(1.U)
      dut.io.step_out.bits.cellExitT.poke(java.lang.Float.floatToRawIntBits(100.0f).U)
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
      dut.io.step_out.bits.ctx.traceSlot.poke(0.U)
      dut.io.step_out.bits.done.poke(true.B)
      dut.io.step_out.bits.watchdog.poke(false.B)
      dut.io.step_out.bits.emitCmd.poke(false.B)
      dut.clock.step()
      dut.io.step_out.valid.poke(false.B)

      dut.io.trace_cmd(0).valid.expect(true.B)
      dut.io.trace_cmd(0).bits.tri.base_addr.expect(7.U)
      dut.io.trace_dda_done(0).expect(false.B)

      dut.io.trace_cmd(0).ready.poke(true.B)
      dut.clock.step()
      dut.io.trace_cmd(0).ready.poke(false.B)
      dut.io.trace_cmd(0).valid.expect(false.B)
      dut.io.trace_dda_done(0).expect(true.B)
    }
  }
}
