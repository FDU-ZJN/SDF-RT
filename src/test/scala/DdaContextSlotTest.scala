import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec
import Backend.ContextSlot
import raytrace_utils._

class DdaContextSlotTest extends AnyFlatSpec with ChiselScalatestTester {
  behavior of "ContextSlot"

  private def clearInputs(dut: ContextSlot): Unit = {
    dut.io.reserveInit.poke(false.B)
    dut.io.initCommit.poke(false.B)
    dut.io.stepIssue.poke(false.B)
    dut.io.stepReturn.poke(false.B)
    dut.io.claim.poke(false.B)
    dut.io.stop.poke(false.B)
    dut.io.release.poke(false.B)
    dut.io.cmdOut.ready.poke(false.B)
  }

  private def pokeContext(ctx: DdaContext, id: Int, iter: Int = 0): Unit = {
    ctx.ray.origin.x.poke(0.U)
    ctx.ray.origin.y.poke(0.U)
    ctx.ray.origin.z.poke(0.U)
    ctx.ray.dir.x.poke(0.U)
    ctx.ray.dir.y.poke(0.U)
    ctx.ray.dir.z.poke(0.U)
    ctx.ray.dist.poke(0.U)
    ctx.meta.rayId.poke(23.U)
    ctx.traceSlot.poke(id.U)
    ctx.initialized.poke(true.B)
    ctx.subX.poke(0.S)
    ctx.subY.poke(0.S)
    ctx.subZ.poke(0.S)
    ctx.iter.poke(iter.U)
    ctx.tMaxX.poke(0.U)
    ctx.tMaxY.poke(0.U)
    ctx.tMaxZ.poke(0.U)
    ctx.tDeltaX.poke(0.U)
    ctx.tDeltaY.poke(0.U)
    ctx.tDeltaZ.poke(0.U)
  }

  private def pokeStepResult(dut: ContextSlot, id: Int, emitCmd: Boolean): Unit = {
    pokeContext(dut.io.stepResult.ctx, id, iter = 1)
    dut.io.stepResult.done.poke(false.B)
    dut.io.stepResult.watchdog.poke(false.B)
    dut.io.stepResult.emitCmd.poke(emitCmd.B)
    dut.io.stepResult.tri.base_addr.poke(17.U)
    dut.io.stepResult.tri.count.poke(2.U)
    dut.io.stepResult.cellExitT.poke("h3f800000".U)
  }

  private def reserveAndCommit(dut: ContextSlot, id: Int): Unit = {
    dut.io.reserveInit.poke(true.B)
    dut.clock.step()
    dut.io.reserveInit.poke(false.B)
    pokeContext(dut.io.initContext, id)
    dut.io.initCommit.poke(true.B)
    dut.clock.step()
    dut.io.initCommit.poke(false.B)
  }

  it should "buffer commands and recycle on a simultaneous release and return" in {
    test(new ContextSlot(FloatConfig.FP32, 32, traceCmdDepth = 2)) { dut =>
      clearInputs(dut)
      pokeContext(dut.io.initContext, 0)
      pokeStepResult(dut, 0, emitCmd = false)
      dut.io.freeForInit.expect(true.B)
      reserveAndCommit(dut, id = 0)
      dut.io.active.expect(true.B)

      dut.io.stepEligible.expect(true.B)
      dut.io.stepIssue.poke(true.B)
      dut.clock.step()
      dut.io.stepIssue.poke(false.B)
      pokeStepResult(dut, 0, emitCmd = true)
      dut.io.stepReturn.poke(true.B)
      dut.clock.step()
      dut.io.stepReturn.poke(false.B)

      dut.io.cmdOut.valid.expect(true.B)
      dut.io.cmdOut.bits.tri.base_addr.expect(17.U)
      dut.io.claimEligible.expect(true.B)
      dut.io.claim.poke(true.B)
      dut.clock.step()
      dut.io.claim.poke(false.B)
      dut.io.claimed.expect(true.B)

      dut.io.cmdOut.ready.poke(true.B)
      dut.clock.step()
      dut.io.cmdOut.ready.poke(false.B)
      dut.io.cmdOut.valid.expect(false.B)

      dut.io.stepIssue.poke(true.B)
      dut.clock.step()
      dut.io.stepIssue.poke(false.B)
      pokeStepResult(dut, 0, emitCmd = true)
      dut.io.stop.poke(true.B)
      dut.io.release.poke(true.B)
      dut.io.stepReturn.poke(true.B)
      dut.clock.step()
      clearInputs(dut)

      dut.io.active.expect(false.B)
      dut.io.freeForInit.expect(true.B)
      dut.io.cmdOut.valid.expect(false.B)
    }
  }

  it should "wait for a stale step return before recycling a released slot" in {
    test(new ContextSlot(FloatConfig.FP32, 32, traceCmdDepth = 2)) { dut =>
      clearInputs(dut)
      pokeContext(dut.io.initContext, 0)
      pokeStepResult(dut, 0, emitCmd = false)
      reserveAndCommit(dut, id = 0)

      dut.io.stepIssue.poke(true.B)
      dut.clock.step()
      dut.io.stepIssue.poke(false.B)
      pokeStepResult(dut, 0, emitCmd = true)
      dut.io.stepReturn.poke(true.B)
      dut.clock.step()
      dut.io.stepReturn.poke(false.B)

      dut.io.claimEligible.expect(true.B)
      dut.io.claim.poke(true.B)
      dut.clock.step()
      dut.io.claim.poke(false.B)
      dut.io.cmdOut.ready.poke(true.B)
      dut.clock.step()
      dut.io.cmdOut.ready.poke(false.B)

      dut.io.stepIssue.poke(true.B)
      dut.clock.step()
      dut.io.stepIssue.poke(false.B)

      dut.io.release.poke(true.B)
      dut.clock.step()
      dut.io.release.poke(false.B)
      dut.io.active.expect(true.B)
      dut.io.freeForInit.expect(false.B)

      pokeStepResult(dut, 0, emitCmd = true)
      dut.io.stepReturn.poke(true.B)
      dut.clock.step()
      dut.io.stepReturn.poke(false.B)
      dut.io.active.expect(false.B)
      dut.io.freeForInit.expect(true.B)
      dut.io.cmdOut.valid.expect(false.B)
    }
  }
}
