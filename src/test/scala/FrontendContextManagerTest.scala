import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec
import SDF.FrontendContextManager
import raytrace_utils._

class FrontendContextManagerTest extends AnyFlatSpec with ChiselScalatestTester {
  behavior of "FrontendContextManager"

  it should "allocate all 32 IDs, preserve records, and recycle only released slots" in {
    test(new FrontendContextManager(FloatConfig.FP32, 32,
      readPorts = 1, releasePorts = 1, stateWritePorts = 1)) { dut =>
      dut.io.ingress.valid.poke(false.B)
      dut.io.init_issue.ready.poke(true.B)
      dut.io.init_complete.valid.poke(false.B)
      dut.io.sdf_issue.ready.poke(true.B)
      dut.io.read_id(0).poke(0.U)
      dut.io.state_write(0).valid.poke(false.B)
      dut.io.release(0).valid.poke(false.B)
      dut.clock.step()

      for (id <- 0 until GlobalConfig.frontendContextCount) {
        dut.io.ingress.bits.origin.x.poke((1000 + id).U)
        dut.io.ingress.bits.origin.y.poke(0.U)
        dut.io.ingress.bits.origin.z.poke(0.U)
        dut.io.ingress.bits.dir.x.poke(0.U)
        dut.io.ingress.bits.dir.y.poke(0.U)
        dut.io.ingress.bits.dir.z.poke(0.U)
        dut.io.ingress.bits.rayId.poke((200 + id).U)
        dut.io.ingress.valid.poke(true.B)
        dut.io.ingress.ready.expect(true.B)
        dut.io.init_issue.valid.expect(true.B)
        dut.io.init_issue.bits.expect(id.U)
        // The table bypass makes the just-allocated record visible to Init.
        dut.io.read_id(0).poke(id.U)
        dut.io.immutable(0).meta.rayId.expect((200 + id).U)
        dut.clock.step()
      }
      dut.io.ingress.valid.poke(false.B)
      dut.io.active_count.expect(GlobalConfig.frontendContextCount.U)
      dut.io.ingress.ready.expect(false.B)

      for (id <- 0 until GlobalConfig.frontendContextCount) {
        dut.io.read_id(0).poke(id.U)
        dut.io.immutable(0).origin.x.expect((1000 + id).U)
        dut.io.immutable(0).meta.rayId.expect((200 + id).U)
      }

      dut.io.release(0).valid.poke(true.B)
      dut.io.release(0).bits.poke(7.U)
      dut.clock.step()
      dut.io.release(0).valid.poke(false.B)
      dut.io.ingress.bits.rayId.poke(999.U)
      dut.io.ingress.valid.poke(true.B)
      dut.io.init_issue.bits.expect(7.U)
      dut.clock.step()
      dut.io.ingress.valid.poke(false.B)
      dut.io.read_id(0).poke(7.U)
      dut.io.immutable(0).meta.rayId.expect(999.U)
    }
  }

  it should "bypass same-cycle mutable state writes to SDF reads" in {
    test(new FrontendContextManager(FloatConfig.FP32, 32,
      readPorts = 1, releasePorts = 1, stateWritePorts = 1)) { dut =>
      dut.io.init_issue.ready.poke(true.B)
      dut.io.init_complete.valid.poke(false.B)
      dut.io.sdf_issue.ready.poke(true.B)
      dut.io.release(0).valid.poke(false.B)
      dut.io.state_write(0).valid.poke(false.B)
      dut.io.ingress.valid.poke(true.B)
      dut.io.ingress.bits.origin.x.poke(0.U); dut.io.ingress.bits.origin.y.poke(0.U); dut.io.ingress.bits.origin.z.poke(0.U)
      dut.io.ingress.bits.dir.x.poke(0.U); dut.io.ingress.bits.dir.y.poke(0.U); dut.io.ingress.bits.dir.z.poke(0.U)
      dut.io.ingress.bits.rayId.poke(12.U)
      dut.clock.step()
      dut.io.ingress.valid.poke(false.B)

      dut.io.read_id(0).poke(0.U)
      dut.io.state_write(0).valid.poke(true.B)
      dut.io.state_write(0).bits.contextId.poke(0.U)
      dut.io.state_write(0).bits.state.dist.poke("h3f800000".U)
      dut.io.state_write(0).bits.state.iter.poke(3.U)
      dut.io.state_write(0).bits.state.prevSdf.poke("h3e800000".U)
      dut.io.sdf_state(0).dist.expect("h3f800000".U)
      dut.io.sdf_state(0).iter.expect(3.U)
      dut.clock.step()
      dut.io.state_write(0).valid.poke(false.B)
      dut.io.sdf_state(0).iter.expect(3.U)
    }
  }

  it should "limit same-cycle ingress bypass to the Init read port" in {
    test(new FrontendContextManager(FloatConfig.FP32, 32,
      readPorts = 2, releasePorts = 1, stateWritePorts = 1)) { dut =>
      dut.io.init_issue.ready.poke(true.B)
      dut.io.init_complete.valid.poke(false.B)
      dut.io.sdf_issue.ready.poke(true.B)
      dut.io.state_write(0).valid.poke(false.B)
      dut.io.release(0).valid.poke(false.B)
      dut.io.read_id(0).poke(0.U)
      dut.io.read_id(1).poke(0.U)

      dut.io.ingress.bits.origin.x.poke(1.U)
      dut.io.ingress.bits.origin.y.poke(2.U)
      dut.io.ingress.bits.origin.z.poke(3.U)
      dut.io.ingress.bits.dir.x.poke(4.U)
      dut.io.ingress.bits.dir.y.poke(5.U)
      dut.io.ingress.bits.dir.z.poke(6.U)
      dut.io.ingress.bits.rayId.poke(10.U)
      dut.io.ingress.valid.poke(true.B)
      dut.clock.step()
      dut.io.ingress.valid.poke(false.B)

      dut.io.release(0).valid.poke(true.B)
      dut.io.release(0).bits.poke(0.U)
      dut.clock.step()
      dut.io.release(0).valid.poke(false.B)

      dut.io.ingress.bits.origin.x.poke(11.U)
      dut.io.ingress.bits.rayId.poke(20.U)
      dut.io.ingress.valid.poke(true.B)
      dut.io.ingress.ready.expect(true.B)
      dut.io.immutable(0).meta.rayId.expect(20.U)
      dut.io.immutable(1).meta.rayId.expect(10.U)
      dut.io.immutable(1).origin.x.expect(1.U)
      dut.clock.step()
      dut.io.ingress.valid.poke(false.B)
      dut.io.immutable(1).meta.rayId.expect(20.U)
      dut.io.immutable(1).origin.x.expect(11.U)
    }
  }

  it should "advertise a context to SDF only after a hit Init writeback" in {
    test(new FrontendContextManager(FloatConfig.FP32, 32,
      readPorts = 1, releasePorts = 1, stateWritePorts = 1)) { dut =>
      dut.io.init_issue.ready.poke(true.B)
      dut.io.init_complete.valid.poke(false.B)
      dut.io.sdf_issue.ready.poke(false.B)
      dut.io.read_id(0).poke(0.U)
      dut.io.state_write(0).valid.poke(false.B)
      dut.io.release(0).valid.poke(false.B)
      dut.io.ingress.bits.origin.x.poke(1.U)
      dut.io.ingress.bits.origin.y.poke(2.U)
      dut.io.ingress.bits.origin.z.poke(3.U)
      dut.io.ingress.bits.dir.x.poke(4.U)
      dut.io.ingress.bits.dir.y.poke(5.U)
      dut.io.ingress.bits.dir.z.poke(6.U)
      dut.io.ingress.bits.rayId.poke(7.U)
      dut.io.ingress.valid.poke(true.B)
      dut.clock.step()
      dut.io.ingress.valid.poke(false.B)
      dut.io.sdf_issue.valid.expect(false.B)

      dut.io.init_complete.bits.contextId.poke(0.U)
      dut.io.init_complete.bits.hit.poke(true.B)
      dut.io.init_complete.bits.entryT.poke("h3f800000".U)
      dut.io.init_complete.valid.poke(true.B)
      dut.clock.step()
      dut.io.init_complete.valid.poke(false.B)
      dut.io.sdf_state(0).dist.expect("h3f800000".U)
      dut.io.sdf_state(0).iter.expect(0.U)

      dut.clock.step()
      dut.io.sdf_issue.valid.expect(true.B)
      dut.io.sdf_issue.bits.expect(0.U)
      dut.clock.step(2)
      dut.io.sdf_issue.valid.expect(true.B)
      dut.io.sdf_issue.bits.expect(0.U)

      dut.io.sdf_issue.ready.poke(true.B)
      dut.clock.step()
      dut.io.sdf_issue.valid.expect(false.B)
    }
  }

  it should "hold init valid independently of downstream ready" in {
    test(new FrontendContextManager(FloatConfig.FP32, 32,
      readPorts = 1, releasePorts = 1, stateWritePorts = 1)) { dut =>
      dut.io.init_issue.ready.poke(false.B)
      dut.io.init_complete.valid.poke(false.B)
      dut.io.sdf_issue.ready.poke(true.B)
      dut.io.read_id(0).poke(0.U)
      dut.io.state_write(0).valid.poke(false.B)
      dut.io.release(0).valid.poke(false.B)
      dut.io.ingress.bits.origin.x.poke(1.U)
      dut.io.ingress.bits.origin.y.poke(2.U)
      dut.io.ingress.bits.origin.z.poke(3.U)
      dut.io.ingress.bits.dir.x.poke(4.U)
      dut.io.ingress.bits.dir.y.poke(5.U)
      dut.io.ingress.bits.dir.z.poke(6.U)
      dut.io.ingress.bits.rayId.poke(7.U)
      dut.io.ingress.valid.poke(true.B)

      dut.io.init_issue.valid.expect(true.B)
      dut.io.init_issue.bits.expect(0.U)
      dut.io.ingress.ready.expect(false.B)
      dut.clock.step(3)
      dut.io.init_issue.valid.expect(true.B)
      dut.io.init_issue.bits.expect(0.U)
      dut.io.active_count.expect(0.U)

      dut.io.init_issue.ready.poke(true.B)
      dut.io.ingress.ready.expect(true.B)
      dut.clock.step()
      dut.io.active_count.expect(1.U)
    }
  }
}
