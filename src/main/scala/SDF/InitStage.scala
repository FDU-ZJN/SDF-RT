package SDF

import chisel3._
import chisel3.util._
import raytrace_utils._
import raytrace_utils.fudian._

class InitStage(cfg: FloatConfig, addrWidth: Int, entryAdvance: Float = 1e-4f) extends Module {
  val io = IO(new Bundle {
    val setup_origin = Input(new Vec3(cfg))
    val setup_grid_min = Input(new Vec3(cfg))
    val setup_grid_max = Input(new Vec3(cfg))

    val in = Flipped(Decoupled(new SdfInitReq(cfg, addrWidth)))
    val to_sdf = Decoupled(new RayIssue(cfg, addrWidth))
    val to_bypass = Decoupled(new SdfBypassResp(addrWidth))
  })

  val aabb = Module(new RayAABBIntersection(cfg))

  val aabbLatency = 4 + cfg.faddLatency + cfg.fdivLatency + cfg.fmulLatency + (4 * cfg.fcmpLatency)
  val entryLatency = cfg.faddLatency
  val entryAdvanceBits = java.lang.Float.floatToRawIntBits(entryAdvance).U(cfg.totalWidth.W)

  val rayWire = Wire(new Ray(cfg))
  rayWire.origin := io.setup_origin
  rayWire.dir := io.in.bits.rd
  rayWire.dist := 0.U

  aabb.io.ray := rayWire
  aabb.io.aabb.min := io.setup_grid_min
  aabb.io.aabb.max := io.setup_grid_max

  io.in.ready := true.B
  val inFire = io.in.fire
  aabb.io.in_valid := inFire

  val rdXAtAabb = PipeUtils.pipeData(io.in.bits.rd.x, aabbLatency)
  val rdYAtAabb = PipeUtils.pipeData(io.in.bits.rd.y, aabbLatency)
  val rdZAtAabb = PipeUtils.pipeData(io.in.bits.rd.z, aabbLatency)

  val roXAtAabb = PipeUtils.pipeData(io.setup_origin.x, aabbLatency)
  val roYAtAabb = PipeUtils.pipeData(io.setup_origin.y, aabbLatency)
  val roZAtAabb = PipeUtils.pipeData(io.setup_origin.z, aabbLatency)

  val slotAtAabb = PipeUtils.pipeData(io.in.bits.meta.slotId, aabbLatency)

  val tEntry = Module(new FADD(cfg))
  tEntry.io.a := aabb.io.tNear
  tEntry.io.b := entryAdvanceBits

  val outAlignLatency = entryLatency
  val outValid = PipeUtils.pipeData(aabb.io.out_valid, outAlignLatency)
  val outHit = PipeUtils.pipeData(aabb.io.hit, outAlignLatency)
  val outSlot = PipeUtils.pipeData(slotAtAabb, entryLatency)
  val outRoX = PipeUtils.pipeData(roXAtAabb, entryLatency)
  val outRoY = PipeUtils.pipeData(roYAtAabb, entryLatency)
  val outRoZ = PipeUtils.pipeData(roZAtAabb, entryLatency)
  val outRdX = PipeUtils.pipeData(rdXAtAabb, entryLatency)
  val outRdY = PipeUtils.pipeData(rdYAtAabb, entryLatency)
  val outRdZ = PipeUtils.pipeData(rdZAtAabb, entryLatency)

  io.to_sdf.valid := outValid && outHit
  io.to_sdf.bits.ray.origin.x := outRoX
  io.to_sdf.bits.ray.origin.y := outRoY
  io.to_sdf.bits.ray.origin.z := outRoZ
  io.to_sdf.bits.ray.dir.x := outRdX
  io.to_sdf.bits.ray.dir.y := outRdY
  io.to_sdf.bits.ray.dir.z := outRdZ
  io.to_sdf.bits.ray.dist := tEntry.io.res
  io.to_sdf.bits.meta.slotId := outSlot
  io.to_sdf.bits.meta.pixelX := 0.U
  io.to_sdf.bits.meta.pixelY := 0.U

  io.to_bypass.valid := outValid && !outHit
  io.to_bypass.bits.meta.slotId := outSlot
  io.to_bypass.bits.meta.pixelX := 0.U
  io.to_bypass.bits.meta.pixelY := 0.U

  when(io.to_sdf.valid) {
    assert(io.to_sdf.ready, "InitStage expects to_sdf.ready to stay high")
  }
  when(io.to_bypass.valid) {
    assert(io.to_bypass.ready, "InitStage expects to_bypass.ready to stay high")
  }
}
