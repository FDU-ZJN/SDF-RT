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

  val rm = RNE
  val aabbLatency = 4 + cfg.faddLatency + cfg.fdivLatency + cfg.fmulLatency
  val entryLatency = cfg.faddLatency
  val pointLatency = cfg.fmulLatency + cfg.faddLatency
  val entryAdvanceBits = java.lang.Float.floatToRawIntBits(entryAdvance).U(cfg.totalWidth.W)

  val rayWire = Wire(new Ray(cfg))
  rayWire.origin := io.setup_origin
  rayWire.dir := io.in.bits.rd

  aabb.io.ray := rayWire
  aabb.io.aabb.min := io.setup_grid_min
  aabb.io.aabb.max := io.setup_grid_max

  io.in.ready := true.B
  val inFire = io.in.fire
  aabb.io.in_valid := inFire

  val rdXAtAabb = ShiftRegister(io.in.bits.rd.x, aabbLatency)
  val rdYAtAabb = ShiftRegister(io.in.bits.rd.y, aabbLatency)
  val rdZAtAabb = ShiftRegister(io.in.bits.rd.z, aabbLatency)

  val roXAtAabb = ShiftRegister(io.setup_origin.x, aabbLatency)
  val roYAtAabb = ShiftRegister(io.setup_origin.y, aabbLatency)
  val roZAtAabb = ShiftRegister(io.setup_origin.z, aabbLatency)

  val slotAtAabb = ShiftRegister(io.in.bits.meta.slotId, aabbLatency)

  val tEntry = Module(new FADD(cfg))
  tEntry.io.a := aabb.io.tNear
  tEntry.io.b := entryAdvanceBits
  tEntry.io.rm := rm

  val rdXAtEntry = ShiftRegister(rdXAtAabb, entryLatency)
  val rdYAtEntry = ShiftRegister(rdYAtAabb, entryLatency)
  val rdZAtEntry = ShiftRegister(rdZAtAabb, entryLatency)

  val roXAtEntry = ShiftRegister(roXAtAabb, entryLatency)
  val roYAtEntry = ShiftRegister(roYAtAabb, entryLatency)
  val roZAtEntry = ShiftRegister(roZAtAabb, entryLatency)

  val slotAtEntry = ShiftRegister(slotAtAabb, entryLatency)

  val mulX = Module(new FMUL(cfg))
  val mulY = Module(new FMUL(cfg))
  val mulZ = Module(new FMUL(cfg))

  mulX.io.a := rdXAtEntry
  mulX.io.b := tEntry.io.res
  mulX.io.rm := rm
  mulY.io.a := rdYAtEntry
  mulY.io.b := tEntry.io.res
  mulY.io.rm := rm
  mulZ.io.a := rdZAtEntry
  mulZ.io.b := tEntry.io.res
  mulZ.io.rm := rm

  val addX = Module(new FADD(cfg))
  val addY = Module(new FADD(cfg))
  val addZ = Module(new FADD(cfg))

  addX.io.a := roXAtEntry
  addX.io.b := mulX.io.result
  addX.io.rm := rm
  addY.io.a := roYAtEntry
  addY.io.b := mulY.io.result
  addY.io.rm := rm
  addZ.io.a := roZAtEntry
  addZ.io.b := mulZ.io.result
  addZ.io.rm := rm

  val outAlignLatency = entryLatency + pointLatency
  val outValid = ShiftRegister(aabb.io.out_valid, outAlignLatency)
  val outHit = ShiftRegister(aabb.io.hit, outAlignLatency)
  val outSlot = ShiftRegister(slotAtEntry, pointLatency)
  val outRdX = ShiftRegister(rdXAtEntry, pointLatency)
  val outRdY = ShiftRegister(rdYAtEntry, pointLatency)
  val outRdZ = ShiftRegister(rdZAtEntry, pointLatency)

  io.to_sdf.valid := outValid && outHit
  io.to_sdf.bits.ray.origin.x := addX.io.res
  io.to_sdf.bits.ray.origin.y := addY.io.res
  io.to_sdf.bits.ray.origin.z := addZ.io.res
  io.to_sdf.bits.ray.dir.x := outRdX
  io.to_sdf.bits.ray.dir.y := outRdY
  io.to_sdf.bits.ray.dir.z := outRdZ
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
