package raytrace_utils

import chisel3._
import chisel3.util._
import raytrace_utils.fudian._

class RayAABBIntersection(cfg: FloatConfig = FloatConfig.FP32) extends Module {
  val io = IO(new Bundle {
    val ray = Input(new Ray(cfg))
    val aabb = Input(new AABB(cfg))
    val in_valid = Input(Bool())

    val hit = Output(Bool())
    val tNear = Output(UInt(cfg.totalWidth.W))
    val tFar = Output(UInt(cfg.totalWidth.W))
    val out_valid = Output(Bool())
  })

  val rm = 0.U(3.W)
  val fpZero = 0.U(cfg.totalWidth.W)
  val fpOne = cfg.oneBigInt.U(cfg.totalWidth.W)
  val fpEps = java.lang.Float.floatToIntBits(1e-9f).U(cfg.totalWidth.W)

  def neg(x: UInt): UInt = Cat(!x(cfg.totalWidth - 1), x(cfg.totalWidth - 2, 0))

  // Parallel per-axis setup: dir' = dir + eps, then invDir = 1 / dir'.
  val dirs = Seq(io.ray.dir.x, io.ray.dir.y, io.ray.dir.z)
  val mins = Seq(io.aabb.min.x, io.aabb.min.y, io.aabb.min.z)
  val maxs = Seq(io.aabb.max.x, io.aabb.max.y, io.aabb.max.z)
  val origs = Seq(io.ray.origin.x, io.ray.origin.y, io.ray.origin.z)

  val dirPlusEps = Seq.fill(3)(Wire(UInt(cfg.totalWidth.W)))
  val invDir = Seq.fill(3)(Wire(UInt(cfg.totalWidth.W)))
  val t0 = Seq.fill(3)(Wire(UInt(cfg.totalWidth.W)))
  val t1 = Seq.fill(3)(Wire(UInt(cfg.totalWidth.W)))
  val axisNear = Seq.fill(3)(Wire(UInt(cfg.totalWidth.W)))
  val axisFar = Seq.fill(3)(Wire(UInt(cfg.totalWidth.W)))

  for (i <- 0 until 3) {
    val addDirEps = Module(new FADD(cfg))
    addDirEps.io.a := dirs(i)
    addDirEps.io.b := fpEps
    addDirEps.io.rm := rm
    dirPlusEps(i) := addDirEps.io.res

    val div = Module(new FDIV(cfg))
    div.io.a := fpOne
    div.io.b := dirPlusEps(i)
    div.io.in_valid := io.in_valid
    invDir(i) := div.io.result

    val subMin = Module(new FADD(cfg))
    subMin.io.a := mins(i)
    subMin.io.b := neg(origs(i))
    subMin.io.rm := rm

    val subMax = Module(new FADD(cfg))
    subMax.io.a := maxs(i)
    subMax.io.b := neg(origs(i))
    subMax.io.rm := rm

    val subMinAligned = ShiftRegister(subMin.io.res, cfg.fdivLatency)
    val subMaxAligned = ShiftRegister(subMax.io.res, cfg.fdivLatency)

    val mul0 = Module(new FMUL(cfg))
    mul0.io.a := subMinAligned
    mul0.io.b := invDir(i)
    mul0.io.rm := rm
    t0(i) := mul0.io.result

    val mul1 = Module(new FMUL(cfg))
    mul1.io.a := subMaxAligned
    mul1.io.b := invDir(i)
    mul1.io.rm := rm
    t1(i) := mul1.io.result

    val cmpAxis = Module(new FCMP(cfg))
    cmpAxis.io.a := t0(i)
    cmpAxis.io.b := t1(i)
    cmpAxis.io.signaling := false.B

    axisNear(i) := Mux(cmpAxis.io.le, t0(i), t1(i))
    axisFar(i) := Mux(cmpAxis.io.le, t1(i), t0(i))
  }

  // Stage align: per-axis values are ready after 2 + 6 + 3 = 11 cycles.
  val nearS0 = VecInit(axisNear).map(v => RegNext(v))
  val farS0 = VecInit(axisFar).map(v => RegNext(v))

  // Reduction level 1: tMin01=max(near0,near1), tMax01=min(far0,far1)
  val cmpNear01 = Module(new FCMP(cfg))
  cmpNear01.io.a := nearS0(0)
  cmpNear01.io.b := nearS0(1)
  cmpNear01.io.signaling := false.B
  val tMin01 = RegNext(Mux(cmpNear01.io.le, nearS0(1), nearS0(0)))

  val cmpFar01 = Module(new FCMP(cfg))
  cmpFar01.io.a := farS0(0)
  cmpFar01.io.b := farS0(1)
  cmpFar01.io.signaling := false.B
  val tMax01 = RegNext(Mux(cmpFar01.io.le, farS0(0), farS0(1)))

  val near2S1 = RegNext(nearS0(2))
  val far2S1 = RegNext(farS0(2))

  // Reduction level 2: tMin=max(tMin01,near2), tMax=min(tMax01,far2)
  val cmpNear012 = Module(new FCMP(cfg))
  cmpNear012.io.a := tMin01
  cmpNear012.io.b := near2S1
  cmpNear012.io.signaling := false.B
  val tMin = RegNext(Mux(cmpNear012.io.le, near2S1, tMin01))

  val cmpFar012 = Module(new FCMP(cfg))
  cmpFar012.io.a := tMax01
  cmpFar012.io.b := far2S1
  cmpFar012.io.signaling := false.B
  val tMax = RegNext(Mux(cmpFar012.io.le, tMax01, far2S1))

  // Hit: tMax >= tMin && tMax >= 0
  val cmpRange = Module(new FCMP(cfg))
  cmpRange.io.a := tMin
  cmpRange.io.b := tMax
  cmpRange.io.signaling := false.B
  val rangeOk = cmpRange.io.le

  val cmpTMaxNonNeg = Module(new FCMP(cfg))
  cmpTMaxNonNeg.io.a := fpZero
  cmpTMaxNonNeg.io.b := tMax
  cmpTMaxNonNeg.io.signaling := false.B
  val tMaxNonNeg = cmpTMaxNonNeg.io.le

  // tNear = (tMin > 0) ? tMin : tMax
  val cmpTMinPos = Module(new FCMP(cfg))
  cmpTMinPos.io.a := fpZero
  cmpTMinPos.io.b := tMin
  cmpTMinPos.io.signaling := false.B
  val tMinPos = cmpTMinPos.io.lt

  val tNearComb = Mux(tMinPos, tMin, tMax)
  val hitComb = rangeOk && tMaxNonNeg

  io.tNear := RegNext(tNearComb)
  io.tFar := RegNext(tMax)
  io.hit := RegNext(hitComb)

  val totalLatency = 4+cfg.faddLatency+cfg.fdivLatency+cfg.fmulLatency
  io.out_valid := ShiftRegister(io.in_valid, totalLatency)
}
