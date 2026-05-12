package raytrace_utils

import chisel3._
import chisel3.util._
import raytrace_utils.fudian._
import raytrace_utils.PipeUtils._

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
    dirPlusEps(i) := addDirEps.io.res

    val div = Module(new FRQ(cfg))
    div.io.in := dirPlusEps(i)
    div.io.in_valid := io.in_valid
    invDir(i) := div.io.result

    val subMin = Module(new FADD(cfg))
    subMin.io.a := mins(i)
    subMin.io.b := neg(origs(i))

    val subMax = Module(new FADD(cfg))
    subMax.io.a := maxs(i)
    subMax.io.b := neg(origs(i))

    val subMinAligned = PipeUtils.pipeData(subMin.io.res, cfg.fdivLatency)
    val subMaxAligned = PipeUtils.pipeData(subMax.io.res, cfg.fdivLatency)

    val mul0 = Module(new FMUL(cfg))
    mul0.io.a := subMinAligned
    mul0.io.b := invDir(i)
    t0(i) := mul0.io.result

    val mul1 = Module(new FMUL(cfg))
    mul1.io.a := subMaxAligned
    mul1.io.b := invDir(i)
    t1(i) := mul1.io.result

    val cmpAxis = Module(new FCMP(cfg))
    cmpAxis.io.a := t0(i)
    cmpAxis.io.b := t1(i)
    cmpAxis.io.signaling := false.B

    val t0Cmp = pipeUInt(t0(i), cfg.fcmpLatency)
    val t1Cmp = pipeUInt(t1(i), cfg.fcmpLatency)
    val axisLe = cmpAxis.io.le
    axisNear(i) := Mux(axisLe, t0Cmp, t1Cmp)
    axisFar(i) := Mux(axisLe, t1Cmp, t0Cmp)
  }

  // Stage align: per-axis values are ready after 2 + 6 + 3 = 11 cycles.
  val nearS0 = VecInit(axisNear).map(v => RegNext(v, 0.U))
  val farS0 = VecInit(axisFar).map(v => RegNext(v, 0.U))

  // Reduction level 1: tMin01=max(near0,near1), tMax01=min(far0,far1)
  val cmpNear01 = Module(new FCMP(cfg))
  cmpNear01.io.a := nearS0(0)
  cmpNear01.io.b := nearS0(1)
  cmpNear01.io.signaling := false.B
  val near01Le = cmpNear01.io.le
  val near00Cmp = pipeUInt(nearS0(0), cfg.fcmpLatency)
  val near01Cmp = pipeUInt(nearS0(1), cfg.fcmpLatency)
  val tMin01 = RegNext(Mux(near01Le, near01Cmp, near00Cmp), 0.U)

  val cmpFar01 = Module(new FCMP(cfg))
  cmpFar01.io.a := farS0(0)
  cmpFar01.io.b := farS0(1)
  cmpFar01.io.signaling := false.B
  val far01Le = cmpFar01.io.le
  val far00Cmp = pipeUInt(farS0(0), cfg.fcmpLatency)
  val far01Cmp = pipeUInt(farS0(1), cfg.fcmpLatency)
  val tMax01 = RegNext(Mux(far01Le, far00Cmp, far01Cmp), 0.U)

  val near2S1 = PipeUtils.pipeData(RegNext(nearS0(2), 0.U), cfg.fcmpLatency)
  val far2S1  = PipeUtils.pipeData(RegNext(farS0(2),  0.U), cfg.fcmpLatency)

  // Reduction level 2: tMin=max(tMin01,near2), tMax=min(tMax01,far2)
  val cmpNear012 = Module(new FCMP(cfg))
  cmpNear012.io.a := tMin01
  cmpNear012.io.b := near2S1
  cmpNear012.io.signaling := false.B
  val near012Le = cmpNear012.io.le
  val near2S1Cmp = pipeUInt(near2S1, cfg.fcmpLatency)
  val tMin01Cmp = pipeUInt(tMin01, cfg.fcmpLatency)
  val tMin = RegNext(Mux(near012Le, near2S1Cmp, tMin01Cmp), 0.U)

  val cmpFar012 = Module(new FCMP(cfg))
  cmpFar012.io.a := tMax01
  cmpFar012.io.b := far2S1
  cmpFar012.io.signaling := false.B
  val far012Le = cmpFar012.io.le
  val tMax01Cmp = pipeUInt(tMax01, cfg.fcmpLatency)
  val far2S1Cmp = pipeUInt(far2S1, cfg.fcmpLatency)
  val tMax = RegNext(Mux(far012Le, tMax01Cmp, far2S1Cmp), 0.U)

  // Hit: tMax >= tMin && tMax >= 0
  val cmpRange = Module(new FCMP(cfg))
  cmpRange.io.a := tMin
  cmpRange.io.b := tMax
  cmpRange.io.signaling := false.B
  val rangeOk = cmpRange.io.le

  val tMinFinal = pipeUInt(tMin, cfg.fcmpLatency)
  val tMaxFinal = pipeUInt(tMax, cfg.fcmpLatency)
  val tMaxNonNeg = !SimpleFPCompare.ltZero(tMaxFinal, cfg.totalWidth)

  // tNear = (tMin > 0) ? tMin : tMax
  val tMinPos = SimpleFPCompare.gtZero(tMinFinal, cfg.totalWidth)

  val tNearComb = Mux(tMinPos, tMinFinal, tMaxFinal)
  val hitComb = rangeOk && tMaxNonNeg

  io.tNear := RegNext(tNearComb, 0.U)
  io.tFar := RegNext(tMaxFinal, 0.U)
  io.hit := RegNext(hitComb, false.B)

  val totalLatency = 4 + cfg.faddLatency + cfg.fdivLatency + cfg.fmulLatency + (4 * cfg.fcmpLatency)
  io.out_valid := PipeUtils.pipeData(io.in_valid, totalLatency)
}
