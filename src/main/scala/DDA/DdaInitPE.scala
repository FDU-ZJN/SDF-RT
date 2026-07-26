package DDA

import chisel3._
import chisel3.util._
import raytrace_utils._
import raytrace_utils.fudian._
import raytrace_utils.PipeUtils._

class DdaInitPE(
  cfg: FloatConfig = FloatConfig.FP32,
  addrWidth: Int = 32
) extends Module {
  val io = IO(new Bundle {
    val in = Flipped(Decoupled(new DdaTraversalReq(cfg, addrWidth)))
    val grid_min = Input(new Vec3(cfg))
    val inv_sub_voxel = Input(new Vec3(cfg))
    val out = Decoupled(new BackendContextInit(cfg, addrWidth))
  })

  private val positionLatency = cfg.ffmaLatency
  private val mapLatency = positionLatency + cfg.faddLatency + cfg.fmulLatency + cfg.fptointLatency
  private val distSelectLatency = mapLatency + cfg.faddLatency + cfg.faddLatency
  private val deltaLatency = cfg.fmulLatency + cfg.fdivLatency
  private val tMaxInputLatency = math.max(distSelectLatency, deltaLatency)
  private val relativeTMaxLatency = tMaxInputLatency + cfg.fmulLatency
  private val axisLatency = relativeTMaxLatency + cfg.faddLatency
  private val requestLatency = axisLatency + 3
  private val coordWidth = GlobalConfig.ddaCoordWidth
  private val fpOne = java.lang.Float.floatToRawIntBits(1.0f).U(cfg.totalWidth.W)

  private def alignToTarget(x: UInt, pathLatency: Int, targetLatency: Int): UInt =
    pipeUInt(x, math.max(0, targetLatency - pathLatency))
  private def fpAbs(x: UInt): UInt = Cat(0.U(1.W), x(cfg.totalWidth - 2, 0))
  private def neg(x: UInt): UInt = Cat(!x(cfg.totalWidth - 1), x(cfg.totalWidth - 2, 0))
  private def selectAxis(v: Vec3, axis: UInt): UInt =
    Mux(axis === 0.U, v.x, Mux(axis === 1.U, v.y, v.z))

  // One ray occupies the axis issuer for three consecutive cycles.  The Z
  // issue cycle may also accept the next ray, which starts its X axis on the
  // following cycle and gives a sustained initiation interval of three.
  val heldReq = Reg(new DdaTraversalReq(cfg, addrWidth))
  val axisBusy = RegInit(false.B)
  val issueAxisReg = RegInit(0.U(2.W))
  val canAccept = !axisBusy || issueAxisReg === 2.U
  io.in.ready := canAccept
  val inFire = io.in.fire

  val issueValid = axisBusy
  val issueAxis = issueAxisReg
  val issueReq = heldReq

  when(!axisBusy) {
    when(inFire) {
      heldReq := io.in.bits
      axisBusy := true.B
      issueAxisReg := 0.U
    }
  }.elsewhen(issueAxisReg === 2.U) {
    when(inFire) {
      heldReq := io.in.bits
      axisBusy := true.B
      issueAxisReg := 0.U
    }.otherwise {
      axisBusy := false.B
      issueAxisReg := 0.U
    }
  }.otherwise {
    issueAxisReg := issueAxisReg + 1.U
  }

  val axisOrigin = selectAxis(issueReq.ray.origin, issueAxis)
  val axisDir = selectAxis(issueReq.ray.dir, issueAxis)
  val axisGridMin = selectAxis(io.grid_min, issueAxis)
  val axisInvSubVoxel = selectAxis(io.inv_sub_voxel, issueAxis)

  // Shared per-axis path: position and subgrid coordinate mapping.
  val position = Module(new FFMA(cfg))
  position.io.a := axisDir
  position.io.b := issueReq.ray.dist
  position.io.c := axisOrigin

  val subGridMin = Module(new FADD(cfg))
  subGridMin.io.a := position.io.res
  subGridMin.io.b := neg(pipeUInt(axisGridMin, positionLatency))

  val mulIdx = Module(new FMUL(cfg))
  mulIdx.io.a := subGridMin.io.res
  mulIdx.io.b := pipeUInt(axisInvSubVoxel, positionLatency + cfg.faddLatency)

  val fpToInt = Module(new FPToInt(cfg.expWidth, cfg.precision, cfg.fptointLatency))
  fpToInt.io.a := mulIdx.io.result
  fpToInt.io.op := "b11".U
  val mapNeg = fpToInt.io.result(31)
  val mapIdx = fpToInt.io.result(addrWidth - 1, 0).asUInt

  val idxToFp = Module(new IntToFP(cfg.expWidth, cfg.precision))
  idxToFp.io.int := mapIdx

  val fracSub = Module(new FADD(cfg))
  val mulIdxAtInt = pipeUInt(mulIdx.io.result, cfg.fptointLatency)
  fracSub.io.a := mulIdxAtInt
  fracSub.io.b := neg(idxToFp.io.result)

  val oneMinusFrac = Module(new FADD(cfg))
  oneMinusFrac.io.a := fpOne
  oneMinusFrac.io.b := neg(fracSub.io.res)

  val fracAligned = pipeUInt(fracSub.io.res, cfg.faddLatency)
  val stepNegAtDist = pipeBool(axisDir(cfg.totalWidth - 1), distSelectLatency)
  val dist = Mux(stepNegAtDist, fracAligned, oneMinusFrac.io.res)

  // Shared per-axis path: reciprocal cell-crossing distance.
  val dsdtMul = Module(new FMUL(cfg))
  dsdtMul.io.a := axisDir
  dsdtMul.io.b := axisInvSubVoxel

  val deltaDiv = Module(new FRQ(cfg))
  deltaDiv.io.in := fpAbs(dsdtMul.io.result)
  deltaDiv.io.in_valid := pipeBool(issueValid, cfg.fmulLatency)

  val distToMul = alignToTarget(dist, distSelectLatency, tMaxInputLatency)
  val deltaToMul = alignToTarget(deltaDiv.io.result, deltaLatency, tMaxInputLatency)

  val tMaxMul = Module(new FMUL(cfg))
  tMaxMul.io.a := distToMul
  tMaxMul.io.b := deltaToMul

  val rayBaseT = pipeUInt(issueReq.ray.dist, relativeTMaxLatency)
  val tMaxAbs = Module(new FADD(cfg))
  tMaxAbs.io.a := tMaxMul.io.result
  tMaxAbs.io.b := rayBaseT

  val tDeltaAtOut = alignToTarget(deltaDiv.io.result, deltaLatency, axisLatency)
  val mapNegAtOut = pipeBool(mapNeg, axisLatency - mapLatency)
  val mapIdxAtOut = pipeUInt(mapIdx, axisLatency - mapLatency)
  val dirNegAtOut = pipeBool(axisDir(cfg.totalWidth - 1), axisLatency)
  val axisAtOut = pipeUInt(issueAxis, axisLatency)
  val axisOutValid = pipeBool(issueValid, axisLatency, false.B)

  val coordLimit = GlobalConfig.DdaRes.U(addrWidth.W)
  val subClamped = Mux(mapIdxAtOut >= coordLimit, coordLimit, mapIdxAtOut)
  val subCoord = Mux(mapNegAtOut,
    (-1).S(coordWidth.W),
    subClamped(coordWidth - 2, 0).zext)

  class AxisResult extends Bundle {
    val sub = SInt(coordWidth.W)
    val tMax = UInt(cfg.totalWidth.W)
    val tDelta = UInt(cfg.totalWidth.W)
    val dirNeg = Bool()
  }

  val axisResult = Wire(new AxisResult)
  axisResult.sub := subCoord
  axisResult.tMax := tMaxAbs.io.res
  axisResult.tDelta := tDeltaAtOut
  axisResult.dirNeg := dirNegAtOut

  val xResult = Reg(new AxisResult)
  val yResult = Reg(new AxisResult)
  when(axisOutValid && axisAtOut === 0.U) {
    xResult := axisResult
  }
  when(axisOutValid && axisAtOut === 1.U) {
    yResult := axisResult
  }

  // The request accepted with X is aligned to the Z result.  Registering the
  // assembled context keeps the wide backend payload off the arithmetic path.
  val reqAtZ = pipeData(io.in.bits, requestLatency)
  val reqAtZValid = pipeBool(inFire, requestLatency, false.B)
  val complete = axisOutValid && axisAtOut === 2.U
  val outBits = Reg(new BackendContextInit(cfg, addrWidth))
  val outValid = RegInit(false.B)

  when(outValid && io.out.ready) {
    outValid := false.B
  }
  when(complete) {
    assert(reqAtZValid, "DDA init Z result lost its request metadata")
    assert(!outValid || io.out.ready, "DdaInitPE output holding register overflow")
    outBits := 0.U.asTypeOf(new BackendContextInit(cfg, addrWidth))
    outBits.sourceFrontendId := reqAtZ.frontendContextId
    outBits.contextId := reqAtZ.traceSlot
    outBits.immutable.origin := reqAtZ.ray.origin
    outBits.immutable.dir := reqAtZ.ray.dir
    outBits.immutable.meta := reqAtZ.meta
    outBits.traversal.contextId := reqAtZ.traceSlot
    outBits.traversal.subX := xResult.sub
    outBits.traversal.subY := yResult.sub
    outBits.traversal.subZ := axisResult.sub
    outBits.traversal.iter := 0.U
    outBits.traversal.tMaxX := xResult.tMax
    outBits.traversal.tMaxY := yResult.tMax
    outBits.traversal.tMaxZ := axisResult.tMax
    outBits.traversal.tDeltaX := xResult.tDelta
    outBits.traversal.tDeltaY := yResult.tDelta
    outBits.traversal.tDeltaZ := axisResult.tDelta
    outBits.traversal.dirNegX := xResult.dirNeg
    outBits.traversal.dirNegY := yResult.dirNeg
    outBits.traversal.dirNegZ := axisResult.dirNeg
    outValid := true.B
  }

  io.out.valid := outValid
  io.out.bits := outBits
  when(io.out.valid) {
    assert(io.out.ready, "DdaInitPE expects io.out.ready to stay high in pipeline mode")
  }
}
