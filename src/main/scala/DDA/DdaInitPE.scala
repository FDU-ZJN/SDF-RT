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
  private val totalInitLatency = relativeTMaxLatency + cfg.faddLatency
  private val coordWidth = GlobalConfig.ddaCoordWidth
  private val fpOne = java.lang.Float.floatToRawIntBits(1.0f).U(cfg.totalWidth.W)

  private def alignToTarget(x: UInt, pathLatency: Int, targetLatency: Int): UInt =
    pipeUInt(x, math.max(0, targetLatency - pathLatency))
  private def fpAbs(x: UInt): UInt = Cat(0.U(1.W), x(cfg.totalWidth - 2, 0))
  private def neg(x: UInt): UInt = Cat(!x(cfg.totalWidth - 1), x(cfg.totalWidth - 2, 0))

  io.in.ready := true.B

  val inFire = io.in.fire
  val reqAtOut = pipeData(io.in.bits, totalInitLatency)

  val stepNegX = io.in.bits.ray.dir.x(cfg.totalWidth - 1)
  val stepNegY = io.in.bits.ray.dir.y(cfg.totalWidth - 1)
  val stepNegZ = io.in.bits.ray.dir.z(cfg.totalWidth - 1)

  // Materialize the current traversal point from the immutable ray origin and accumulated distance.
  val posX = Module(new FFMA(cfg))
  val posY = Module(new FFMA(cfg))
  val posZ = Module(new FFMA(cfg))
  posX.io.a := io.in.bits.ray.dir.x
  posX.io.b := io.in.bits.ray.dist
  posX.io.c := io.in.bits.ray.origin.x
  posY.io.a := io.in.bits.ray.dir.y
  posY.io.b := io.in.bits.ray.dist
  posY.io.c := io.in.bits.ray.origin.y
  posZ.io.a := io.in.bits.ray.dir.z
  posZ.io.b := io.in.bits.ray.dist
  posZ.io.c := io.in.bits.ray.origin.z

  val subGx = Module(new FADD(cfg))
  val subGy = Module(new FADD(cfg))
  val subGz = Module(new FADD(cfg))
  subGx.io.a := posX.io.res
  subGx.io.b := neg(io.grid_min.x)
  subGy.io.a := posY.io.res
  subGy.io.b := neg(io.grid_min.y)
  subGz.io.a := posZ.io.res
  subGz.io.b := neg(io.grid_min.z)

  val mulIdxX = Module(new FMUL(cfg))
  val mulIdxY = Module(new FMUL(cfg))
  val mulIdxZ = Module(new FMUL(cfg))
  mulIdxX.io.a := subGx.io.res
  mulIdxX.io.b := io.inv_sub_voxel.x
  mulIdxY.io.a := subGy.io.res
  mulIdxY.io.b := io.inv_sub_voxel.y
  mulIdxZ.io.a := subGz.io.res
  mulIdxZ.io.b := io.inv_sub_voxel.z

  val fpToIntX = Module(new FPToInt(cfg.expWidth, cfg.precision, cfg.fptointLatency))
  val fpToIntY = Module(new FPToInt(cfg.expWidth, cfg.precision, cfg.fptointLatency))
  val fpToIntZ = Module(new FPToInt(cfg.expWidth, cfg.precision, cfg.fptointLatency))
  fpToIntX.io.a := mulIdxX.io.result
  fpToIntX.io.op := "b11".U
  fpToIntY.io.a := mulIdxY.io.result
  fpToIntY.io.op := "b11".U
  fpToIntZ.io.a := mulIdxZ.io.result
  fpToIntZ.io.op := "b11".U

  val mapXNeg = fpToIntX.io.result(31)
  val mapYNeg = fpToIntY.io.result(31)
  val mapZNeg = fpToIntZ.io.result(31)
  val mapXIdx = fpToIntX.io.result(addrWidth - 1, 0).asUInt
  val mapYIdx = fpToIntY.io.result(addrWidth - 1, 0).asUInt
  val mapZIdx = fpToIntZ.io.result(addrWidth - 1, 0).asUInt

  val idxToFpX = Module(new IntToFP(cfg.expWidth, cfg.precision))
  val idxToFpY = Module(new IntToFP(cfg.expWidth, cfg.precision))
  val idxToFpZ = Module(new IntToFP(cfg.expWidth, cfg.precision))
  idxToFpX.io.int := mapXIdx
  idxToFpY.io.int := mapYIdx
  idxToFpZ.io.int := mapZIdx

  val fracSubX = Module(new FADD(cfg))
  val fracSubY = Module(new FADD(cfg))
  val fracSubZ = Module(new FADD(cfg))
  val mulIdxXAtInt = pipeUInt(mulIdxX.io.result, cfg.fptointLatency)
  val mulIdxYAtInt = pipeUInt(mulIdxY.io.result, cfg.fptointLatency)
  val mulIdxZAtInt = pipeUInt(mulIdxZ.io.result, cfg.fptointLatency)
  fracSubX.io.a := mulIdxXAtInt
  fracSubX.io.b := neg(idxToFpX.io.result)
  fracSubY.io.a := mulIdxYAtInt
  fracSubY.io.b := neg(idxToFpY.io.result)
  fracSubZ.io.a := mulIdxZAtInt
  fracSubZ.io.b := neg(idxToFpZ.io.result)

  val oneMinusFracX = Module(new FADD(cfg))
  val oneMinusFracY = Module(new FADD(cfg))
  val oneMinusFracZ = Module(new FADD(cfg))
  oneMinusFracX.io.a := fpOne
  oneMinusFracX.io.b := neg(fracSubX.io.res)
  oneMinusFracY.io.a := fpOne
  oneMinusFracY.io.b := neg(fracSubY.io.res)
  oneMinusFracZ.io.a := fpOne
  oneMinusFracZ.io.b := neg(fracSubZ.io.res)

  val fracXAligned = pipeUInt(fracSubX.io.res, cfg.faddLatency)
  val fracYAligned = pipeUInt(fracSubY.io.res, cfg.faddLatency)
  val fracZAligned = pipeUInt(fracSubZ.io.res, cfg.faddLatency)
  val stepNegXAtDist = pipeBool(stepNegX, distSelectLatency)
  val stepNegYAtDist = pipeBool(stepNegY, distSelectLatency)
  val stepNegZAtDist = pipeBool(stepNegZ, distSelectLatency)
  val distX = Mux(stepNegXAtDist, fracXAligned, oneMinusFracX.io.res)
  val distY = Mux(stepNegYAtDist, fracYAligned, oneMinusFracY.io.res)
  val distZ = Mux(stepNegZAtDist, fracZAligned, oneMinusFracZ.io.res)

  val dsdtMulX = Module(new FMUL(cfg))
  val dsdtMulY = Module(new FMUL(cfg))
  val dsdtMulZ = Module(new FMUL(cfg))
  dsdtMulX.io.a := io.in.bits.ray.dir.x
  dsdtMulX.io.b := io.inv_sub_voxel.x
  dsdtMulY.io.a := io.in.bits.ray.dir.y
  dsdtMulY.io.b := io.inv_sub_voxel.y
  dsdtMulZ.io.a := io.in.bits.ray.dir.z
  dsdtMulZ.io.b := io.inv_sub_voxel.z

  val absDsdtX = fpAbs(dsdtMulX.io.result)
  val absDsdtY = fpAbs(dsdtMulY.io.result)
  val absDsdtZ = fpAbs(dsdtMulZ.io.result)

  val deltaDivX = Module(new FRQ(cfg))
  val deltaDivY = Module(new FRQ(cfg))
  val deltaDivZ = Module(new FRQ(cfg))
  deltaDivX.io.in := absDsdtX
  deltaDivX.io.in_valid := inFire
  deltaDivY.io.in := absDsdtY
  deltaDivY.io.in_valid := inFire
  deltaDivZ.io.in := absDsdtZ
  deltaDivZ.io.in_valid := inFire

  val distToMulX = alignToTarget(distX, distSelectLatency, tMaxInputLatency)
  val distToMulY = alignToTarget(distY, distSelectLatency, tMaxInputLatency)
  val distToMulZ = alignToTarget(distZ, distSelectLatency, tMaxInputLatency)
  val deltaToMulX = alignToTarget(deltaDivX.io.result, deltaLatency, tMaxInputLatency)
  val deltaToMulY = alignToTarget(deltaDivY.io.result, deltaLatency, tMaxInputLatency)
  val deltaToMulZ = alignToTarget(deltaDivZ.io.result, deltaLatency, tMaxInputLatency)

  val tMaxMulX = Module(new FMUL(cfg))
  val tMaxMulY = Module(new FMUL(cfg))
  val tMaxMulZ = Module(new FMUL(cfg))
  tMaxMulX.io.a := distToMulX
  tMaxMulX.io.b := deltaToMulX
  tMaxMulY.io.a := distToMulY
  tMaxMulY.io.b := deltaToMulY
  tMaxMulZ.io.a := distToMulZ
  tMaxMulZ.io.b := deltaToMulZ

  // Triangle intersections report t from the original ray origin. Convert the
  // cell-boundary distances, which are computed from the SDF hit position, to
  // that same absolute coordinate before DDA traversal starts.
  val rayBaseT = pipeUInt(io.in.bits.ray.dist, relativeTMaxLatency)
  val tMaxAbsX = Module(new FADD(cfg))
  val tMaxAbsY = Module(new FADD(cfg))
  val tMaxAbsZ = Module(new FADD(cfg))
  tMaxAbsX.io.a := tMaxMulX.io.result
  tMaxAbsX.io.b := rayBaseT
  tMaxAbsY.io.a := tMaxMulY.io.result
  tMaxAbsY.io.b := rayBaseT
  tMaxAbsZ.io.a := tMaxMulZ.io.result
  tMaxAbsZ.io.b := rayBaseT

  val tDeltaCapX = alignToTarget(deltaDivX.io.result, deltaLatency, totalInitLatency)
  val tDeltaCapY = alignToTarget(deltaDivY.io.result, deltaLatency, totalInitLatency)
  val tDeltaCapZ = alignToTarget(deltaDivZ.io.result, deltaLatency, totalInitLatency)

  val subXNegOut = pipeBool(mapXNeg, totalInitLatency - mapLatency)
  val subYNegOut = pipeBool(mapYNeg, totalInitLatency - mapLatency)
  val subZNegOut = pipeBool(mapZNeg, totalInitLatency - mapLatency)
  val subXIdxOut = pipeUInt(mapXIdx, totalInitLatency - mapLatency)
  val subYIdxOut = pipeUInt(mapYIdx, totalInitLatency - mapLatency)
  val subZIdxOut = pipeUInt(mapZIdx, totalInitLatency - mapLatency)
  val coordLimit = GlobalConfig.DdaRes.U(addrWidth.W)
  val subXClamped = Mux(subXIdxOut >= coordLimit, coordLimit, subXIdxOut)
  val subYClamped = Mux(subYIdxOut >= coordLimit, coordLimit, subYIdxOut)
  val subZClamped = Mux(subZIdxOut >= coordLimit, coordLimit, subZIdxOut)

  io.out.valid := pipeBool(inFire, totalInitLatency)
  io.out.bits := 0.U.asTypeOf(new BackendContextInit(cfg, addrWidth))
  io.out.bits.sourceFrontendId := reqAtOut.frontendContextId
  io.out.bits.contextId := reqAtOut.traceSlot
  io.out.bits.immutable.origin := reqAtOut.ray.origin
  io.out.bits.immutable.dir := reqAtOut.ray.dir
  io.out.bits.immutable.meta := reqAtOut.meta
  io.out.bits.traversal.contextId := reqAtOut.traceSlot
  io.out.bits.traversal.subX := Mux(subXNegOut, (-1).S(coordWidth.W), subXClamped(coordWidth - 2, 0).zext)
  io.out.bits.traversal.subY := Mux(subYNegOut, (-1).S(coordWidth.W), subYClamped(coordWidth - 2, 0).zext)
  io.out.bits.traversal.subZ := Mux(subZNegOut, (-1).S(coordWidth.W), subZClamped(coordWidth - 2, 0).zext)
  io.out.bits.traversal.iter := 0.U
  io.out.bits.traversal.tMaxX := tMaxAbsX.io.res
  io.out.bits.traversal.tMaxY := tMaxAbsY.io.res
  io.out.bits.traversal.tMaxZ := tMaxAbsZ.io.res
  io.out.bits.traversal.tDeltaX := tDeltaCapX
  io.out.bits.traversal.tDeltaY := tDeltaCapY
  io.out.bits.traversal.tDeltaZ := tDeltaCapZ
  io.out.bits.traversal.dirNegX := reqAtOut.ray.dir.x(cfg.totalWidth - 1)
  io.out.bits.traversal.dirNegY := reqAtOut.ray.dir.y(cfg.totalWidth - 1)
  io.out.bits.traversal.dirNegZ := reqAtOut.ray.dir.z(cfg.totalWidth - 1)

  when(io.out.valid) {
    assert(io.out.ready, "DdaInitPE expects io.out.ready to stay high in pipeline mode")
  }
}
