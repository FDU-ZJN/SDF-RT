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
    val out = Decoupled(new DdaContext(cfg, addrWidth))
  })

  private val positionLatency = cfg.fmulLatency + cfg.faddLatency
  private val mapLatency = positionLatency + cfg.faddLatency + cfg.fmulLatency + cfg.fptointLatency
  private val distSelectLatency = mapLatency + cfg.faddLatency + cfg.faddLatency
  private val deltaLatency = cfg.fmulLatency + cfg.fcmpLatency + cfg.fdivLatency
  private val tMaxInputLatency = math.max(distSelectLatency, deltaLatency)
  private val totalInitLatency = tMaxInputLatency + cfg.fmulLatency
  private val fpOne = java.lang.Float.floatToRawIntBits(1.0f).U(cfg.totalWidth.W)
  private val fpEps = java.lang.Float.floatToRawIntBits(1.0e-9f).U(cfg.totalWidth.W)

  private def alignToTarget(x: UInt, pathLatency: Int, targetLatency: Int): UInt =
    pipeUInt(x, math.max(0, targetLatency - pathLatency))
  private def fpAbs(x: UInt): UInt = Cat(0.U(1.W), x(cfg.totalWidth - 2, 0))
  private def neg(x: UInt): UInt = Cat(!x(cfg.totalWidth - 1), x(cfg.totalWidth - 2, 0))

  io.in.ready := true.B

  val inFire = io.in.fire
  val reqAtOut = pipeData(io.in.bits, totalInitLatency)

  val rdNegX = io.in.bits.ray.dir.x(cfg.totalWidth - 1)
  val rdNegY = io.in.bits.ray.dir.y(cfg.totalWidth - 1)
  val rdNegZ = io.in.bits.ray.dir.z(cfg.totalWidth - 1)
  val stepNegX = rdNegX ^ io.in.bits.reverseTraversal
  val stepNegY = rdNegY ^ io.in.bits.reverseTraversal
  val stepNegZ = rdNegZ ^ io.in.bits.reverseTraversal

  // Materialize the current traversal point from the immutable ray origin and accumulated distance.
  val posMulX = Module(new FMUL(cfg))
  val posMulY = Module(new FMUL(cfg))
  val posMulZ = Module(new FMUL(cfg))
  posMulX.io.a := io.in.bits.ray.dir.x
  posMulX.io.b := io.in.bits.ray.dist
  posMulX.io.rm := RNE
  posMulY.io.a := io.in.bits.ray.dir.y
  posMulY.io.b := io.in.bits.ray.dist
  posMulY.io.rm := RNE
  posMulZ.io.a := io.in.bits.ray.dir.z
  posMulZ.io.b := io.in.bits.ray.dist
  posMulZ.io.rm := RNE

  val originXAtPos = pipeUInt(io.in.bits.ray.origin.x, cfg.fmulLatency)
  val originYAtPos = pipeUInt(io.in.bits.ray.origin.y, cfg.fmulLatency)
  val originZAtPos = pipeUInt(io.in.bits.ray.origin.z, cfg.fmulLatency)

  val posAddX = Module(new FADD(cfg))
  val posAddY = Module(new FADD(cfg))
  val posAddZ = Module(new FADD(cfg))
  posAddX.io.a := originXAtPos
  posAddX.io.b := posMulX.io.result
  posAddX.io.rm := RNE
  posAddY.io.a := originYAtPos
  posAddY.io.b := posMulY.io.result
  posAddY.io.rm := RNE
  posAddZ.io.a := originZAtPos
  posAddZ.io.b := posMulZ.io.result
  posAddZ.io.rm := RNE

  val subGx = Module(new FADD(cfg))
  val subGy = Module(new FADD(cfg))
  val subGz = Module(new FADD(cfg))
  subGx.io.a := posAddX.io.res
  subGx.io.b := neg(io.grid_min.x)
  subGx.io.rm := RNE
  subGy.io.a := posAddY.io.res
  subGy.io.b := neg(io.grid_min.y)
  subGy.io.rm := RNE
  subGz.io.a := posAddZ.io.res
  subGz.io.b := neg(io.grid_min.z)
  subGz.io.rm := RNE

  val mulIdxX = Module(new FMUL(cfg))
  val mulIdxY = Module(new FMUL(cfg))
  val mulIdxZ = Module(new FMUL(cfg))
  mulIdxX.io.a := subGx.io.res
  mulIdxX.io.b := io.inv_sub_voxel.x
  mulIdxX.io.rm := RNE
  mulIdxY.io.a := subGy.io.res
  mulIdxY.io.b := io.inv_sub_voxel.y
  mulIdxY.io.rm := RNE
  mulIdxZ.io.a := subGz.io.res
  mulIdxZ.io.b := io.inv_sub_voxel.z
  mulIdxZ.io.rm := RNE

  val fpToIntX = Module(new FPToInt(cfg.expWidth, cfg.precision, cfg.fptointLatency))
  val fpToIntY = Module(new FPToInt(cfg.expWidth, cfg.precision, cfg.fptointLatency))
  val fpToIntZ = Module(new FPToInt(cfg.expWidth, cfg.precision, cfg.fptointLatency))
  fpToIntX.io.a := mulIdxX.io.result
  fpToIntX.io.rm := RTZ
  fpToIntX.io.op := "b11".U
  fpToIntY.io.a := mulIdxY.io.result
  fpToIntY.io.rm := RTZ
  fpToIntY.io.op := "b11".U
  fpToIntZ.io.a := mulIdxZ.io.result
  fpToIntZ.io.rm := RTZ
  fpToIntZ.io.op := "b11".U

  val mapXNeg = fpToIntX.io.result(63)
  val mapYNeg = fpToIntY.io.result(63)
  val mapZNeg = fpToIntZ.io.result(63)
  val mapXIdx = fpToIntX.io.result(addrWidth - 1, 0).asUInt
  val mapYIdx = fpToIntY.io.result(addrWidth - 1, 0).asUInt
  val mapZIdx = fpToIntZ.io.result(addrWidth - 1, 0).asUInt

  val idxToFpX = Module(new IntToFP(cfg.expWidth, cfg.precision))
  val idxToFpY = Module(new IntToFP(cfg.expWidth, cfg.precision))
  val idxToFpZ = Module(new IntToFP(cfg.expWidth, cfg.precision))
  idxToFpX.io.int := mapXIdx
  idxToFpX.io.sign := false.B
  idxToFpX.io.long := false.B
  idxToFpX.io.rm := RNE
  idxToFpY.io.int := mapYIdx
  idxToFpY.io.sign := false.B
  idxToFpY.io.long := false.B
  idxToFpY.io.rm := RNE
  idxToFpZ.io.int := mapZIdx
  idxToFpZ.io.sign := false.B
  idxToFpZ.io.long := false.B
  idxToFpZ.io.rm := RNE

  val fracSubX = Module(new FADD(cfg))
  val fracSubY = Module(new FADD(cfg))
  val fracSubZ = Module(new FADD(cfg))
  val mulIdxXAtInt = pipeUInt(mulIdxX.io.result, cfg.fptointLatency)
  val mulIdxYAtInt = pipeUInt(mulIdxY.io.result, cfg.fptointLatency)
  val mulIdxZAtInt = pipeUInt(mulIdxZ.io.result, cfg.fptointLatency)
  fracSubX.io.a := mulIdxXAtInt
  fracSubX.io.b := neg(idxToFpX.io.result)
  fracSubX.io.rm := RNE
  fracSubY.io.a := mulIdxYAtInt
  fracSubY.io.b := neg(idxToFpY.io.result)
  fracSubY.io.rm := RNE
  fracSubZ.io.a := mulIdxZAtInt
  fracSubZ.io.b := neg(idxToFpZ.io.result)
  fracSubZ.io.rm := RNE

  val oneMinusFracX = Module(new FADD(cfg))
  val oneMinusFracY = Module(new FADD(cfg))
  val oneMinusFracZ = Module(new FADD(cfg))
  oneMinusFracX.io.a := fpOne
  oneMinusFracX.io.b := neg(fracSubX.io.res)
  oneMinusFracX.io.rm := RNE
  oneMinusFracY.io.a := fpOne
  oneMinusFracY.io.b := neg(fracSubY.io.res)
  oneMinusFracY.io.rm := RNE
  oneMinusFracZ.io.a := fpOne
  oneMinusFracZ.io.b := neg(fracSubZ.io.res)
  oneMinusFracZ.io.rm := RNE

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
  dsdtMulX.io.rm := RNE
  dsdtMulY.io.a := io.in.bits.ray.dir.y
  dsdtMulY.io.b := io.inv_sub_voxel.y
  dsdtMulY.io.rm := RNE
  dsdtMulZ.io.a := io.in.bits.ray.dir.z
  dsdtMulZ.io.b := io.inv_sub_voxel.z
  dsdtMulZ.io.rm := RNE

  val absDsdtX = fpAbs(dsdtMulX.io.result)
  val absDsdtY = fpAbs(dsdtMulY.io.result)
  val absDsdtZ = fpAbs(dsdtMulZ.io.result)

  val cmpEpsX = Module(new FCMP(cfg))
  val cmpEpsY = Module(new FCMP(cfg))
  val cmpEpsZ = Module(new FCMP(cfg))
  cmpEpsX.io.a := absDsdtX
  cmpEpsX.io.b := fpEps
  cmpEpsX.io.signaling := false.B
  cmpEpsY.io.a := absDsdtY
  cmpEpsY.io.b := fpEps
  cmpEpsY.io.signaling := false.B
  cmpEpsZ.io.a := absDsdtZ
  cmpEpsZ.io.b := fpEps
  cmpEpsZ.io.signaling := false.B

  val absDsdtXAligned = pipeUInt(absDsdtX, cfg.fcmpLatency)
  val absDsdtYAligned = pipeUInt(absDsdtY, cfg.fcmpLatency)
  val absDsdtZAligned = pipeUInt(absDsdtZ, cfg.fcmpLatency)
  val denomX = Mux(cmpEpsX.io.le, fpEps, absDsdtXAligned)
  val denomY = Mux(cmpEpsY.io.le, fpEps, absDsdtYAligned)
  val denomZ = Mux(cmpEpsZ.io.le, fpEps, absDsdtZAligned)

  val deltaDivX = Module(new FRQ(cfg))
  val deltaDivY = Module(new FRQ(cfg))
  val deltaDivZ = Module(new FRQ(cfg))
  deltaDivX.io.in := denomX
  deltaDivX.io.in_valid := inFire
  deltaDivY.io.in := denomY
  deltaDivY.io.in_valid := inFire
  deltaDivZ.io.in := denomZ
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
  tMaxMulX.io.rm := RNE
  tMaxMulY.io.a := distToMulY
  tMaxMulY.io.b := deltaToMulY
  tMaxMulY.io.rm := RNE
  tMaxMulZ.io.a := distToMulZ
  tMaxMulZ.io.b := deltaToMulZ
  tMaxMulZ.io.rm := RNE

  val tDeltaCapX = alignToTarget(deltaDivX.io.result, deltaLatency, totalInitLatency)
  val tDeltaCapY = alignToTarget(deltaDivY.io.result, deltaLatency, totalInitLatency)
  val tDeltaCapZ = alignToTarget(deltaDivZ.io.result, deltaLatency, totalInitLatency)

  val subXNegOut = pipeBool(mapXNeg, totalInitLatency - mapLatency)
  val subYNegOut = pipeBool(mapYNeg, totalInitLatency - mapLatency)
  val subZNegOut = pipeBool(mapZNeg, totalInitLatency - mapLatency)
  val subXIdxOut = pipeUInt(mapXIdx, totalInitLatency - mapLatency)
  val subYIdxOut = pipeUInt(mapYIdx, totalInitLatency - mapLatency)
  val subZIdxOut = pipeUInt(mapZIdx, totalInitLatency - mapLatency)

  io.out.valid := pipeBool(inFire, totalInitLatency)
  io.out.bits := 0.U.asTypeOf(new DdaContext(cfg, addrWidth))
  io.out.bits.ray := reqAtOut.ray
  io.out.bits.meta := reqAtOut.meta
  io.out.bits.reverseTraversal := reqAtOut.reverseTraversal
  io.out.bits.traceSlot := reqAtOut.traceSlot
  io.out.bits.initialized := true.B
  io.out.bits.subX := Mux(subXNegOut, (-1).S((addrWidth + 1).W), subXIdxOut.zext)
  io.out.bits.subY := Mux(subYNegOut, (-1).S((addrWidth + 1).W), subYIdxOut.zext)
  io.out.bits.subZ := Mux(subZNegOut, (-1).S((addrWidth + 1).W), subZIdxOut.zext)
  io.out.bits.iter := 0.U
  io.out.bits.tMaxX := tMaxMulX.io.result
  io.out.bits.tMaxY := tMaxMulY.io.result
  io.out.bits.tMaxZ := tMaxMulZ.io.result
  io.out.bits.tDeltaX := tDeltaCapX
  io.out.bits.tDeltaY := tDeltaCapY
  io.out.bits.tDeltaZ := tDeltaCapZ

  when(io.out.valid) {
    assert(io.out.ready, "DdaInitPE expects io.out.ready to stay high in pipeline mode")
  }
}
