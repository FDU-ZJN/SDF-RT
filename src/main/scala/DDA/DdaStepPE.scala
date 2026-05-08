package DDA

import chisel3._
import chisel3.util._
import raytrace_utils._
import raytrace_utils.fudian._
import raytrace_utils.PipeUtils._

class DdaStepPE(
  cfg: FloatConfig = FloatConfig.FP32,
  addrWidth: Int = 32,
  globalRes: Int = 16,
  subRes: Int = 16,
  maxTraversalSteps: Int = 1024
) extends Module {
  require((globalRes & (globalRes - 1)) == 0, s"globalRes must be power-of-two, got $globalRes")
  require((subRes & (subRes - 1)) == 0, s"subRes must be power-of-two, got $subRes")

  val io = IO(new Bundle {
    val in = Flipped(Decoupled(new DdaContext(cfg, addrWidth)))
    val grid_min = Input(new Vec3(cfg))
    val inv_sub_voxel = Input(new Vec3(cfg))
    val out = Decoupled(new DdaStepResult(cfg, addrWidth))
  })

  private val subShift = log2Ceil(subRes)
  private val globalShift = log2Ceil(globalRes)
  private val globalPlaneShift = globalShift + globalShift
  private val subPlaneShift = subShift + subShift
  private val subMask = (subRes - 1).S((addrWidth + 1).W)
  private val totalSub = globalRes * subRes
  private val totalSubS = totalSub.S((addrWidth + 1).W)
  private val mapLatency = cfg.faddLatency + cfg.fmulLatency + cfg.fptointLatency
  private val mapWaitW = math.max(1, log2Ceil(mapLatency + 1))
  private val mapInit = (if (mapLatency > 0) mapLatency - 1 else 0).U(mapWaitW.W)
  private val ddaDistLatency = cfg.faddLatency + cfg.faddLatency
  private val ddaDeltaLatency = cfg.fmulLatency + cfg.fcmpLatency + cfg.fdivLatency
  private val ddaAlignLatency = math.max(ddaDistLatency, ddaDeltaLatency)
  private val ddaInitLatency = ddaAlignLatency + cfg.fmulLatency
  private val ddaInitWaitW = math.max(1, log2Ceil(ddaInitLatency + 1))
  private val ddaInitWaitInit = (if (ddaInitLatency > 0) ddaInitLatency - 1 else 0).U(ddaInitWaitW.W)
  private val stepAddLatency = cfg.fcmpLatency + cfg.faddLatency
  private val stepWaitW = math.max(1, log2Ceil(stepAddLatency + 1))
  private val stepWaitInit = (if (stepAddLatency > 0) stepAddLatency - 1 else 0).U(stepWaitW.W)
  private val stepAxisCaptureWait = math.max(0, cfg.faddLatency - 1).U(stepWaitW.W)
  private val fpOne = java.lang.Float.floatToRawIntBits(1.0f).U(cfg.totalWidth.W)
  private val fpEps = java.lang.Float.floatToRawIntBits(1.0e-9f).U(cfg.totalWidth.W)

  private def alignToTarget(x: UInt, pathLatency: Int, targetLatency: Int): UInt =
    pipeUInt(x, math.max(0, targetLatency - pathLatency))
  private def fpAbs(x: UInt): UInt = Cat(0.U(1.W), x(cfg.totalWidth - 2, 0))
  private def neg(x: UInt): UInt = Cat(!x(cfg.totalWidth - 1), x(cfg.totalWidth - 2, 0))

  val subgridMem = Module(new SubgridMetaMemDPI(addrWidth, latency = GlobalConfig.subgridMemDpiLatency))

  val sIdle :: sMapCoord :: sInitDdaWait :: sFetchMeta :: sWaitMeta :: sStep :: sStepApply :: sOut :: Nil = Enum(8)
  val state = RegInit(sIdle)

  val ctxReg = RegInit(0.U.asTypeOf(new DdaContext(cfg, addrWidth)))
  val outReg = RegInit(0.U.asTypeOf(new DdaStepResult(cfg, addrWidth)))
  val mapWait = RegInit(0.U(mapWaitW.W))
  val ddaInitWait = RegInit(0.U(ddaInitWaitW.W))
  val stepWait = RegInit(0.U(stepWaitW.W))
  val stepAxis = RegInit(0.U(2.W))

  val inBounds = ctxReg.subX >= 0.S && ctxReg.subY >= 0.S && ctxReg.subZ >= 0.S &&
    ctxReg.subX < totalSubS && ctxReg.subY < totalSubS && ctxReg.subZ < totalSubS

  val globalX = (ctxReg.subX.asUInt >> subShift).pad(addrWidth)(addrWidth - 1, 0)
  val globalY = (ctxReg.subY.asUInt >> subShift).pad(addrWidth)(addrWidth - 1, 0)
  val globalZ = (ctxReg.subZ.asUInt >> subShift).pad(addrWidth)(addrWidth - 1, 0)
  val subCellX = (ctxReg.subX & subMask).asUInt.pad(addrWidth)(addrWidth - 1, 0)
  val subCellY = (ctxReg.subY & subMask).asUInt.pad(addrWidth)(addrWidth - 1, 0)
  val subCellZ = (ctxReg.subZ & subMask).asUInt.pad(addrWidth)(addrWidth - 1, 0)
  val globalYScaled = (globalY << globalShift).asUInt
  val globalZScaled = (globalZ << globalPlaneShift).asUInt
  val subYScaled = (subCellY << subShift).asUInt
  val subZScaled = (subCellZ << subPlaneShift).asUInt
  val globalLinear = globalX + globalYScaled + globalZScaled
  val subLinear = subCellX + subYScaled + subZScaled

  val stepNegX = ctxReg.ray.dir.x(cfg.totalWidth - 1)
  val stepNegY = ctxReg.ray.dir.y(cfg.totalWidth - 1)
  val stepNegZ = ctxReg.ray.dir.z(cfg.totalWidth - 1)

  val subGx = Module(new FADD(cfg))
  val subGy = Module(new FADD(cfg))
  val subGz = Module(new FADD(cfg))
  subGx.io.a := ctxReg.ray.origin.x
  subGx.io.b := neg(io.grid_min.x)
  subGx.io.rm := RNE
  subGy.io.a := ctxReg.ray.origin.y
  subGy.io.b := neg(io.grid_min.y)
  subGy.io.rm := RNE
  subGz.io.a := ctxReg.ray.origin.z
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
  fracSubX.io.a := mulIdxX.io.result
  fracSubX.io.b := neg(idxToFpX.io.result)
  fracSubX.io.rm := RNE
  fracSubY.io.a := mulIdxY.io.result
  fracSubY.io.b := neg(idxToFpY.io.result)
  fracSubY.io.rm := RNE
  fracSubZ.io.a := mulIdxZ.io.result
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
  val distX = Mux(stepNegX, fracXAligned, oneMinusFracX.io.res)
  val distY = Mux(stepNegY, fracYAligned, oneMinusFracY.io.res)
  val distZ = Mux(stepNegZ, fracZAligned, oneMinusFracZ.io.res)

  val dsdtMulX = Module(new FMUL(cfg))
  val dsdtMulY = Module(new FMUL(cfg))
  val dsdtMulZ = Module(new FMUL(cfg))
  dsdtMulX.io.a := ctxReg.ray.dir.x
  dsdtMulX.io.b := io.inv_sub_voxel.x
  dsdtMulX.io.rm := RNE
  dsdtMulY.io.a := ctxReg.ray.dir.y
  dsdtMulY.io.b := io.inv_sub_voxel.y
  dsdtMulY.io.rm := RNE
  dsdtMulZ.io.a := ctxReg.ray.dir.z
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
  deltaDivX.io.in_valid := true.B
  deltaDivY.io.in := denomY
  deltaDivY.io.in_valid := true.B
  deltaDivZ.io.in := denomZ
  deltaDivZ.io.in_valid := true.B

  val distToMulX = alignToTarget(distX, ddaDistLatency, ddaAlignLatency)
  val distToMulY = alignToTarget(distY, ddaDistLatency, ddaAlignLatency)
  val distToMulZ = alignToTarget(distZ, ddaDistLatency, ddaAlignLatency)
  val deltaToMulX = alignToTarget(deltaDivX.io.result, ddaDeltaLatency, ddaAlignLatency)
  val deltaToMulY = alignToTarget(deltaDivY.io.result, ddaDeltaLatency, ddaAlignLatency)
  val deltaToMulZ = alignToTarget(deltaDivZ.io.result, ddaDeltaLatency, ddaAlignLatency)

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

  val tDeltaCapX = alignToTarget(deltaDivX.io.result, ddaDeltaLatency, ddaInitLatency)
  val tDeltaCapY = alignToTarget(deltaDivY.io.result, ddaDeltaLatency, ddaInitLatency)
  val tDeltaCapZ = alignToTarget(deltaDivZ.io.result, ddaDeltaLatency, ddaInitLatency)

  val cmpXY = Module(new FCMP(cfg))
  val cmpXZ = Module(new FCMP(cfg))
  val cmpYZ = Module(new FCMP(cfg))
  cmpXY.io.a := ctxReg.tMaxX
  cmpXY.io.b := ctxReg.tMaxY
  cmpXY.io.signaling := false.B
  cmpXZ.io.a := ctxReg.tMaxX
  cmpXZ.io.b := ctxReg.tMaxZ
  cmpXZ.io.signaling := false.B
  cmpYZ.io.a := ctxReg.tMaxY
  cmpYZ.io.b := ctxReg.tMaxZ
  cmpYZ.io.signaling := false.B

  val nextAxis = Wire(UInt(2.W))
  when(cmpXY.io.lt) {
    nextAxis := Mux(cmpXZ.io.lt, 0.U, 2.U)
  }.otherwise {
    nextAxis := Mux(cmpYZ.io.lt, 1.U, 2.U)
  }

  val addTMaxX = Module(new FADD(cfg))
  val addTMaxY = Module(new FADD(cfg))
  val addTMaxZ = Module(new FADD(cfg))
  addTMaxX.io.a := ctxReg.tMaxX
  addTMaxX.io.b := ctxReg.tDeltaX
  addTMaxX.io.rm := RNE
  addTMaxY.io.a := ctxReg.tMaxY
  addTMaxY.io.b := ctxReg.tDeltaY
  addTMaxY.io.rm := RNE
  addTMaxZ.io.a := ctxReg.tMaxZ
  addTMaxZ.io.b := ctxReg.tDeltaZ
  addTMaxZ.io.rm := RNE

  io.in.ready := state === sIdle
  io.out.valid := state === sOut
  io.out.bits := outReg

  subgridMem.io.clk := clock
  subgridMem.io.reset := reset
  subgridMem.io.globalIdx := globalLinear
  subgridMem.io.subIdx := subLinear
  subgridMem.io.en := state === sFetchMeta

  switch(state) {
    is(sIdle) {
      when(io.in.fire) {
        ctxReg := io.in.bits
        when(io.in.bits.initialized) {
          state := sFetchMeta
        }.otherwise {
          mapWait := mapInit + 1.U
          state := sMapCoord
        }
      }
    }

    is(sMapCoord) {
      when(mapWait === 0.U) {
        ctxReg.subX := Mux(mapXNeg, (-1).S((addrWidth + 1).W), mapXIdx.zext)
        ctxReg.subY := Mux(mapYNeg, (-1).S((addrWidth + 1).W), mapYIdx.zext)
        ctxReg.subZ := Mux(mapZNeg, (-1).S((addrWidth + 1).W), mapZIdx.zext)
        ctxReg.iter := 0.U
        ctxReg.initialized := true.B
        ddaInitWait := ddaInitWaitInit
        state := sInitDdaWait
      }.otherwise {
        mapWait := mapWait - 1.U
      }
    }

    is(sInitDdaWait) {
      when(ddaInitWait === 0.U) {
        ctxReg.tMaxX := tMaxMulX.io.result
        ctxReg.tMaxY := tMaxMulY.io.result
        ctxReg.tMaxZ := tMaxMulZ.io.result
        ctxReg.tDeltaX := tDeltaCapX
        ctxReg.tDeltaY := tDeltaCapY
        ctxReg.tDeltaZ := tDeltaCapZ
        state := sFetchMeta
      }.otherwise {
        ddaInitWait := ddaInitWait - 1.U
      }
    }

    is(sFetchMeta) {
      when(!inBounds || ctxReg.iter >= maxTraversalSteps.U) {
        outReg.ctx := ctxReg
        outReg.done := true.B
        outReg.emitCmd := false.B
        outReg.tri := 0.U.asTypeOf(new TriBatch(addrWidth))
        state := sOut
      }.otherwise {
        state := sWaitMeta
      }
    }

    is(sWaitMeta) {
      when(subgridMem.io.valid) {
        outReg.tri.base_addr := subgridMem.io.triStart
        outReg.tri.count := subgridMem.io.triCount
        outReg.emitCmd := subgridMem.io.triCount =/= 0.U
        stepAxis := nextAxis
        stepWait := stepWaitInit
        state := sStep
      }
    }

    is(sStep) {
      state := sStepApply
    }

    is(sStepApply) {
      when(cfg.fcmpLatency.U =/= 0.U && stepWait === stepAxisCaptureWait) {
        stepAxis := nextAxis
      }
      when(stepWait === 0.U) {
        val nextCtx = Wire(new DdaContext(cfg, addrWidth))
        nextCtx := ctxReg
        when(stepAxis === 0.U) {
          nextCtx.subX := ctxReg.subX + Mux(stepNegX, -1.S, 1.S)
          nextCtx.tMaxX := addTMaxX.io.res
        }.elsewhen(stepAxis === 1.U) {
          nextCtx.subY := ctxReg.subY + Mux(stepNegY, -1.S, 1.S)
          nextCtx.tMaxY := addTMaxY.io.res
        }.otherwise {
          nextCtx.subZ := ctxReg.subZ + Mux(stepNegZ, -1.S, 1.S)
          nextCtx.tMaxZ := addTMaxZ.io.res
        }
        nextCtx.iter := ctxReg.iter + 1.U
        ctxReg := nextCtx
        outReg.ctx := nextCtx
        outReg.done := false.B
        state := sOut
      }.otherwise {
        stepWait := stepWait - 1.U
      }
    }

    is(sOut) {
      when(io.out.ready) {
        state := sIdle
      }
    }
  }
}
