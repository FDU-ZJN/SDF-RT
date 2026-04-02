package DDA

import Trace._
import chisel3._
import chisel3.util._
import raytrace_utils._
import raytrace_utils.fudian._

class DDA(
  cfg: FloatConfig = FloatConfig.FP32,
  addrWidth: Int = 32,
  globalRes: Int = 16,
  subRes: Int = 16,
  maxTraversalSteps: Int = 1024
) extends Module {
  require((globalRes & (globalRes - 1)) == 0, s"globalRes must be power-of-two, got $globalRes")
  require((subRes & (subRes - 1)) == 0, s"subRes must be power-of-two, got $subRes")

  val io = IO(new Bundle {
    val in = Flipped(Decoupled(new DdaTraversalReq(cfg, addrWidth)))
    // Float -> subgrid mapping parameters, same formula as SdfPE.
    val grid_min = Input(new Vec3(cfg))
    val inv_sub_voxel = Input(new Vec3(cfg))
    val out = Decoupled(new DdaTraversalResult(cfg, addrWidth))
  })

  private val subShift = log2Ceil(subRes)
  private val globalShift = log2Ceil(globalRes)
  private val globalPlaneShift = globalShift + globalShift
  private val subPlaneShift = subShift + subShift
  private val subMask = (subRes - 1).S((addrWidth + 1).W)
  private val totalSub = globalRes * subRes
  private val totalSubS = totalSub.S((addrWidth + 1).W)
  private val missT = "h7F7FFFFF".U(cfg.totalWidth.W)

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
  // Capture compare result after FCMP latency has elapsed: elapsed = (fcmp + fadd - 1) - stepWait.
  private val stepAxisCaptureWait = math.max(0, cfg.faddLatency - 1).U(stepWaitW.W)

  private val fpOne = java.lang.Float.floatToRawIntBits(1.0f).U(cfg.totalWidth.W)
  private val fpEps = java.lang.Float.floatToRawIntBits(1.0e-9f).U(cfg.totalWidth.W)

  private def align(x: UInt, n: Int): UInt = if (n > 0) ShiftRegister(x, n) else x
  // Align a value produced at `pathLatency` to the common `targetLatency` stage.
  private def alignToTarget(x: UInt, pathLatency: Int, targetLatency: Int): UInt = {
    align(x, math.max(0, targetLatency - pathLatency))
  }

  def fpAbs(x: UInt): UInt = Cat(0.U(1.W), x(cfg.totalWidth - 2, 0))
  def neg(x: UInt): UInt = Cat(!x(cfg.totalWidth - 1), x(cfg.totalWidth - 2, 0))

  val trace = Module(new TraceStage(TriPeConfig(cfg = cfg, addrWidth = addrWidth)))
  val subgridMem = Module(new SubgridMetaMemDPI(addrWidth))

  val sIdle :: sMapCoord :: sInitDdaWait :: sFetchMeta :: sWaitMeta :: sIssueTrace :: sWaitTrace :: sStep :: sStepApply :: sDone :: Nil = Enum(10)
  val state = RegInit(sIdle)

  val rayReg = Reg(new Ray(cfg))
  val metaReg = Reg(new RayMeta(addrWidth))
  val resultReg = Reg(new DdaTraversalResult(cfg, addrWidth))
  val reverseTraversalReg = RegInit(false.B)

  val subX = RegInit(0.S((addrWidth + 1).W))
  val subY = RegInit(0.S((addrWidth + 1).W))
  val subZ = RegInit(0.S((addrWidth + 1).W))
  val iter = RegInit(0.U(16.W))
  val mapWait = RegInit(0.U(mapWaitW.W))
  val ddaInitWait = RegInit(0.U(ddaInitWaitW.W))
  val stepWait = RegInit(0.U(stepWaitW.W))
  val stepAxis = RegInit(0.U(2.W))

  val tMaxX = RegInit(0.U(cfg.totalWidth.W))
  val tMaxY = RegInit(0.U(cfg.totalWidth.W))
  val tMaxZ = RegInit(0.U(cfg.totalWidth.W))
  val tDeltaX = RegInit(fpOne)
  val tDeltaY = RegInit(fpOne)
  val tDeltaZ = RegInit(fpOne)

  val triStartReg = Reg(UInt(addrWidth.W))
  val triCountReg = Reg(UInt(16.W))

  val inBounds = subX >= 0.S && subY >= 0.S && subZ >= 0.S &&
    subX < totalSubS && subY < totalSubS && subZ < totalSubS

  val globalX = (subX.asUInt >> subShift).pad(addrWidth)(addrWidth - 1, 0)
  val globalY = (subY.asUInt >> subShift).pad(addrWidth)(addrWidth - 1, 0)
  val globalZ = (subZ.asUInt >> subShift).pad(addrWidth)(addrWidth - 1, 0)
  val subCellX = (subX & subMask).asUInt.pad(addrWidth)(addrWidth - 1, 0)
  val subCellY = (subY & subMask).asUInt.pad(addrWidth)(addrWidth - 1, 0)
  val subCellZ = (subZ & subMask).asUInt.pad(addrWidth)(addrWidth - 1, 0)

  val globalYScaled = (globalY << globalShift).asUInt
  val globalZScaled = (globalZ << globalPlaneShift).asUInt
  val subYScaled = (subCellY << subShift).asUInt
  val subZScaled = (subCellZ << subPlaneShift).asUInt

  val globalLinear = globalX + globalYScaled + globalZScaled
  val subLinear = subCellX + subYScaled + subZScaled

  val rdNegX = rayReg.dir.x(cfg.totalWidth - 1)
  val rdNegY = rayReg.dir.y(cfg.totalWidth - 1)
  val rdNegZ = rayReg.dir.z(cfg.totalWidth - 1)
  val stepNegX = rdNegX ^ reverseTraversalReg
  val stepNegY = rdNegY ^ reverseTraversalReg
  val stepNegZ = rdNegZ ^ reverseTraversalReg

  val rdAbsX = fpAbs(rayReg.dir.x)
  val rdAbsY = fpAbs(rayReg.dir.y)
  val rdAbsZ = fpAbs(rayReg.dir.z)

  // Float origin -> integer subgrid index path.
  val subGx = Module(new FADD(cfg))
  val subGy = Module(new FADD(cfg))
  val subGz = Module(new FADD(cfg))
  subGx.io.a := rayReg.origin.x
  subGx.io.b := neg(io.grid_min.x)
  subGx.io.rm := RNE
  subGy.io.a := rayReg.origin.y
  subGy.io.b := neg(io.grid_min.y)
  subGy.io.rm := RNE
  subGz.io.a := rayReg.origin.z
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

  // DDA init path: frac(s) + delta_t + tMax (Amanatides & Woo style).
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

  val fracXAligned = align(fracSubX.io.res, cfg.faddLatency)
  val fracYAligned = align(fracSubY.io.res, cfg.faddLatency)
  val fracZAligned = align(fracSubZ.io.res, cfg.faddLatency)
  val distX = Mux(stepNegX, fracXAligned, oneMinusFracX.io.res)
  val distY = Mux(stepNegY, fracYAligned, oneMinusFracY.io.res)
  val distZ = Mux(stepNegZ, fracZAligned, oneMinusFracZ.io.res)

  val dsdtMulX = Module(new FMUL(cfg))
  val dsdtMulY = Module(new FMUL(cfg))
  val dsdtMulZ = Module(new FMUL(cfg))
  dsdtMulX.io.a := rayReg.dir.x
  dsdtMulX.io.b := io.inv_sub_voxel.x
  dsdtMulX.io.rm := RNE
  dsdtMulY.io.a := rayReg.dir.y
  dsdtMulY.io.b := io.inv_sub_voxel.y
  dsdtMulY.io.rm := RNE
  dsdtMulZ.io.a := rayReg.dir.z
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

  val absDsdtXAligned = align(absDsdtX, cfg.fcmpLatency)
  val absDsdtYAligned = align(absDsdtY, cfg.fcmpLatency)
  val absDsdtZAligned = align(absDsdtZ, cfg.fcmpLatency)
  val cmpEpsXLe = cmpEpsX.io.le
  val cmpEpsYLe = cmpEpsY.io.le
  val cmpEpsZLe = cmpEpsZ.io.le

  val denomX = Mux(cmpEpsXLe, fpEps, absDsdtXAligned)
  val denomY = Mux(cmpEpsYLe, fpEps, absDsdtYAligned)
  val denomZ = Mux(cmpEpsZLe, fpEps, absDsdtZAligned)

  val deltaDivX = Module(new FDIV(cfg))
  val deltaDivY = Module(new FDIV(cfg))
  val deltaDivZ = Module(new FDIV(cfg))
  deltaDivX.io.a := fpOne
  deltaDivX.io.b := denomX
  deltaDivX.io.in_valid := true.B
  deltaDivY.io.a := fpOne
  deltaDivY.io.b := denomY
  deltaDivY.io.in_valid := true.B
  deltaDivZ.io.a := fpOne
  deltaDivZ.io.b := denomZ
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
  cmpXY.io.a := tMaxX
  cmpXY.io.b := tMaxY
  cmpXY.io.signaling := false.B
  cmpXZ.io.a := tMaxX
  cmpXZ.io.b := tMaxZ
  cmpXZ.io.signaling := false.B
  cmpYZ.io.a := tMaxY
  cmpYZ.io.b := tMaxZ
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
  addTMaxX.io.a := tMaxX
  addTMaxX.io.b := tDeltaX
  addTMaxX.io.rm := RNE
  addTMaxY.io.a := tMaxY
  addTMaxY.io.b := tDeltaY
  addTMaxY.io.rm := RNE
  addTMaxZ.io.a := tMaxZ
  addTMaxZ.io.b := tDeltaZ
  addTMaxZ.io.rm := RNE

  io.in.ready := state === sIdle

  subgridMem.io.clk := clock
  subgridMem.io.reset := reset
  subgridMem.io.globalIdx := globalLinear
  subgridMem.io.subIdx := subLinear
  subgridMem.io.en := state === sFetchMeta

  trace.io.issue_in.valid := false.B
  trace.io.issue_in.bits.ray := rayReg
  trace.io.issue_in.bits.meta := metaReg
  trace.io.tri_batch_in.valid := false.B
  trace.io.tri_batch_in.bits.base_addr := triStartReg
  trace.io.tri_batch_in.bits.count := triCountReg
  trace.io.end_exec := false.B
  trace.io.result_out.ready := true.B

  io.out.valid := state === sDone
  io.out.bits := resultReg

  switch(state) {
    is(sIdle) {
      when(io.in.fire) {
        rayReg := io.in.bits.ray
        metaReg := io.in.bits.meta
        reverseTraversalReg := io.in.bits.reverseTraversal
        iter := 0.U
        mapWait := mapInit+1.U
        state := sMapCoord
      }
    }

    is(sMapCoord) {
      when(mapWait === 0.U) {
        subX := Mux(mapXNeg, (-1).S((addrWidth + 1).W), mapXIdx.zext)
        subY := Mux(mapYNeg, (-1).S((addrWidth + 1).W), mapYIdx.zext)
        subZ := Mux(mapZNeg, (-1).S((addrWidth + 1).W), mapZIdx.zext)
        ddaInitWait := ddaInitWaitInit
        state := sInitDdaWait
      }.otherwise {
        mapWait := mapWait - 1.U
      }
    }

    is(sInitDdaWait) {
      when(ddaInitWait === 0.U) {
        tMaxX := tMaxMulX.io.result
        tMaxY := tMaxMulY.io.result
        tMaxZ := tMaxMulZ.io.result
        tDeltaX := tDeltaCapX
        tDeltaY := tDeltaCapY
        tDeltaZ := tDeltaCapZ
        state := sFetchMeta
      }.otherwise {
        ddaInitWait := ddaInitWait - 1.U
      }
    }

    is(sFetchMeta) {
      when(!inBounds || iter >= maxTraversalSteps.U) {
        resultReg.meta := metaReg
        resultReg.hit := false.B
        resultReg.hitId := 0.U
        resultReg.hitT := missT
        state := sDone
      }.otherwise {
        state := sWaitMeta
      }
    }

    is(sWaitMeta) {
      when(subgridMem.io.valid) {
        triStartReg := subgridMem.io.triStart
        triCountReg := subgridMem.io.triCount
        when(subgridMem.io.triCount === 0.U) {
          state := sStep
        }.otherwise {
          state := sIssueTrace
        }
      }
    }

    is(sIssueTrace) {
      trace.io.issue_in.valid := true.B
      trace.io.tri_batch_in.valid := true.B
      trace.io.end_exec := trace.io.issue_in.ready && trace.io.tri_batch_in.ready
      when(trace.io.issue_in.ready && trace.io.tri_batch_in.ready) {
        state := sWaitTrace
      }
    }

    is(sWaitTrace) {
      when(trace.io.result_out.valid) {
        when(trace.io.result_out.bits.hit) {
          resultReg.meta := trace.io.result_out.bits.meta
          resultReg.hit := true.B
          resultReg.hitId := trace.io.result_out.bits.hitId
          resultReg.hitT := trace.io.result_out.bits.hitT
          state := sDone
        }.otherwise {
          state := sStep
        }
      }
    }

    is(sStep) {
      // Seed stepAxis for zero-latency compare configs.
      stepAxis := nextAxis
      stepWait := stepWaitInit
      state := sStepApply
    }

    is(sStepApply) {
      // Re-sample axis once FCMP latency has elapsed for current tMax.
      when(cfg.fcmpLatency.U =/= 0.U && stepWait === stepAxisCaptureWait) {
        stepAxis := nextAxis
      }
      when(stepWait === 0.U) {
        when(stepAxis === 0.U) {
          subX := subX + Mux(stepNegX, -1.S, 1.S)
          tMaxX := addTMaxX.io.res
        }.elsewhen(stepAxis === 1.U) {
          subY := subY + Mux(stepNegY, -1.S, 1.S)
          tMaxY := addTMaxY.io.res
        }.otherwise {
          subZ := subZ + Mux(stepNegZ, -1.S, 1.S)
          tMaxZ := addTMaxZ.io.res
        }
        iter := iter + 1.U
        state := sFetchMeta
      }.otherwise {
        stepWait := stepWait - 1.U
      }
    }

    is(sDone) {
      when(io.out.ready) {
        state := sIdle
      }
    }
  }
}
