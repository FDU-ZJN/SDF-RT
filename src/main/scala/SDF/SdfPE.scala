package SDF

import chisel3._
import chisel3.util._
import raytrace_utils._
import raytrace_utils.fudian._
import raytrace_utils.PipeUtils._

class SdfPE(val c: SdfPeConfig = SdfPeConfig()) extends Module {
  val io = IO(new Bundle {
    val in = Flipped(Decoupled(new SdfStepReq(c.cfg)))
    val sdf_mem_req = Decoupled(new SdfMemReq(c.addrWidth))
    val sdf_mem_resp = Flipped(Decoupled(UInt(c.cfg.totalWidth.W)))

    // Grid mapping parameters: idx = floor((p - gridMin) * invVoxel)
    val grid_min = Input(new Vec3(c.cfg))
    val inv_voxel = Input(new Vec3(c.cfg))

    val out = Decoupled(new SdfStepResp(c.cfg))
  })

  // Only the mutable state and context ID cross the fixed-latency pipeline;
  // immutable origin/direction are consumed by the position stage above.
  class SdfPipeCtx extends Bundle {
    val contextId = UInt(GlobalConfig.frontendContextBits.W)
    val dist = UInt(c.cfg.totalWidth.W)
    val iter = UInt(16.W)
    val prevSdf = UInt(c.cfg.totalWidth.W)
    val inBounds = Bool()
    val valid = Bool()
  }

  val hitThreshold1 = BigInt(c.threshold1Bits & 0xffffffffL).U(c.cfg.totalWidth.W)
  val hitThreshold2 = BigInt(c.threshold2Bits & 0xffffffffL).U(c.cfg.totalWidth.W)
  val hitThreshold3 = BigInt(c.threshold3Bits & 0xffffffffL).U(c.cfg.totalWidth.W)
  val fpZero = 0.U(c.cfg.totalWidth.W)
  val maxStepsU = c.maxSteps.U(16.W)
  val halfStepsU = (c.maxSteps / 2).U(16.W)
  val threeQuarterStepsU = ((c.maxSteps * 3) / 4).U(16.W)

  val fullResXU = (c.GlobalResX * c.LocalResX).U(c.addrWidth.W)
  val fullResYU = (c.GlobalResY * c.LocalResY).U(c.addrWidth.W)
  val fullResZU = (c.GlobalResZ * c.LocalResZ).U(c.addrWidth.W)


  require((c.GlobalResX & (c.GlobalResX - 1)) == 0, s"GlobalResX must be power-of-two, got ${c.GlobalResX}")
  require((c.GlobalResY & (c.GlobalResY - 1)) == 0, s"GlobalResY must be power-of-two, got ${c.GlobalResY}")
  // Local grid resolution is expected to be power-of-two for shift/mask mapping.
  require((c.LocalResX & (c.LocalResX - 1)) == 0, s"LocalResX must be power-of-two, got ${c.LocalResX}")
  require((c.LocalResY & (c.LocalResY - 1)) == 0, s"LocalResY must be power-of-two, got ${c.LocalResY}")
  require((c.LocalResZ & (c.LocalResZ - 1)) == 0, s"LocalResZ must be power-of-two, got ${c.LocalResZ}")

  val globalShiftX = log2Ceil(c.GlobalResX)
  val globalShiftY = log2Ceil(c.GlobalResY)
  val globalPlaneShift = globalShiftX + globalShiftY

  val localShiftX = log2Ceil(c.LocalResX)
  val localShiftY = log2Ceil(c.LocalResY)
  val localShiftZ = log2Ceil(c.LocalResZ)
  val localPlaneShift = localShiftX + localShiftY
  val localMaskX = (c.LocalResX - 1).U(c.addrWidth.W)
  val localMaskY = (c.LocalResY - 1).U(c.addrWidth.W)
  val localMaskZ = (c.LocalResZ - 1).U(c.addrWidth.W)

  // Address path first materializes p = origin + dir * dist, then maps p into the SDF grid.
  val positionLatency = c.cfg.ffmaLatency
  val addrLatency = positionLatency + c.cfg.faddLatency + c.cfg.fmulLatency + c.cfg.fptointLatency
  val advanceLatency = c.cfg.faddLatency
  val totalStepLatency = addrLatency + GlobalConfig.sdfMemDpiLatency +
    c.cfg.fcmpLatency + advanceLatency

  def neg(x: UInt): UInt = Cat(!x(c.cfg.totalWidth - 1), x(c.cfg.totalWidth - 2, 0))

  def fpHalf(x: UInt): UInt = {
    val fracWidth = c.cfg.precision - 1
    val expHi = c.cfg.totalWidth - 2
    val expLo = fracWidth
    val sign = x(c.cfg.totalWidth - 1)
    val exp = x(expHi, expLo)
    val frac = x(fracWidth - 1, 0)
    val expMax = ((1 << c.cfg.expWidth) - 1).U(c.cfg.expWidth.W)
    val isSpecial = exp === expMax
    val isZeroOrSubnormal = exp === 0.U
    val becomesSubnormal = exp === 1.U
    val subnormalShift = Cat(0.U(1.W), frac) >> 1
    val normalizedToSub = Cat(1.U(1.W), frac) >> 1

    Mux(isSpecial, x,
      Mux(isZeroOrSubnormal,
        Cat(sign, 0.U(c.cfg.expWidth.W), subnormalShift(fracWidth - 1, 0)),
        Mux(becomesSubnormal,
          Cat(sign, 0.U(c.cfg.expWidth.W), normalizedToSub(fracWidth - 1, 0)),
          Cat(sign, (exp - 1.U)(c.cfg.expWidth - 1, 0), frac)
        )
      )
    )
  }

  // --------------------
  // Stage A: ray distance -> current position -> grid address pipeline
  // --------------------
  val posX = Module(new FFMA(c.cfg))
  val posY = Module(new FFMA(c.cfg))
  val posZ = Module(new FFMA(c.cfg))
  posX.io.a := io.in.bits.dir.x
  posX.io.b := io.in.bits.state.dist
  posX.io.c := io.in.bits.origin.x
  posY.io.a := io.in.bits.dir.y
  posY.io.b := io.in.bits.state.dist
  posY.io.c := io.in.bits.origin.y
  posZ.io.a := io.in.bits.dir.z
  posZ.io.b := io.in.bits.state.dist
  posZ.io.c := io.in.bits.origin.z

  val subGx = Module(new FADD(c.cfg))
  val subGy = Module(new FADD(c.cfg))
  val subGz = Module(new FADD(c.cfg))

  subGx.io.a := posX.io.res
  subGx.io.b := neg(io.grid_min.x)
  subGy.io.a := posY.io.res
  subGy.io.b := neg(io.grid_min.y)
  subGz.io.a := posZ.io.res
  subGz.io.b := neg(io.grid_min.z)

  val mulIdxX = Module(new FMUL(c.cfg))
  val mulIdxY = Module(new FMUL(c.cfg))
  val mulIdxZ = Module(new FMUL(c.cfg))

  mulIdxX.io.a := subGx.io.res
  mulIdxX.io.b := io.inv_voxel.x
  mulIdxY.io.a := subGy.io.res
  mulIdxY.io.b := io.inv_voxel.y
  mulIdxZ.io.a := subGz.io.res
  mulIdxZ.io.b := io.inv_voxel.z

  val fpToIntX = Module(new FPToInt(c.cfg.expWidth, c.cfg.precision, c.cfg.fptointLatency))
  val fpToIntY = Module(new FPToInt(c.cfg.expWidth, c.cfg.precision, c.cfg.fptointLatency))
  val fpToIntZ = Module(new FPToInt(c.cfg.expWidth, c.cfg.precision, c.cfg.fptointLatency))

  fpToIntX.io.a := mulIdxX.io.result
  fpToIntX.io.op := "b11".U
  fpToIntY.io.a := mulIdxY.io.result
  fpToIntY.io.op := "b11".U
  fpToIntZ.io.a := mulIdxZ.io.result
  fpToIntZ.io.op := "b11".U

  val xNeg = fpToIntX.io.result(31)
  val yNeg = fpToIntY.io.result(31)
  val zNeg = fpToIntZ.io.result(31)

  val xIdx = fpToIntX.io.result(c.addrWidth - 1, 0).asUInt
  val yIdx = fpToIntY.io.result(c.addrWidth - 1, 0).asUInt
  val zIdx = fpToIntZ.io.result(c.addrWidth - 1, 0).asUInt

  val addrInBounds = !xNeg && !yNeg && !zNeg &&
    (xIdx < fullResXU) && (yIdx < fullResYU) && (zIdx < fullResZU)

  val xGlobal = xIdx >> localShiftX
  val yGlobal = yIdx >> localShiftY
  val zGlobal = zIdx >> localShiftZ
  val xLocal = xIdx & localMaskX
  val yLocal = yIdx & localMaskY
  val zLocal = zIdx & localMaskZ

  val xGlobalWide = Wire(UInt((2 * c.addrWidth).W))
  val yGlobalScaled = Wire(UInt((2 * c.addrWidth).W))
  val zGlobalScaled = Wire(UInt((2 * c.addrWidth).W))
  val xLocalWide = Wire(UInt((2 * c.addrWidth).W))
  val yLocalScaled = Wire(UInt((2 * c.addrWidth).W))
  val zLocalScaled = Wire(UInt((2 * c.addrWidth).W))

  xGlobalWide := xGlobal
  yGlobalScaled := Cat(yGlobal, 0.U(globalShiftX.W)).asUInt
  zGlobalScaled := Cat(zGlobal, 0.U(globalPlaneShift.W)).asUInt
  xLocalWide := xLocal
  yLocalScaled := Cat(yLocal, 0.U(localShiftX.W)).asUInt
  zLocalScaled := Cat(zLocal, 0.U(localPlaneShift.W)).asUInt

  val globalLinearWide = xGlobalWide + yGlobalScaled + zGlobalScaled
  val localLinearWide = xLocalWide + yLocalScaled + zLocalScaled
  val globalLinear = globalLinearWide(c.addrWidth - 1, 0)
  val localLinear = localLinearWide(c.addrWidth - 1, 0)

  io.in.ready := true.B
  val inFire = io.in.fire
  val inCtx = Wire(new SdfPipeCtx)
  inCtx.contextId := io.in.bits.contextId
  inCtx.dist := io.in.bits.state.dist
  inCtx.iter := io.in.bits.state.iter
  inCtx.prevSdf := io.in.bits.state.prevSdf
  inCtx.inBounds := false.B
  inCtx.valid := inFire
  val addrCtxBase = pipeData(inCtx, addrLatency)
  val addrCtx = Wire(new SdfPipeCtx)
  addrCtx := addrCtxBase
  addrCtx.inBounds := addrInBounds
  val addrValid = addrCtx.valid


  val sdfMemLatency = GlobalConfig.sdfMemDpiLatency
  io.sdf_mem_req.valid := addrValid && addrInBounds
  io.sdf_mem_req.bits.globalIdx := globalLinear
  io.sdf_mem_req.bits.localIdx := localLinear
  when(io.sdf_mem_req.valid) {
    assert(io.sdf_mem_req.ready, "SdfPE expects sdf_mem_req.ready to stay high in pipeline mode")
  }

  val bCtx = pipeData(addrCtx, sdfMemLatency)
  val bValid = bCtx.valid
  val bInBounds = bCtx.inBounds

  io.sdf_mem_resp.ready := bValid && bInBounds
  when(bValid && bInBounds) {
    assert(io.sdf_mem_resp.valid, "SdfPE expects fixed-latency sdf_mem_resp.valid for in-bounds requests")
  }

  val bSample = Mux(bInBounds&&bValid, io.sdf_mem_resp.bits, fpZero)
  val bSampleNeg = bSample(c.cfg.totalWidth - 1)
  val bSampleAbs = Cat(0.U(1.W), bSample(c.cfg.totalWidth - 2, 0))

  // --------------------
  // Stage C: one-step march update
  // --------------------
  val currentHitThreshold = Wire(UInt(c.cfg.totalWidth.W))
  when(bCtx.iter < halfStepsU) {
    currentHitThreshold := hitThreshold1
  }.elsewhen(bCtx.iter < threeQuarterStepsU) {
    currentHitThreshold := hitThreshold2
  }.otherwise {
    currentHitThreshold := hitThreshold3
  }

  val cmpHit = Module(new FCMP(c.cfg))
  cmpHit.io.a := bSampleAbs
  cmpHit.io.b := currentHitThreshold
  cmpHit.io.signaling := false.B

  val cmpLatency = c.cfg.fcmpLatency
  val bCtxCmp = pipeData(bCtx, cmpLatency)
  val bValidCmp = bCtxCmp.valid
  val bInBoundsCmp = bCtxCmp.inBounds
  val bSampleCmp = pipeUInt(bSample, cmpLatency)
  val bSampleNegCmp = pipeBool(bSampleNeg, cmpLatency)
  // Positive samples use the normal threshold test.  A negative sample does
  // not hit early; it becomes a terminal hit only when this sample consumes
  // the final march step.
  val bSamplePositive = !bSampleNegCmp && (bSampleCmp =/= fpZero)
  val positiveHit = bSamplePositive && cmpHit.io.lt
  val finalStepNegativeHit = bSampleNegCmp && (bCtxCmp.iter + 1.U >= maxStepsU)
  val bHit = bInBoundsCmp && (positiveHit || finalStepNegativeHit)

  val prevSdfHalf = fpHalf(bCtxCmp.prevSdf)
  val negPrevSdfHalf = neg(prevSdfHalf)
  val sampleIsNeg = bSampleNegCmp

  val posDistAdd = Module(new FADD(c.cfg))
  posDistAdd.io.a := bCtxCmp.dist
  posDistAdd.io.b := bSampleCmp

  val negDistAdd = Module(new FADD(c.cfg))
  negDistAdd.io.a := bCtxCmp.dist
  negDistAdd.io.b := negPrevSdfHalf

  val nextPrevSdfRaw = Mux(sampleIsNeg, prevSdfHalf, bSampleCmp)
  val advanceCtx = pipeData(bCtxCmp, advanceLatency)
  val sampleIsNegAtOut = pipeBool(sampleIsNeg, advanceLatency)

  // Miss-only path uses prevSdf carry-forward with negative-sample backoff.
  val outValid = pipeBool(bValidCmp && !bHit, advanceLatency)
  val outInBounds = pipeBool(bInBoundsCmp && !bHit && bValidCmp, advanceLatency)
  val outIter = Mux(outInBounds, advanceCtx.iter + 1.U, maxStepsU)
  val outCurrDist = Mux(sampleIsNegAtOut, negDistAdd.io.res, posDistAdd.io.res)
  val outPrevSdf = pipeUInt(nextPrevSdfRaw, advanceLatency)

  val outDist = outCurrDist

  val hitOutValid = pipeBool(bValidCmp && bHit, advanceLatency)
  val hitOutIter = advanceCtx.iter + 1.U
  val hitStayDist = advanceCtx.dist
  // On a final-step negative hit, report the current crossing t.  The backoff
  // result is deliberately not committed because tracing terminates here.
  val hitOutDist = hitStayDist
  val hitOutPrevSdf = pipeUInt(nextPrevSdfRaw, advanceLatency)

  // --------------------
  // Stage D: output
  // --------------------
  io.out.valid := outValid || hitOutValid
  io.out.bits.contextId := advanceCtx.contextId
  io.out.bits.state.dist := Mux(hitOutValid, hitOutDist, outDist)
  io.out.bits.state.iter := Mux(hitOutValid, hitOutIter, outIter)
  io.out.bits.state.prevSdf := Mux(hitOutValid, hitOutPrevSdf, outPrevSdf)
  io.out.bits.hit := hitOutValid
  io.out.bits.done := hitOutValid || !outInBounds || outIter >= maxStepsU

  when(io.out.valid) {
    assert(io.out.ready, "SdfPE expects io.out.ready to stay high in pipeline mode")
  }
}
