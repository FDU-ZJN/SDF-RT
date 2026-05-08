package SDF

import chisel3._
import chisel3.util._
import raytrace_utils._
import raytrace_utils.fudian._
import raytrace_utils.PipeUtils._

class SdfPE(val c: SdfPeConfig = SdfPeConfig()) extends Module {
  val io = IO(new Bundle {
    val in = Flipped(Decoupled(new SdfRayReq(c.cfg, c.addrWidth)))
    val sdf_mem_req = Decoupled(new SdfMemReq(c.addrWidth))
    val sdf_mem_resp = Flipped(Decoupled(UInt(c.cfg.totalWidth.W)))

    // Grid mapping parameters: idx = floor((p - gridMin) * invVoxel)
    val grid_min = Input(new Vec3(c.cfg))
    val inv_voxel = Input(new Vec3(c.cfg))

    // Miss-only output (continue/terminal miss path)
    val out = Decoupled(new SdfRayResp(c.cfg, c.addrWidth))
    // Hit output (terminal hit path)
    val out_hit = Decoupled(new SdfRayResp(c.cfg, c.addrWidth))
  })

  val rmRne = RNE
  val rmRtz = RTZ

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
  val positionLatency = c.cfg.fmulLatency + c.cfg.faddLatency
  val addrLatency = positionLatency + c.cfg.faddLatency + c.cfg.fmulLatency + c.cfg.fptointLatency
  val advanceLatency = c.cfg.faddLatency

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

  def sampleThenDelay(x: UInt, en: Bool, latency: Int): UInt = {
    val sampled = RegInit(0.U(c.cfg.totalWidth.W))
    when(en) {
      sampled := x
    }
    if (latency <= 1) sampled else pipeUInt(sampled, latency - 1)
  }

  // --------------------
  // Stage A: ray distance -> current position -> grid address pipeline
  // --------------------
  val posMulX = Module(new FMUL(c.cfg))
  val posMulY = Module(new FMUL(c.cfg))
  val posMulZ = Module(new FMUL(c.cfg))
  posMulX.io.a := io.in.bits.ray.dir.x
  posMulX.io.b := io.in.bits.ray.dist
  posMulX.io.rm := rmRne
  posMulY.io.a := io.in.bits.ray.dir.y
  posMulY.io.b := io.in.bits.ray.dist
  posMulY.io.rm := rmRne
  posMulZ.io.a := io.in.bits.ray.dir.z
  posMulZ.io.b := io.in.bits.ray.dist
  posMulZ.io.rm := rmRne

  val originXAtPos = pipeUInt(io.in.bits.ray.origin.x, c.cfg.fmulLatency)
  val originYAtPos = pipeUInt(io.in.bits.ray.origin.y, c.cfg.fmulLatency)
  val originZAtPos = pipeUInt(io.in.bits.ray.origin.z, c.cfg.fmulLatency)

  val posAddX = Module(new FADD(c.cfg))
  val posAddY = Module(new FADD(c.cfg))
  val posAddZ = Module(new FADD(c.cfg))
  posAddX.io.a := originXAtPos
  posAddX.io.b := posMulX.io.result
  posAddX.io.rm := rmRne
  posAddY.io.a := originYAtPos
  posAddY.io.b := posMulY.io.result
  posAddY.io.rm := rmRne
  posAddZ.io.a := originZAtPos
  posAddZ.io.b := posMulZ.io.result
  posAddZ.io.rm := rmRne

  val subGx = Module(new FADD(c.cfg))
  val subGy = Module(new FADD(c.cfg))
  val subGz = Module(new FADD(c.cfg))

  subGx.io.a := posAddX.io.res
  subGx.io.b := neg(io.grid_min.x)
  subGx.io.rm := rmRne
  subGy.io.a := posAddY.io.res
  subGy.io.b := neg(io.grid_min.y)
  subGy.io.rm := rmRne
  subGz.io.a := posAddZ.io.res
  subGz.io.b := neg(io.grid_min.z)
  subGz.io.rm := rmRne

  val mulIdxX = Module(new FMUL(c.cfg))
  val mulIdxY = Module(new FMUL(c.cfg))
  val mulIdxZ = Module(new FMUL(c.cfg))

  mulIdxX.io.a := subGx.io.res
  mulIdxX.io.b := io.inv_voxel.x
  mulIdxX.io.rm := rmRne
  mulIdxY.io.a := subGy.io.res
  mulIdxY.io.b := io.inv_voxel.y
  mulIdxY.io.rm := rmRne
  mulIdxZ.io.a := subGz.io.res
  mulIdxZ.io.b := io.inv_voxel.z
  mulIdxZ.io.rm := rmRne

  val fpToIntX = Module(new FPToInt(c.cfg.expWidth, c.cfg.precision, c.cfg.fptointLatency))
  val fpToIntY = Module(new FPToInt(c.cfg.expWidth, c.cfg.precision, c.cfg.fptointLatency))
  val fpToIntZ = Module(new FPToInt(c.cfg.expWidth, c.cfg.precision, c.cfg.fptointLatency))

  fpToIntX.io.a := mulIdxX.io.result
  fpToIntX.io.rm := rmRtz
  fpToIntX.io.op := "b11".U
  fpToIntY.io.a := mulIdxY.io.result
  fpToIntY.io.rm := rmRtz
  fpToIntY.io.op := "b11".U
  fpToIntZ.io.a := mulIdxZ.io.result
  fpToIntZ.io.rm := rmRtz
  fpToIntZ.io.op := "b11".U

  val xNeg = fpToIntX.io.result(63)
  val yNeg = fpToIntY.io.result(63)
  val zNeg = fpToIntZ.io.result(63)

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
  val addrValid = pipeBool(inFire, addrLatency)

  val rayOXAtAddr = pipeUInt(io.in.bits.ray.origin.x, addrLatency)
  val rayOYAtAddr = pipeUInt(io.in.bits.ray.origin.y, addrLatency)
  val rayOZAtAddr = pipeUInt(io.in.bits.ray.origin.z, addrLatency)
  val rayDXAtAddr = pipeUInt(io.in.bits.ray.dir.x, addrLatency)
  val rayDYAtAddr = pipeUInt(io.in.bits.ray.dir.y, addrLatency)
  val rayDZAtAddr = pipeUInt(io.in.bits.ray.dir.z, addrLatency)
  val rayDistAtAddr = pipeUInt(io.in.bits.ray.dist, addrLatency)
  val prevSdfAtAddr = pipeUInt(io.in.bits.prevSdf, addrLatency)

  val iterAtAddr = pipeUInt(io.in.bits.iter, addrLatency)

  val slotIdAtAddr = pipeUInt(io.in.bits.meta.slotId, addrLatency)
  val pixelXAtAddr = pipeUInt(io.in.bits.meta.pixelX, addrLatency)
  val pixelYAtAddr = pipeUInt(io.in.bits.meta.pixelY, addrLatency)


  val sdfMemLatency = GlobalConfig.sdfMemDpiLatency
  io.sdf_mem_req.valid := addrValid && addrInBounds
  io.sdf_mem_req.bits.globalIdx := globalLinear
  io.sdf_mem_req.bits.localIdx := localLinear
  when(io.sdf_mem_req.valid) {
    assert(io.sdf_mem_req.ready, "SdfPE expects sdf_mem_req.ready to stay high in pipeline mode")
  }

  val bValid = pipeBool(addrValid, sdfMemLatency)
  val bInBounds = pipeBool(addrInBounds, sdfMemLatency)

  val bRayOX = sampleThenDelay(rayOXAtAddr,addrValid, sdfMemLatency)
  val bRayOY = sampleThenDelay(rayOYAtAddr, addrValid, sdfMemLatency)
  val bRayOZ = sampleThenDelay(rayOZAtAddr, addrValid, sdfMemLatency)
  val bRayDX = sampleThenDelay(rayDXAtAddr, addrValid, sdfMemLatency)
  val bRayDY = sampleThenDelay(rayDYAtAddr, addrValid, sdfMemLatency)
  val bRayDZ = sampleThenDelay(rayDZAtAddr, addrValid, sdfMemLatency)
  val bRayDist = sampleThenDelay(rayDistAtAddr, addrValid, sdfMemLatency)
  val bPrevSdf = sampleThenDelay(prevSdfAtAddr, addrValid, sdfMemLatency)

  val bIter = pipeUInt(iterAtAddr,sdfMemLatency)

  val bSlotId = pipeUInt(slotIdAtAddr,sdfMemLatency)
  val bPixelX = pipeUInt(pixelXAtAddr,  sdfMemLatency)
  val bPixelY = pipeUInt(pixelYAtAddr, sdfMemLatency)

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
  when(bIter < halfStepsU) {
    currentHitThreshold := hitThreshold1
  }.elsewhen(bIter < threeQuarterStepsU) {
    currentHitThreshold := hitThreshold2
  }.otherwise {
    currentHitThreshold := hitThreshold3
  }

  val cmpHit = Module(new FCMP(c.cfg))
  cmpHit.io.a := bSampleAbs
  cmpHit.io.b := currentHitThreshold
  cmpHit.io.signaling := false.B

  val cmpLatency = c.cfg.fcmpLatency
  val bValidCmp = pipeBool(bValid, cmpLatency)
  val bInBoundsCmp = pipeBool(bInBounds, cmpLatency)
  val bHit = bInBoundsCmp && cmpHit.io.lt

  val bIterCmp = pipeUInt(bIter, cmpLatency)
  val bRayOXCmp = pipeUInt(bRayOX, cmpLatency)
  val bRayOYCmp = pipeUInt(bRayOY, cmpLatency)
  val bRayOZCmp = pipeUInt(bRayOZ, cmpLatency)
  val bRayDXCmp = pipeUInt(bRayDX, cmpLatency)
  val bRayDYCmp = pipeUInt(bRayDY, cmpLatency)
  val bRayDZCmp = pipeUInt(bRayDZ, cmpLatency)
  val bRayDistCmp = pipeUInt(bRayDist, cmpLatency)
  val bPrevSdfCmp = pipeUInt(bPrevSdf, cmpLatency)
  val bSampleCmp = pipeUInt(bSample, cmpLatency)
  val bSlotIdCmp = pipeUInt(bSlotId, cmpLatency)
  val bPixelXCmp = pipeUInt(bPixelX, cmpLatency)
  val bPixelYCmp = pipeUInt(bPixelY, cmpLatency)
  val bSampleNegCmp = pipeBool(bSampleNeg, cmpLatency)

  val prevSdfHalf = fpHalf(bPrevSdfCmp)
  val negPrevSdfHalf = neg(prevSdfHalf)
  val sampleIsNeg = bSampleNegCmp

  val posDistAdd = Module(new FADD(c.cfg))
  posDistAdd.io.a := bRayDistCmp
  posDistAdd.io.b := bSampleCmp
  posDistAdd.io.rm := rmRne

  val negDistAdd = Module(new FADD(c.cfg))
  negDistAdd.io.a := bRayDistCmp
  negDistAdd.io.b := negPrevSdfHalf
  negDistAdd.io.rm := rmRne

  val nextPrevSdfRaw = Mux(sampleIsNeg, prevSdfHalf, bSampleCmp)
  val sampleIsNegAtOut = pipeBool(sampleIsNeg, advanceLatency)

  // Miss-only path uses prevSdf carry-forward with negative-sample backoff.
  val outValid = pipeBool(bValidCmp && !bHit, advanceLatency)
  val outInBounds = pipeBool(bInBoundsCmp && !bHit && bValidCmp, advanceLatency)
  val outIterRaw = pipeUInt(Mux(bValidCmp, bIterCmp + 1.U, 0.U), advanceLatency)
  val outIter = Mux(outInBounds, outIterRaw, maxStepsU)

  val outCurrX = pipeUInt(bRayOXCmp, advanceLatency)
  val outCurrY = pipeUInt(bRayOYCmp, advanceLatency)
  val outCurrZ = pipeUInt(bRayOZCmp, advanceLatency)

  val outRayDX = pipeUInt(bRayDXCmp, advanceLatency)
  val outRayDY = pipeUInt(bRayDYCmp, advanceLatency)
  val outRayDZ = pipeUInt(bRayDZCmp, advanceLatency)
  val outCurrDist = Mux(sampleIsNegAtOut, negDistAdd.io.res, posDistAdd.io.res)
  val outPrevSdf = pipeUInt(nextPrevSdfRaw, advanceLatency)

  val outSlotId = pipeUInt(bSlotIdCmp, advanceLatency)
  val outPixelX = pipeUInt(bPixelXCmp, advanceLatency)
  val outPixelY = pipeUInt(bPixelYCmp, advanceLatency)

  val outDist = outCurrDist

  val hitOutValid = pipeBool(bValidCmp && bHit, advanceLatency)
  val hitOutIter = pipeUInt(bIterCmp + 1.U, advanceLatency)
  val hitOutSlotId = pipeUInt(bSlotIdCmp, advanceLatency)
  val hitOutPixelX = pipeUInt(bPixelXCmp, advanceLatency)
  val hitOutPixelY = pipeUInt(bPixelYCmp, advanceLatency)
  val hitOutOriginX = pipeUInt(bRayOXCmp, advanceLatency)
  val hitOutOriginY = pipeUInt(bRayOYCmp, advanceLatency)
  val hitOutOriginZ = pipeUInt(bRayOZCmp, advanceLatency)
  val hitOutRayDX = pipeUInt(bRayDXCmp, advanceLatency)
  val hitOutRayDY = pipeUInt(bRayDYCmp, advanceLatency)
  val hitOutRayDZ = pipeUInt(bRayDZCmp, advanceLatency)
  val hitStayDist = pipeUInt(bRayDistCmp, advanceLatency)
  val hitOutDist = Mux(sampleIsNegAtOut, negDistAdd.io.res, hitStayDist)
  val hitOutPrevSdf = pipeUInt(nextPrevSdfRaw, advanceLatency)

  // --------------------
  // Stage D: output
  // --------------------
  io.out.valid := outValid
  io.out.bits.meta.slotId := outSlotId
  io.out.bits.meta.pixelX := outPixelX
  io.out.bits.meta.pixelY := outPixelY
  io.out.bits.hit := false.B
  io.out.bits.iter := outIter
  io.out.bits.prevSdf := outPrevSdf

  io.out.bits.ray.origin.x := outCurrX
  io.out.bits.ray.origin.y := outCurrY
  io.out.bits.ray.origin.z := outCurrZ
  io.out.bits.ray.dir.x := outRayDX
  io.out.bits.ray.dir.y := outRayDY
  io.out.bits.ray.dir.z := outRayDZ
  io.out.bits.ray.dist := outDist

  io.out_hit.valid := hitOutValid
  io.out_hit.bits.meta.slotId := hitOutSlotId
  io.out_hit.bits.meta.pixelX := hitOutPixelX
  io.out_hit.bits.meta.pixelY := hitOutPixelY
  io.out_hit.bits.hit := true.B
  io.out_hit.bits.iter := hitOutIter
  io.out_hit.bits.prevSdf := hitOutPrevSdf

  io.out_hit.bits.ray.origin.x := hitOutOriginX
  io.out_hit.bits.ray.origin.y := hitOutOriginY
  io.out_hit.bits.ray.origin.z := hitOutOriginZ
  io.out_hit.bits.ray.dir.x := hitOutRayDX
  io.out_hit.bits.ray.dir.y := hitOutRayDY
  io.out_hit.bits.ray.dir.z := hitOutRayDZ
  io.out_hit.bits.ray.dist := hitOutDist

  when(io.out.valid) {
    assert(io.out.ready, "SdfPE expects io.out.ready to stay high in pipeline mode")
  }
  when(io.out_hit.valid) {
    assert(io.out_hit.ready, "SdfPE expects io.out_hit.ready to stay high in pipeline mode")
  }
}
