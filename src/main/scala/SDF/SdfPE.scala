package SDF

import chisel3._
import chisel3.util._
import raytrace_utils._
import raytrace_utils.fudian._

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
  val stepScale = BigInt(c.stepScaleBits & 0xffffffffL).U(c.cfg.totalWidth.W)
  val minStep = BigInt(c.minStepBits & 0xffffffffL).U(c.cfg.totalWidth.W)
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

  // Address path is FADD -> FMUL -> FPToInt (FPToInt is combinational here).
  val addrLatency = c.cfg.faddLatency + c.cfg.fmulLatency
  val marchLatency = (2 * c.cfg.fmulLatency) + c.cfg.faddLatency
  val hitStepLatency = c.cfg.faddLatency

  def neg(x: UInt): UInt = Cat(!x(c.cfg.totalWidth - 1), x(c.cfg.totalWidth - 2, 0))

  def pipeBool(x: Bool, n: Int): Bool = {
    var v = x
    for (_ <- 0 until n) v = RegNext(v, false.B)
    v
  }

  def pipeUInt(x: UInt, n: Int): UInt = {
    var v = x
    for (_ <- 0 until n) v = RegNext(v)
    v
  }

  // --------------------
  // Stage A: origin -> grid address pipeline
  // --------------------
  val subGx = Module(new FADD(c.cfg))
  val subGy = Module(new FADD(c.cfg))
  val subGz = Module(new FADD(c.cfg))

  subGx.io.a := io.in.bits.ray.origin.x
  subGx.io.b := neg(io.grid_min.x)
  subGx.io.rm := rmRne
  subGy.io.a := io.in.bits.ray.origin.y
  subGy.io.b := neg(io.grid_min.y)
  subGy.io.rm := rmRne
  subGz.io.a := io.in.bits.ray.origin.z
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

  val fpToIntX = Module(new FPToInt(c.cfg.expWidth, c.cfg.precision))
  val fpToIntY = Module(new FPToInt(c.cfg.expWidth, c.cfg.precision))
  val fpToIntZ = Module(new FPToInt(c.cfg.expWidth, c.cfg.precision))

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

  val iterAtAddr = pipeUInt(io.in.bits.iter, addrLatency)

  val slotIdAtAddr = pipeUInt(io.in.bits.meta.slotId, addrLatency)
  val pixelXAtAddr = pipeUInt(io.in.bits.meta.pixelX, addrLatency)
  val pixelYAtAddr = pipeUInt(io.in.bits.meta.pixelY, addrLatency)

  // --------------------
  // Stage B: memory request + response alignment (fixed 1-cycle mem latency)
  // --------------------
  io.sdf_mem_req.valid := addrValid && addrInBounds
  io.sdf_mem_req.bits.globalIdx := globalLinear
  io.sdf_mem_req.bits.localIdx := localLinear
  when(io.sdf_mem_req.valid) {
    assert(io.sdf_mem_req.ready, "SdfPE expects sdf_mem_req.ready to stay high in pipeline mode")
  }

  val bValid = RegNext(addrValid, false.B)
  val bInBounds = RegNext(addrInBounds, false.B)

  val bRayOX = RegEnable(rayOXAtAddr, addrValid)
  val bRayOY = RegEnable(rayOYAtAddr, addrValid)
  val bRayOZ = RegEnable(rayOZAtAddr, addrValid)
  val bRayDX = RegEnable(rayDXAtAddr, addrValid)
  val bRayDY = RegEnable(rayDYAtAddr, addrValid)
  val bRayDZ = RegEnable(rayDZAtAddr, addrValid)

  val bIter = RegEnable(iterAtAddr, addrValid)

  val bSlotId = RegEnable(slotIdAtAddr, addrValid)
  val bPixelX = RegEnable(pixelXAtAddr, addrValid)
  val bPixelY = RegEnable(pixelYAtAddr, addrValid)

  io.sdf_mem_resp.ready := bValid && bInBounds
  when(bValid && bInBounds) {
    assert(io.sdf_mem_resp.valid, "SdfPE expects fixed 1-cycle sdf_mem_resp.valid for in-bounds requests")
  }

  val bSample = Mux(bInBounds, io.sdf_mem_resp.bits, fpZero)
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

  val cmpStep = Module(new FCMP(c.cfg))
  cmpStep.io.a := minStep
  cmpStep.io.b := bSampleAbs
  cmpStep.io.signaling := false.B

  val selectedStep = Mux(cmpStep.io.le, bSampleAbs, minStep)
  val bHit = bInBounds && cmpHit.io.lt

  // Miss-only path keeps original one-step sphere tracing march.
  val cValid = pipeBool(bValid && !bHit, hitStepLatency)
  val cInBounds = pipeBool(bInBounds && !bHit, hitStepLatency)
  val cIter = pipeUInt(bIter + 1.U, hitStepLatency)

  val cRayOX = pipeUInt(bRayOX, hitStepLatency)
  val cRayOY = pipeUInt(bRayOY, hitStepLatency)
  val cRayOZ = pipeUInt(bRayOZ, hitStepLatency)
  val cRayDX = pipeUInt(bRayDX, hitStepLatency)
  val cRayDY = pipeUInt(bRayDY, hitStepLatency)
  val cRayDZ = pipeUInt(bRayDZ, hitStepLatency)

  val cSlotId = pipeUInt(bSlotId, hitStepLatency)
  val cPixelX = pipeUInt(bPixelX, hitStepLatency)
  val cPixelY = pipeUInt(bPixelY, hitStepLatency)
  val cStep = pipeUInt(selectedStep, hitStepLatency)

  val stepScaleMul = Module(new FMUL(c.cfg))
  stepScaleMul.io.a := cStep
  stepScaleMul.io.b := stepScale
  stepScaleMul.io.rm := rmRne

  val mulNx = Module(new FMUL(c.cfg))
  val mulNy = Module(new FMUL(c.cfg))
  val mulNz = Module(new FMUL(c.cfg))
  mulNx.io.a := cRayDX
  mulNx.io.b := stepScaleMul.io.result
  mulNx.io.rm := rmRne
  mulNy.io.a := cRayDY
  mulNy.io.b := stepScaleMul.io.result
  mulNy.io.rm := rmRne
  mulNz.io.a := cRayDZ
  mulNz.io.b := stepScaleMul.io.result
  mulNz.io.rm := rmRne

  val cRayOXForAdd = pipeUInt(cRayOX, 2 * c.cfg.fmulLatency)
  val cRayOYForAdd = pipeUInt(cRayOY, 2 * c.cfg.fmulLatency)
  val cRayOZForAdd = pipeUInt(cRayOZ, 2 * c.cfg.fmulLatency)

  val addNx = Module(new FADD(c.cfg))
  val addNy = Module(new FADD(c.cfg))
  val addNz = Module(new FADD(c.cfg))
  addNx.io.a := cRayOXForAdd
  addNx.io.b := mulNx.io.result
  addNx.io.rm := rmRne
  addNy.io.a := cRayOYForAdd
  addNy.io.b := mulNy.io.result
  addNy.io.rm := rmRne
  addNz.io.a := cRayOZForAdd
  addNz.io.b := mulNz.io.result
  addNz.io.rm := rmRne

  val outValid = pipeBool(cValid, marchLatency)
  val outInBounds = pipeBool(cInBounds, marchLatency)
  val outIterRaw = pipeUInt(cIter, marchLatency)
  val outIter = Mux(outInBounds, outIterRaw, maxStepsU)

  val outCurrX = pipeUInt(cRayOX, marchLatency)
  val outCurrY = pipeUInt(cRayOY, marchLatency)
  val outCurrZ = pipeUInt(cRayOZ, marchLatency)

  val outRayDX = pipeUInt(cRayDX, marchLatency)
  val outRayDY = pipeUInt(cRayDY, marchLatency)
  val outRayDZ = pipeUInt(cRayDZ, marchLatency)

  val outSlotId = pipeUInt(cSlotId, marchLatency)
  val outPixelX = pipeUInt(cPixelX, marchLatency)
  val outPixelY = pipeUInt(cPixelY, marchLatency)

  val outOriginX = Mux(!outInBounds, outCurrX, addNx.io.res)
  val outOriginY = Mux(!outInBounds, outCurrY, addNy.io.res)
  val outOriginZ = Mux(!outInBounds, outCurrZ, addNz.io.res)

  val hitPathLatency = c.cfg.fmulLatency + c.cfg.faddLatency
  val hitOutValid = pipeBool(bValid && bHit, hitPathLatency)
  val hitOutIter = pipeUInt(bIter + 1.U, hitPathLatency)
  val hitOutSlotId = pipeUInt(bSlotId, hitPathLatency)
  val hitOutPixelX = pipeUInt(bPixelX, hitPathLatency)
  val hitOutPixelY = pipeUInt(bPixelY, hitPathLatency)
  val hitOutOriginX = pipeUInt(bRayOX, hitPathLatency)
  val hitOutOriginY = pipeUInt(bRayOY, hitPathLatency)
  val hitOutOriginZ = pipeUInt(bRayOZ, hitPathLatency)
  val hitOutRayDX = pipeUInt(bRayDX, hitPathLatency)
  val hitOutRayDY = pipeUInt(bRayDY, hitPathLatency)
  val hitOutRayDZ = pipeUInt(bRayDZ, hitPathLatency)
  val hitOutReverse = pipeBool(bSampleNeg, hitPathLatency)

  // --------------------
  // Stage D: output
  // --------------------
  io.out.valid := outValid
  io.out.bits.meta.slotId := outSlotId
  io.out.bits.meta.pixelX := outPixelX
  io.out.bits.meta.pixelY := outPixelY
  io.out.bits.hit := false.B
  io.out.bits.iter := outIter
  io.out.bits.reverseTraversal := false.B

  io.out.bits.ray.origin.x := outOriginX
  io.out.bits.ray.origin.y := outOriginY
  io.out.bits.ray.origin.z := outOriginZ
  io.out.bits.ray.dir.x := outRayDX
  io.out.bits.ray.dir.y := outRayDY
  io.out.bits.ray.dir.z := outRayDZ

  io.out_hit.valid := hitOutValid
  io.out_hit.bits.meta.slotId := hitOutSlotId
  io.out_hit.bits.meta.pixelX := hitOutPixelX
  io.out_hit.bits.meta.pixelY := hitOutPixelY
  io.out_hit.bits.hit := true.B
  io.out_hit.bits.iter := hitOutIter
  io.out_hit.bits.reverseTraversal := hitOutReverse

  io.out_hit.bits.ray.origin.x := hitOutOriginX
  io.out_hit.bits.ray.origin.y := hitOutOriginY
  io.out_hit.bits.ray.origin.z := hitOutOriginZ
  io.out_hit.bits.ray.dir.x := hitOutRayDX
  io.out_hit.bits.ray.dir.y := hitOutRayDY
  io.out_hit.bits.ray.dir.z := hitOutRayDZ

  when(io.out.valid) {
    assert(io.out.ready, "SdfPE expects io.out.ready to stay high in pipeline mode")
  }
  when(io.out_hit.valid) {
    assert(io.out_hit.ready, "SdfPE expects io.out_hit.ready to stay high in pipeline mode")
  }
}
