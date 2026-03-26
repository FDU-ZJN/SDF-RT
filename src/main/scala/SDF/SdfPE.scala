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

    val out = Decoupled(new SdfRayResp(c.cfg, c.addrWidth))
  })

  val rmRne = RNE
  val rmRtz = RTZ

  val hitThreshold = BigInt(c.thresholdBits & 0xffffffffL).U(c.cfg.totalWidth.W)
  val minStep = BigInt(c.minStepBits & 0xffffffffL).U(c.cfg.totalWidth.W)
  val fpZero = 0.U(c.cfg.totalWidth.W)
  val maxStepsU = c.maxSteps.U(16.W)

  val fullResXU = (c.GlobalResX * c.LocalResX).U(c.addrWidth.W)
  val fullResYU = (c.GlobalResY * c.LocalResY).U(c.addrWidth.W)
  val fullResZU = (c.GlobalResZ * c.LocalResZ).U(c.addrWidth.W)

  val globalResXU = c.GlobalResX.U(c.addrWidth.W)
  val globalPlaneU = (c.GlobalResX * c.GlobalResY).U((2 * c.addrWidth).W)

  val localResXU = c.LocalResX.U(c.addrWidth.W)
  val localPlaneU = (c.LocalResX * c.LocalResY).U((2 * c.addrWidth).W)

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
  val marchLatency = c.cfg.fmulLatency + c.cfg.faddLatency
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

  // --------------------
  // Stage C: one-step march update
  // --------------------
  val sampleAbs = Wire(UInt(c.cfg.totalWidth.W))
  sampleAbs := Cat(0.U(1.W), bSample(c.cfg.totalWidth - 2, 0))

  val cmpHit = Module(new FCMP(c.cfg))
  cmpHit.io.a := sampleAbs
  cmpHit.io.b := hitThreshold
  cmpHit.io.signaling := false.B

  val cmpStep = Module(new FCMP(c.cfg))
  cmpStep.io.a := minStep
  cmpStep.io.b := bSample
  cmpStep.io.signaling := false.B

  val selectedStep = Mux(cmpStep.io.le, bSample, minStep)

  val cValid = pipeBool(bValid, hitStepLatency)
  val cInBounds = pipeBool(bInBounds, hitStepLatency)
  val cHit = pipeBool(bInBounds && cmpHit.io.le, hitStepLatency)
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

  val mulNx = Module(new FMUL(c.cfg))
  val mulNy = Module(new FMUL(c.cfg))
  val mulNz = Module(new FMUL(c.cfg))

  mulNx.io.a := cRayDX
  mulNx.io.b := cStep
  mulNx.io.rm := rmRne
  mulNy.io.a := cRayDY
  mulNy.io.b := cStep
  mulNy.io.rm := rmRne
  mulNz.io.a := cRayDZ
  mulNz.io.b := cStep
  mulNz.io.rm := rmRne

  // Align ray origin with FMUL output latency before FADD.
  val cRayOXForAdd = pipeUInt(cRayOX, c.cfg.fmulLatency)
  val cRayOYForAdd = pipeUInt(cRayOY, c.cfg.fmulLatency)
  val cRayOZForAdd = pipeUInt(cRayOZ, c.cfg.fmulLatency)

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
  val outHit = pipeBool(cHit, marchLatency)
  val outIterRaw = pipeUInt(cIter, marchLatency)
  // If the marched point has gone out of grid, force terminal iteration count to stop retries.
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

  io.out.valid := outValid
  io.out.bits.meta.slotId := outSlotId
  io.out.bits.meta.pixelX := outPixelX
  io.out.bits.meta.pixelY := outPixelY
  io.out.bits.hit := outHit
  io.out.bits.iter := outIter

  io.out.bits.ray.origin.x := Mux(!outInBounds || outHit, outCurrX, addNx.io.res)
  io.out.bits.ray.origin.y := Mux(!outInBounds || outHit, outCurrY, addNy.io.res)
  io.out.bits.ray.origin.z := Mux(!outInBounds || outHit, outCurrZ, addNz.io.res)
  io.out.bits.ray.dir.x := outRayDX
  io.out.bits.ray.dir.y := outRayDY
  io.out.bits.ray.dir.z := outRayDZ

  when(io.out.valid) {
    assert(io.out.ready, "SdfPE expects io.out.ready to stay high in pipeline mode")
  }
}
