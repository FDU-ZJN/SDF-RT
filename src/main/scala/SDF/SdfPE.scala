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

  val rmRNE = RNE
  val rmRTZ = RTZ

  val fpThreshold = BigInt(c.thresholdHex, 16).U(c.cfg.totalWidth.W)
  val fpMinStep = BigInt(c.minStepHex, 16).U(c.cfg.totalWidth.W)
  val fpZero = 0.U(c.cfg.totalWidth.W)
  val maxStepsU = c.maxSteps.U(16.W)

  val fullResX = c.GlobalResX * c.LocalResX
  val fullResY = c.GlobalResY * c.LocalResY
  val fullResZ = c.GlobalResZ * c.LocalResZ

  val fullResXU = fullResX.U(c.addrWidth.W)
  val fullResYU = fullResY.U(c.addrWidth.W)
  val fullResZU = fullResZ.U(c.addrWidth.W)
  val fullPlaneU = (fullResX * fullResY).U((2 * c.addrWidth).W)

  val globalResXU = c.GlobalResX.U(c.addrWidth.W)
  val globalResYU = c.GlobalResY.U(c.addrWidth.W)
  val globalPlaneU = (c.GlobalResX * c.GlobalResY).U((2 * c.addrWidth).W)

  val localResXU = c.LocalResX.U(c.addrWidth.W)
  val localResYU = c.LocalResY.U(c.addrWidth.W)
  val localPlaneU = (c.LocalResX * c.LocalResY).U((2 * c.addrWidth).W)

  val addrPipeLatency = c.cfg.fmulLatency + c.cfg.faddLatency + c.cfg.faddLatency + c.cfg.fmulLatency
  val pointPipeLatency = c.cfg.fmulLatency + c.cfg.faddLatency
  val tAddLatency = c.cfg.faddLatency

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
  // Stage A: input -> current point + grid address pipeline
  // --------------------
  val mulTx = Module(new FMUL(c.cfg))
  val mulTy = Module(new FMUL(c.cfg))
  val mulTz = Module(new FMUL(c.cfg))

  mulTx.io.a := io.in.bits.ray.dir.x;
  mulTx.io.b := io.in.bits.tNear;
  mulTx.io.rm := rmRNE
  mulTy.io.a := io.in.bits.ray.dir.y;
  mulTy.io.b := io.in.bits.tNear;
  mulTy.io.rm := rmRNE
  mulTz.io.a := io.in.bits.ray.dir.z;
  mulTz.io.b := io.in.bits.tNear;
  mulTz.io.rm := rmRNE

  val addPx = Module(new FADD(c.cfg))
  val addPy = Module(new FADD(c.cfg))
  val addPz = Module(new FADD(c.cfg))

  addPx.io.a := io.in.bits.ray.origin.x;
  addPx.io.b := mulTx.io.result;
  addPx.io.rm := rmRNE
  addPy.io.a := io.in.bits.ray.origin.y;
  addPy.io.b := mulTy.io.result;
  addPy.io.rm := rmRNE
  addPz.io.a := io.in.bits.ray.origin.z;
  addPz.io.b := mulTz.io.result;
  addPz.io.rm := rmRNE

  val subGx = Module(new FADD(c.cfg))
  val subGy = Module(new FADD(c.cfg))
  val subGz = Module(new FADD(c.cfg))

  subGx.io.a := addPx.io.res;
  subGx.io.b := neg(io.grid_min.x);
  subGx.io.rm := rmRNE
  subGy.io.a := addPy.io.res;
  subGy.io.b := neg(io.grid_min.y);
  subGy.io.rm := rmRNE
  subGz.io.a := addPz.io.res;
  subGz.io.b := neg(io.grid_min.z);
  subGz.io.rm := rmRNE

  val mulIdxX = Module(new FMUL(c.cfg))
  val mulIdxY = Module(new FMUL(c.cfg))
  val mulIdxZ = Module(new FMUL(c.cfg))

  mulIdxX.io.a := subGx.io.res;
  mulIdxX.io.b := io.inv_voxel.x;
  mulIdxX.io.rm := rmRNE
  mulIdxY.io.a := subGy.io.res;
  mulIdxY.io.b := io.inv_voxel.y;
  mulIdxY.io.rm := rmRNE
  mulIdxZ.io.a := subGz.io.res;
  mulIdxZ.io.b := io.inv_voxel.z;
  mulIdxZ.io.rm := rmRNE

  val fpToIntX = Module(new FPToInt(c.cfg.expWidth, c.cfg.precision))
  val fpToIntY = Module(new FPToInt(c.cfg.expWidth, c.cfg.precision))
  val fpToIntZ = Module(new FPToInt(c.cfg.expWidth, c.cfg.precision))

  fpToIntX.io.a := mulIdxX.io.result;
  fpToIntX.io.rm := rmRTZ;
  fpToIntX.io.op := "b11".U
  fpToIntY.io.a := mulIdxY.io.result;
  fpToIntY.io.rm := rmRTZ;
  fpToIntY.io.op := "b11".U
  fpToIntZ.io.a := mulIdxZ.io.result;
  fpToIntZ.io.rm := rmRTZ;
  fpToIntZ.io.op := "b11".U

  val xNeg = fpToIntX.io.result(63)
  val yNeg = fpToIntY.io.result(63)
  val zNeg = fpToIntZ.io.result(63)

  val xIdx = fpToIntX.io.result(c.addrWidth - 1, 0)
  val yIdx = fpToIntY.io.result(c.addrWidth - 1, 0)
  val zIdx = fpToIntZ.io.result(c.addrWidth - 1, 0)

  val inBounds = !xNeg && !yNeg && !zNeg &&
    (xIdx < fullResXU) && (yIdx < fullResYU) && (zIdx < fullResZU)

  // Hierarchical index split: fullIdx = globalIdx * LocalRes + localIdx
  val xGlobal = xIdx / localResXU
  val yGlobal = yIdx / c.LocalResY.U(c.addrWidth.W)
  val zGlobal = zIdx / c.LocalResZ.U(c.addrWidth.W)
  val xLocal = xIdx % localResXU
  val yLocal = yIdx % c.LocalResY.U(c.addrWidth.W)
  val zLocal = zIdx % c.LocalResZ.U(c.addrWidth.W)

  val globalLinearWide = xGlobal + yGlobal * globalResXU + zGlobal * globalPlaneU
  val localLinearWide = xLocal + yLocal * localResXU + zLocal * localPlaneU
  val globalLinear = globalLinearWide(c.addrWidth - 1, 0)
  val localLinear = localLinearWide(c.addrWidth - 1, 0)

  io.in.ready := true.B
  val inFire = io.in.fire
  val addrValid = pipeBool(inFire, addrPipeLatency)
  val posDelay = addrPipeLatency - pointPipeLatency

  val inBoundsAtAddr = pipeBool(inBounds, addrPipeLatency)

  val rayOXAtAddr = pipeUInt(io.in.bits.ray.origin.x, addrPipeLatency)
  val rayOYAtAddr = pipeUInt(io.in.bits.ray.origin.y, addrPipeLatency)
  val rayOZAtAddr = pipeUInt(io.in.bits.ray.origin.z, addrPipeLatency)
  val rayDXAtAddr = pipeUInt(io.in.bits.ray.dir.x, addrPipeLatency)
  val rayDYAtAddr = pipeUInt(io.in.bits.ray.dir.y, addrPipeLatency)
  val rayDZAtAddr = pipeUInt(io.in.bits.ray.dir.z, addrPipeLatency)

  val tNearAtAddr = pipeUInt(io.in.bits.tNear, addrPipeLatency)
  val iterAtAddr = pipeUInt(io.in.bits.iter, addrPipeLatency)

  val slotIdAtAddr = pipeUInt(io.in.bits.meta.slotId, addrPipeLatency)
  val pixelXAtAddr = pipeUInt(io.in.bits.meta.pixelX, addrPipeLatency)
  val pixelYAtAddr = pipeUInt(io.in.bits.meta.pixelY, addrPipeLatency)

  val currXAtAddr = pipeUInt(addPx.io.res, posDelay)
  val currYAtAddr = pipeUInt(addPy.io.res, posDelay)
  val currZAtAddr = pipeUInt(addPz.io.res, posDelay)

  // --------------------
  // Stage B: memory request + response alignment (fixed 1-cycle mem latency)
  // --------------------
  io.sdf_mem_req.valid := addrValid && inBoundsAtAddr
  io.sdf_mem_req.bits.globalIdx := globalLinear
  io.sdf_mem_req.bits.localIdx := localLinear
  when(io.sdf_mem_req.valid) {
    assert(io.sdf_mem_req.ready, "SdfPE expects sdf_mem_req.ready to stay high in pipeline mode")
  }

  val bValid = RegNext(addrValid, false.B)
  val bInBounds = RegEnable(inBoundsAtAddr, addrValid)

  val bRayOX = RegEnable(rayOXAtAddr, addrValid)
  val bRayOY = RegEnable(rayOYAtAddr, addrValid)
  val bRayOZ = RegEnable(rayOZAtAddr, addrValid)
  val bRayDX = RegEnable(rayDXAtAddr, addrValid)
  val bRayDY = RegEnable(rayDYAtAddr, addrValid)
  val bRayDZ = RegEnable(rayDZAtAddr, addrValid)

  val bTNear = RegEnable(tNearAtAddr, addrValid)
  val bIter = RegEnable(iterAtAddr, addrValid)

  val bSlotId = RegEnable(slotIdAtAddr, addrValid)
  val bPixelX = RegEnable(pixelXAtAddr, addrValid)
  val bPixelY = RegEnable(pixelYAtAddr, addrValid)

  val bCurrX = RegEnable(currXAtAddr, addrValid)
  val bCurrY = RegEnable(currYAtAddr, addrValid)
  val bCurrZ = RegEnable(currZAtAddr, addrValid)

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
  cmpHit.io.b := fpThreshold
  cmpHit.io.signaling := false.B

  val cmpStep = Module(new FCMP(c.cfg))
  cmpStep.io.a := fpMinStep
  cmpStep.io.b := bSample
  cmpStep.io.signaling := false.B

  val stepSel = Mux(cmpStep.io.le, bSample, fpMinStep)

  val addT = Module(new FADD(c.cfg))
  addT.io.a := bTNear
  addT.io.b := stepSel
  addT.io.rm := rmRNE

  val cValid = pipeBool(bValid, tAddLatency)
  val cInBounds = pipeBool(bInBounds, tAddLatency)
  val cHit = pipeBool(bInBounds && cmpHit.io.le, tAddLatency)
  val cHitT = pipeUInt(Mux(bInBounds && cmpHit.io.le, bTNear, fpZero), tAddLatency)
  val cIter = pipeUInt(bIter + 1.U, tAddLatency)

  val cCurrX = pipeUInt(bCurrX, tAddLatency)
  val cCurrY = pipeUInt(bCurrY, tAddLatency)
  val cCurrZ = pipeUInt(bCurrZ, tAddLatency)

  val cRayOX = pipeUInt(bRayOX, tAddLatency)
  val cRayOY = pipeUInt(bRayOY, tAddLatency)
  val cRayOZ = pipeUInt(bRayOZ, tAddLatency)
  val cRayDX = pipeUInt(bRayDX, tAddLatency)
  val cRayDY = pipeUInt(bRayDY, tAddLatency)
  val cRayDZ = pipeUInt(bRayDZ, tAddLatency)

  val cSlotId = pipeUInt(bSlotId, tAddLatency)
  val cPixelX = pipeUInt(bPixelX, tAddLatency)
  val cPixelY = pipeUInt(bPixelY, tAddLatency)

  val cTNext = pipeUInt(addT.io.res, tAddLatency)

  // Compute next origin for miss case; hit keeps current position.
  val mulNx = Module(new FMUL(c.cfg))
  val mulNy = Module(new FMUL(c.cfg))
  val mulNz = Module(new FMUL(c.cfg))

  mulNx.io.a := cRayDX;
  mulNx.io.b := cTNext;
  mulNx.io.rm := rmRNE
  mulNy.io.a := cRayDY;
  mulNy.io.b := cTNext;
  mulNy.io.rm := rmRNE
  mulNz.io.a := cRayDZ;
  mulNz.io.b := cTNext;
  mulNz.io.rm := rmRNE

  val addNx = Module(new FADD(c.cfg))
  val addNy = Module(new FADD(c.cfg))
  val addNz = Module(new FADD(c.cfg))

  addNx.io.a := cRayOX;
  addNx.io.b := mulNx.io.result;
  addNx.io.rm := rmRNE
  addNy.io.a := cRayOY;
  addNy.io.b := mulNy.io.result;
  addNy.io.rm := rmRNE
  addNz.io.a := cRayOZ;
  addNz.io.b := mulNz.io.result;
  addNz.io.rm := rmRNE

  val outValid = pipeBool(cValid, pointPipeLatency)
  val outInBounds = pipeBool(cInBounds, pointPipeLatency)
  val outHit = pipeBool(cHit, pointPipeLatency)
  val outHitT = pipeUInt(cHitT, pointPipeLatency)
  val outIterRaw = pipeUInt(cIter, pointPipeLatency)
  // If the marched point has gone out of grid, force terminal iteration count to stop retries.
  val outIter = Mux(outInBounds, outIterRaw, maxStepsU)

  val outCurrX = pipeUInt(cCurrX, pointPipeLatency)
  val outCurrY = pipeUInt(cCurrY, pointPipeLatency)
  val outCurrZ = pipeUInt(cCurrZ, pointPipeLatency)

  val outRayDX = pipeUInt(cRayDX, pointPipeLatency)
  val outRayDY = pipeUInt(cRayDY, pointPipeLatency)
  val outRayDZ = pipeUInt(cRayDZ, pointPipeLatency)

  val outSlotId = pipeUInt(cSlotId, pointPipeLatency)
  val outPixelX = pipeUInt(cPixelX, pointPipeLatency)
  val outPixelY = pipeUInt(cPixelY, pointPipeLatency)

  io.out.valid := outValid
  io.out.bits.meta.slotId := outSlotId
  io.out.bits.meta.pixelX := outPixelX
  io.out.bits.meta.pixelY := outPixelY
  io.out.bits.hit := outHit
  io.out.bits.hitT := outHitT
  io.out.bits.steps := outIter
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
