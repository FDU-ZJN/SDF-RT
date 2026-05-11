package DDA

import chisel3._
import chisel3.util._
import raytrace_utils._
import raytrace_utils.fudian._
import raytrace_utils.PipeUtils._

class DdaStepPipelinePE(
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
    val mem_req = Decoupled(new DdaSubgridMetaReq(addrWidth))
    val mem_resp = Flipped(Valid(new DdaSubgridMetaResp))
    val out = Decoupled(new DdaStepResult(cfg, addrWidth))
  })

  private val subShift = log2Ceil(subRes)
  private val globalShift = log2Ceil(globalRes)
  private val globalPlaneShift = globalShift + globalShift
  private val subPlaneShift = subShift + subShift
  private val subMask = (subRes - 1).S((addrWidth + 1).W)
  private val totalSub = globalRes * subRes
  private val totalSubS = totalSub.S((addrWidth + 1).W)
  private val memLatency = GlobalConfig.subgridMemDpiLatency
  private val stepCalcLatency = cfg.fcmpLatency + cfg.faddLatency
  private val totalStepLatency = math.max(memLatency, stepCalcLatency)
  private val outQueueDepth = 16

  private def negStep(isNeg: Bool): SInt =
    Mux(isNeg, (-1).S((addrWidth + 1).W), 1.S((addrWidth + 1).W))

  val ctx = io.in.bits
  val inBounds = ctx.subX >= 0.S && ctx.subY >= 0.S && ctx.subZ >= 0.S &&
    ctx.subX < totalSubS && ctx.subY < totalSubS && ctx.subZ < totalSubS
  val terminalNow = !inBounds || ctx.iter >= maxTraversalSteps.U
  val nonTerminalValid = io.in.valid && !terminalNow

  val globalX = (ctx.subX.asUInt >> subShift).pad(addrWidth)(addrWidth - 1, 0)
  val globalY = (ctx.subY.asUInt >> subShift).pad(addrWidth)(addrWidth - 1, 0)
  val globalZ = (ctx.subZ.asUInt >> subShift).pad(addrWidth)(addrWidth - 1, 0)
  val subCellX = (ctx.subX & subMask).asUInt.pad(addrWidth)(addrWidth - 1, 0)
  val subCellY = (ctx.subY & subMask).asUInt.pad(addrWidth)(addrWidth - 1, 0)
  val subCellZ = (ctx.subZ & subMask).asUInt.pad(addrWidth)(addrWidth - 1, 0)
  val globalYScaled = (globalY << globalShift).asUInt
  val globalZScaled = (globalZ << globalPlaneShift).asUInt
  val subYScaled = (subCellY << subShift).asUInt
  val subZScaled = (subCellZ << subPlaneShift).asUInt
  val globalLinear = globalX + globalYScaled + globalZScaled
  val subLinear = subCellX + subYScaled + subZScaled

  io.mem_req.valid := nonTerminalValid
  io.mem_req.bits.globalIdx := globalLinear
  io.mem_req.bits.subIdx := subLinear
  io.in.ready := terminalNow || io.mem_req.ready

  val inFire = io.in.fire
  val nonTerminalFire = inFire && !terminalNow

  val cmpXY = Module(new FCMP(cfg))
  val cmpXZ = Module(new FCMP(cfg))
  val cmpYZ = Module(new FCMP(cfg))
  cmpXY.io.a := ctx.tMaxX
  cmpXY.io.b := ctx.tMaxY
  cmpXY.io.signaling := false.B
  cmpXZ.io.a := ctx.tMaxX
  cmpXZ.io.b := ctx.tMaxZ
  cmpXZ.io.signaling := false.B
  cmpYZ.io.a := ctx.tMaxY
  cmpYZ.io.b := ctx.tMaxZ
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
  addTMaxX.io.a := ctx.tMaxX
  addTMaxX.io.b := ctx.tDeltaX
  addTMaxY.io.a := ctx.tMaxY
  addTMaxY.io.b := ctx.tDeltaY
  addTMaxZ.io.a := ctx.tMaxZ
  addTMaxZ.io.b := ctx.tDeltaZ

  val expectedResp = pipeBool(nonTerminalFire, memLatency)
  when(expectedResp) {
    assert(io.mem_resp.valid, "DdaStepPipelinePE expects fixed-latency mem response after grant")
  }

  val outValid = pipeBool(inFire, totalStepLatency)
  val terminalAtOut = pipeBool(terminalNow, totalStepLatency)
  val ctxAtOut = pipeData(ctx, totalStepLatency)
  val axisAtOut = pipeUInt(nextAxis, totalStepLatency - cfg.fcmpLatency)
  val addXAtOut = pipeUInt(addTMaxX.io.res, totalStepLatency - cfg.faddLatency)
  val addYAtOut = pipeUInt(addTMaxY.io.res, totalStepLatency - cfg.faddLatency)
  val addZAtOut = pipeUInt(addTMaxZ.io.res, totalStepLatency - cfg.faddLatency)
  val triStartAtOut = pipeUInt(io.mem_resp.bits.triStart, totalStepLatency - memLatency)
  val triCountAtOut = pipeUInt(io.mem_resp.bits.triCount, totalStepLatency - memLatency)

  val stepNegXAtOut = ctxAtOut.ray.dir.x(cfg.totalWidth - 1)
  val stepNegYAtOut = ctxAtOut.ray.dir.y(cfg.totalWidth - 1)
  val stepNegZAtOut = ctxAtOut.ray.dir.z(cfg.totalWidth - 1)

  val nextCtx = Wire(new DdaContext(cfg, addrWidth))
  nextCtx := ctxAtOut
  when(axisAtOut === 0.U) {
    nextCtx.subX := ctxAtOut.subX + negStep(stepNegXAtOut)
    nextCtx.tMaxX := addXAtOut
  }.elsewhen(axisAtOut === 1.U) {
    nextCtx.subY := ctxAtOut.subY + negStep(stepNegYAtOut)
    nextCtx.tMaxY := addYAtOut
  }.otherwise {
    nextCtx.subZ := ctxAtOut.subZ + negStep(stepNegZAtOut)
    nextCtx.tMaxZ := addZAtOut
  }
  nextCtx.iter := ctxAtOut.iter + 1.U

  val outQ = Module(new Queue(new DdaStepResult(cfg, addrWidth), outQueueDepth))
  outQ.io.enq.valid := outValid
  outQ.io.enq.bits.ctx := Mux(terminalAtOut, ctxAtOut, nextCtx)
  outQ.io.enq.bits.done := terminalAtOut
  outQ.io.enq.bits.emitCmd := !terminalAtOut && triCountAtOut =/= 0.U
  outQ.io.enq.bits.tri.base_addr := triStartAtOut
  outQ.io.enq.bits.tri.count := triCountAtOut
  io.out <> outQ.io.deq

  when(outQ.io.enq.valid) {
    assert(outQ.io.enq.ready, "DdaStepPipelinePE output queue overflow")
  }
}
