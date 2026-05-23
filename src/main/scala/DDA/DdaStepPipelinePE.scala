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

  private def negStep(isNeg: Bool): SInt =
    Mux(isNeg, (-1).S((addrWidth + 1).W), 1.S((addrWidth + 1).W))

  val subgridMem = Module(new SubgridMetaMemDPI(addrWidth, latency = memLatency))

  io.in.ready := true.B
  val inFire = io.in.fire
  val ctx = io.in.bits

  val inBounds = ctx.subX >= 0.S && ctx.subY >= 0.S && ctx.subZ >= 0.S &&
    ctx.subX < totalSubS && ctx.subY < totalSubS && ctx.subZ < totalSubS
  val terminalNow = !inBounds || ctx.iter >= maxTraversalSteps.U
  val nonTerminalFire = inFire && !terminalNow

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

  subgridMem.io.clk := clock
  subgridMem.io.reset := reset
  subgridMem.io.globalIdx := globalLinear
  subgridMem.io.subIdx := subLinear
  subgridMem.io.en := nonTerminalFire

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

  val outValid = pipeBool(inFire, totalStepLatency)
  val terminalAtOut = pipeBool(terminalNow, totalStepLatency)
  val ctxAtOut = pipeData(ctx, totalStepLatency)
  val axisAtOut = pipeUInt(nextAxis, totalStepLatency - cfg.fcmpLatency)
  val addXAtOut = pipeUInt(addTMaxX.io.res, totalStepLatency - cfg.faddLatency)
  val addYAtOut = pipeUInt(addTMaxY.io.res, totalStepLatency - cfg.faddLatency)
  val addZAtOut = pipeUInt(addTMaxZ.io.res, totalStepLatency - cfg.faddLatency)
  val triStartAtOut = pipeUInt(subgridMem.io.triStart, totalStepLatency - memLatency)
  val triCountAtOut = pipeUInt(subgridMem.io.triCount, totalStepLatency - memLatency)

  val nonTerminalAtMem = pipeBool(nonTerminalFire, memLatency)
  when(nonTerminalAtMem) {
    assert(subgridMem.io.valid, "DdaStepPipelinePE expects fixed-latency subgridMem response")
  }

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

  io.out.valid := outValid
  io.out.bits.ctx := Mux(terminalAtOut, ctxAtOut, nextCtx)
  io.out.bits.done := terminalAtOut
  io.out.bits.emitCmd := !terminalAtOut && triCountAtOut =/= 0.U
  io.out.bits.tri.base_addr := triStartAtOut
  io.out.bits.tri.count := triCountAtOut

  when(io.out.valid) {
    assert(io.out.ready, "DdaStepPipelinePE expects io.out.ready to stay high in pipeline mode")
  }
}
