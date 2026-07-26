package DDA

import chisel3._
import chisel3.util._
import raytrace_utils._
import raytrace_utils.fudian._
import raytrace_utils.PipeUtils._

class DdaStepPipelinePE(
  cfg: FloatConfig = FloatConfig.FP32,
  addrWidth: Int = 32,
  globalRes: Int = GlobalConfig.GlobalDdaRes,
  subRes: Int = GlobalConfig.SubDdaRes,
  maxTraversalSteps: Int = 1024
) extends Module {
  require((globalRes & (globalRes - 1)) == 0, s"globalRes must be power-of-two, got $globalRes")
  require((subRes & (subRes - 1)) == 0, s"subRes must be power-of-two, got $subRes")

  val io = IO(new Bundle {
    val in = Flipped(Decoupled(new DdaTraversalState(cfg)))
    val subgrid_mem_req = Valid(new DdaSubgridMetaReq(addrWidth))
    val subgrid_mem_resp = Input(Valid(new DdaSubgridMetaResp))
    val out = Decoupled(new DdaStepResult(cfg, addrWidth))
  })

  private val subShift = log2Ceil(subRes)
  private val globalShift = log2Ceil(globalRes)
  private val globalPlaneShift = globalShift + globalShift
  private val subPlaneShift = subShift + subShift
  private val coordWidth = GlobalConfig.ddaCoordWidth
  private val subMask = (subRes - 1).S(coordWidth.W)
  private val totalSub = globalRes * subRes
  require(totalSub <= (1 << (coordWidth - 1)) - 1,
    s"DDA resolution $totalSub exceeds $coordWidth-bit signed coordinate range")
  private val totalSubS = totalSub.S(coordWidth.W)
  private val memLatency = GlobalConfig.subgridMemDpiLatency
  private val stepCalcLatency = cfg.fcmpLatency + cfg.faddLatency
  private val totalStepLatency = math.max(memLatency, stepCalcLatency)

  private def negStep(isNeg: Bool): SInt =
    Mux(isNeg, (-1).S(coordWidth.W), 1.S(coordWidth.W))

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

  io.subgrid_mem_req.valid := nonTerminalFire
  io.subgrid_mem_req.bits.globalIdx := globalLinear
  io.subgrid_mem_req.bits.subIdx := subLinear

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

  val tMaxXAtAxis = pipeUInt(ctx.tMaxX, cfg.fcmpLatency)
  val tMaxYAtAxis = pipeUInt(ctx.tMaxY, cfg.fcmpLatency)
  val tMaxZAtAxis = pipeUInt(ctx.tMaxZ, cfg.fcmpLatency)
  val tDeltaXAtAxis = pipeUInt(ctx.tDeltaX, cfg.fcmpLatency)
  val tDeltaYAtAxis = pipeUInt(ctx.tDeltaY, cfg.fcmpLatency)
  val tDeltaZAtAxis = pipeUInt(ctx.tDeltaZ, cfg.fcmpLatency)
  val selectedTMax = MuxLookup(nextAxis, tMaxZAtAxis)(Seq(
    0.U -> tMaxXAtAxis,
    1.U -> tMaxYAtAxis
  ))
  val selectedTDelta = MuxLookup(nextAxis, tDeltaZAtAxis)(Seq(
    0.U -> tDeltaXAtAxis,
    1.U -> tDeltaYAtAxis
  ))
  val addSelectedTMax = Module(new FADD(cfg))
  addSelectedTMax.io.a := selectedTMax
  addSelectedTMax.io.b := selectedTDelta

  val outValid = pipeBool(inFire, totalStepLatency)
  val terminalAtOut = pipeBool(terminalNow, totalStepLatency)
  val ctxAtOut = pipeData(ctx, totalStepLatency)
  val axisAtOut = pipeUInt(nextAxis, totalStepLatency - cfg.fcmpLatency)
  val selectedAddAtOut = pipeUInt(addSelectedTMax.io.res, totalStepLatency - stepCalcLatency)
  val triStartAtOut = pipeUInt(io.subgrid_mem_resp.bits.triStart, totalStepLatency - memLatency)
  val triCountAtOut = pipeUInt(io.subgrid_mem_resp.bits.triCount, totalStepLatency - memLatency)

  val nonTerminalAtMem = pipeBool(nonTerminalFire, memLatency)
  when(nonTerminalAtMem) {
    assert(io.subgrid_mem_resp.valid, "DdaStepPipelinePE expects fixed-latency subgridMem response")
  }

  val stepNegXAtOut = ctxAtOut.dirNegX
  val stepNegYAtOut = ctxAtOut.dirNegY
  val stepNegZAtOut = ctxAtOut.dirNegZ

  val nextCtx = Wire(new DdaTraversalState(cfg))
  nextCtx := ctxAtOut
  when(axisAtOut === 0.U) {
    nextCtx.subX := ctxAtOut.subX + negStep(stepNegXAtOut)
    nextCtx.tMaxX := selectedAddAtOut
  }.elsewhen(axisAtOut === 1.U) {
    nextCtx.subY := ctxAtOut.subY + negStep(stepNegYAtOut)
    nextCtx.tMaxY := selectedAddAtOut
  }.otherwise {
    nextCtx.subZ := ctxAtOut.subZ + negStep(stepNegZAtOut)
    nextCtx.tMaxZ := selectedAddAtOut
  }
  nextCtx.iter := ctxAtOut.iter + 1.U

  io.out.valid := outValid
  io.out.bits.state := Mux(terminalAtOut, ctxAtOut, nextCtx)
  io.out.bits.done := terminalAtOut
  io.out.bits.emitCmd := !terminalAtOut && triCountAtOut =/= 0.U
  io.out.bits.tri.base_addr := triStartAtOut
  io.out.bits.tri.count := triCountAtOut

  when(io.out.valid) {
    assert(io.out.ready, "DdaStepPipelinePE expects io.out.ready to stay high in pipeline mode")
  }
}
