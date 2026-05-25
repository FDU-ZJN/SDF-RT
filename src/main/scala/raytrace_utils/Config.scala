package raytrace_utils

import chisel3._
import chisel3.util.log2Ceil

object GlobalConfig {
  val frameWidth =1920
  val frameHeight = 1080
  val pixelQueueDepth = 512
  val traceNumWorkers = 8
  val triMemNumPEs = 2
  val triMemNumBanks = 8
  val ddaNumWorkers = 2
  val sdfStepNumWorkers = 2
  val triRefPackFactor = 32

  private var useBlackBoxState = 0
  def memImplMode: Int = useBlackBoxState
  def setMemImplMode(value: Int): Unit = {
    require(value >= 0 && value <= 2, s"memImplMode must be 0/1/2, got $value")
    useBlackBoxState = value
  }
  def withMemImplMode[T](value: Int)(body: => T): T = {
    require(value >= 0 && value <= 2, s"memImplMode must be 0/1/2, got $value")
    val prev = useBlackBoxState
    useBlackBoxState = value
    try body finally useBlackBoxState = prev
  }

  // Backward-compatible boolean API: false -> mode 0, true -> mode 1
  def useBlackBox: Boolean = useBlackBoxState != 0
  def setUseBlackBox(value: Boolean): Unit = setMemImplMode(if (value) 1 else 0)
  def withUseBlackBox[T](value: Boolean)(body: => T): T = {
    withMemImplMode(if (value) 1 else 0)(body)
  }

  private var useFloatIPState = false
  def useFloatIP: Boolean = useFloatIPState
  def setUseFloatIP(value: Boolean): Unit = { useFloatIPState = value }
  def withUseFloatIP[T](value: Boolean)(body: => T): T = {
    val prev = useFloatIPState
    useFloatIPState = value
    try body finally useFloatIPState = prev
  }

  val commitQueueDepth = 128
  val slotBits = log2Ceil(commitQueueDepth)

  val ddaRetryQueueDepth = 16
  val ddaTraceSlotCount = ddaNumWorkers * ddaRetryQueueDepth
  val triBatchQueueDepth = ddaTraceSlotCount
  val ddaTraceSlotBits = log2Ceil(ddaTraceSlotCount)

  val sdfRetryQueueDepth = 128
  val sdfFinalQueueDepth = 128
  val sdfMissWritebackQueueDepth = 4
  val renderOutputQueueDepth = 16

  val normalMemDpiLatency = 3
  val triMemDpiLatency = 3
  val triRefMemDpiLatency = 2
  val sdfMemDpiLatency = 3
  val subgridMemDpiLatency = 2
  val fsqrtLatency = 11
  val fmulLatency = 4
  val faddLatency = 5
  val fmaLatency = 8
  val fcmpLatency = 1
  val fptointLatency = 1
  val fdivLatency = 10

  val Trinum = 33205
  val TriOriginalNum = 16000

  val GlobalSdfRes = 16
  val LocalSdfRes = 4
  val LocalCell = 2048

  val GlobalDdaRes = 16
  val SubDdaRes = 1
  val DdaRes= GlobalDdaRes*SubDdaRes

  val triMemAddrWidth = 32
  // Triangle data width per TriPE request block.
  val triMemDataWidth = triMemNumPEs * 9 * 32
  val triRefIdWidth = 16
  val triRefMemDataWidth = triRefPackFactor * triRefIdWidth

  // Normal memory: address width (triangle index)
  val normalMemAddrWidth = 16
  // Normal data: 3 floats (x, y, z)
  val normalMemDataWidth = 3 * 32  // = 96 bits

  val subgridMetaMemAddrWidth = 32
  val subgridMetaMemTriStartWidth = 22
  val subgridMetaMemTriCountWidth = 10

  // SDF memory: global and local address widths
  val sdfMemAddrWidth = 32           // External interface address width
  val sdfMemGlobalAddrWidth = 12   // 2^12 = 4096 global entries
  val sdfMemLocalAddrWidth = 20    // 2^20 = 1M local entries
  val sdfMemDataWidth = 32         // Single FP32 per access

  // SDF memory internal banking (for synthesis BlackBox)
  val sdfMemBankDepth = 4096
  val sdfMemUramCount = 64
  val sdfMemLocalPerCell = LocalSdfRes * LocalSdfRes * LocalSdfRes
  val sdfMemLocalLaneBits = log2Ceil(sdfMemLocalPerCell)
  val sdfMemLocalGridSize = sdfMemLocalPerCell

  val addrWidth = 32

  // ============================================================
  // SDF PE algorithm parameters
  // ============================================================
  val sdfMaxSteps = 64
  val sdfThreshold1 = 0.01f
  val sdfThreshold2 = 0.02f
  val sdfThreshold3 = 0.04f
  val sdfStepScale = 0.6f
  val sdfMinStep = -0.500f
  val sdfHitAdvance = 1e-3f
  val sdfHitBackoffN = 1


  val ddaMaxSteps =10
  val triRefMemMaxRefs = Trinum

  private def alignUp(value: Int, quantum: Int): Int = {
    require(quantum > 0, s"alignment quantum must be > 0, got $quantum")
    ((value + quantum - 1) / quantum) * quantum
  }

  private val bramDepthAlign = 512

  def triMemDepthFor(numBanks: Int, numPEs: Int = triMemNumPEs): Int = {
    require(numBanks > 0, s"triMem numBanks must be > 0, got $numBanks")
    require(numPEs > 0, s"triMem numPEs must be > 0, got $numPEs")
    val totalDepth = (TriOriginalNum + numPEs - 1) / numPEs
    alignUp((totalDepth + numBanks - 1) / numBanks, bramDepthAlign)
  }

  val triMemDepth = triMemDepthFor(1)
  val triMemBankDepth = triMemDepthFor(triMemNumBanks)
  val triRefMemDepth = alignUp((triRefMemMaxRefs + triRefPackFactor - 1) / triRefPackFactor, bramDepthAlign)
  val normalMemDepth = TriOriginalNum
  val subgridMetaMemDepth =  DdaRes*DdaRes*DdaRes
  val sdfGlobalMemDepth = GlobalSdfRes*GlobalSdfRes*GlobalSdfRes
  val sdfLocalMemDepth = LocalCell
}
case class FloatConfig(
                        expWidth: Int,
                        precision: Int,
                        fmulLatency: Int = GlobalConfig.fmulLatency,
                        faddLatency: Int = GlobalConfig.faddLatency,
                        fcmpLatency: Int = GlobalConfig.fcmpLatency,
                        fptointLatency: Int = GlobalConfig.fptointLatency,
                        fdivLatency: Int = GlobalConfig.fdivLatency,
                        fsqrtLatency: Int = GlobalConfig.fsqrtLatency,
                        useFloatIP: Boolean = GlobalConfig.useFloatIP,
) {
  val totalWidth = expWidth + precision
  val fmacLatency = fmulLatency + faddLatency
  val ffmaLatency = if (useFloatIP) GlobalConfig.fmaLatency else fmulLatency + faddLatency
  val fdotLatency = fmulLatency + faddLatency + faddLatency
  val fcrossLatency = fmulLatency + faddLatency
  val bias = (1 << (expWidth - 1)) - 1
  val maxExp = (1 << expWidth) - 1
  val sigWidth = precision
  val oneHex = "3F800000"
  val oneBigInt = BigInt(oneHex, 16)
  // Alias to GlobalConfig.addrWidth for backward compatibility
  val addrWidth = GlobalConfig.addrWidth
}

object FloatConfig {
  def FP32 = FloatConfig(8, 24)
  def FP16 = FloatConfig(5, 11, fmulLatency = 2, faddLatency = 1)
}

case class TriPeConfig(
  numPEs: Int = GlobalConfig.triMemNumPEs,
  cfg: FloatConfig = FloatConfig.FP32
) {
  val addrWidth = GlobalConfig.addrWidth
}

case class SdfPeConfig(
  cfg: FloatConfig = FloatConfig.FP32,
  GlobalResX: Int = GlobalConfig.GlobalSdfRes,
  GlobalResY: Int = GlobalConfig.GlobalSdfRes,
  GlobalResZ: Int = GlobalConfig.GlobalSdfRes,
  LocalResX: Int = GlobalConfig.LocalSdfRes,
  LocalResY: Int = GlobalConfig.LocalSdfRes,
  LocalResZ: Int = GlobalConfig.LocalSdfRes,
  DDAGlobalRes: Int = GlobalConfig.GlobalDdaRes,
  SubRes: Int = GlobalConfig.SubDdaRes,
  maxSteps: Int = GlobalConfig.sdfMaxSteps,
  DDAMaxSteps: Int = GlobalConfig.ddaMaxSteps,
  threshold3: Float = GlobalConfig.sdfThreshold3,
  threshold2: Float = GlobalConfig.sdfThreshold2,
  threshold1: Float = GlobalConfig.sdfThreshold1,
  StepScale: Float = GlobalConfig.sdfStepScale,
  minStep: Float = GlobalConfig.sdfMinStep,
  hitAdvance: Float = GlobalConfig.sdfHitAdvance,
  hitBackoffN: Int = GlobalConfig.sdfHitBackoffN
) {
  val addrWidth = GlobalConfig.addrWidth
  require(hitBackoffN >= 1, s"hitBackoffN must be >= 1, got $hitBackoffN")
  require(StepScale > 0.0f, s"StepScale must be > 0, got $StepScale")
  val threshold1Bits = java.lang.Float.floatToRawIntBits(threshold1)
  val threshold2Bits = java.lang.Float.floatToRawIntBits(threshold2)
  val threshold3Bits = java.lang.Float.floatToRawIntBits(threshold3)
  val thresholdBits = threshold1Bits
  val stepScaleBits = java.lang.Float.floatToRawIntBits(StepScale)
  val minStepBits = java.lang.Float.floatToRawIntBits(minStep)
  val hitAdvanceBits = java.lang.Float.floatToRawIntBits(hitAdvance)
  val hitBackoffBits = java.lang.Float.floatToRawIntBits(hitAdvance * hitBackoffN.toFloat)
}
