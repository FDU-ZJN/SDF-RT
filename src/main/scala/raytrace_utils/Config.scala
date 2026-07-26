package raytrace_utils

import chisel3._
import chisel3.util.log2Ceil

object GlobalConfig {
  val frameWidth =1920
  val frameHeight = 1080
  val traceNumWorkers = 4
  val traceContextsPerWorker = 4
  val traceContextCount = traceNumWorkers * traceContextsPerWorker
  val triMemNumPEs = 1
  // One BRAM36-native 512-deep row per way: 2 triangles (576 bits) per line.
  val triMemCacheLineTriangles = 4
  val triMemNumBanks = 4
  private var useBlackBoxState = 0
  def memImplMode: Int = synchronized { useBlackBoxState }
  def setMemImplMode(value: Int): Unit = synchronized {
    require(value >= 0 && value <= 2, s"memImplMode must be 0/1/2, got $value")
    useBlackBoxState = value
  }
  def withMemImplMode[T](value: Int)(body: => T): T = synchronized {
    require(value >= 0 && value <= 2, s"memImplMode must be 0/1/2, got $value")
    val prev = useBlackBoxState
    useBlackBoxState = value
    try body finally useBlackBoxState = prev
  }

  // Backward-compatible boolean API: false -> mode 0, true -> mode 1
  def useBlackBox: Boolean = synchronized { useBlackBoxState != 0 }
  def setUseBlackBox(value: Boolean): Unit = setMemImplMode(if (value) 1 else 0)
  def withUseBlackBox[T](value: Boolean)(body: => T): T = {
    withMemImplMode(if (value) 1 else 0)(body)
  }

  // The FPGA implementation uses DataMover for triangle-cache refills.  Keep
  // this independently selectable so Verilator can exercise that path while
  // retaining DPI models for the much larger SDF and subgrid memories.
  private var triangleDmaRefillState = false
  def useTriangleDmaRefill: Boolean = synchronized { useBlackBoxState == 2 || triangleDmaRefillState }
  def withTriangleDmaRefill[T](value: Boolean)(body: => T): T = {
    val prev = triangleDmaRefillState
    triangleDmaRefillState = value
    try body finally triangleDmaRefillState = prev
  }

  private var useFloatIPState = false
  def useFloatIP: Boolean = useFloatIPState
  def setUseFloatIP(value: Boolean): Unit = { useFloatIPState = value }
  def withUseFloatIP[T](value: Boolean)(body: => T): T = {
    val prev = useFloatIPState
    useFloatIPState = value
    try body finally useFloatIPState = prev
  }

  val resultArbiterFifoDepth = 2
  val frontendContextCount = 16
  val frontendContextBits = log2Ceil(frontendContextCount)
  val backendContextCount = 16
  val backendContextBits = log2Ceil(backendContextCount)
  require(traceContextCount == backendContextCount,
    "trace context count must cover backend contexts exactly")
  // Compatibility aliases for the existing Trace interfaces.  Backend and
  // frontend IDs are independent even though both are five bits wide.
  val ddaTraceSlotCount = backendContextCount
  val ddaTraceSlotBits = backendContextBits
  val backendContextCmdFifoDepth = 8

  val triMemDpiLatency = 3
  val triMemCacheHitLatency = 1
  val triMemMissPenalty = 60
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
  val SubDdaRes = 4
  val DdaRes= GlobalDdaRes*SubDdaRes
  // Signed range must include the two traversal sentinels -1 and DdaRes.
  val ddaCoordWidth = log2Ceil(DdaRes + 1) + 1

  val triMemAddrWidth = 32
  // Triangle data width per TriPE request block.
  val triMemDataWidth = triMemCacheLineTriangles * 9 * 32
  val triMemCacheSets = 512
  val triMemCacheWays = 1
  val triMemReqQueueDepth = 8
  val triMemMergeQueueDepth = 8
  val triMemMshrEntries = 4

  // Normal memory: address width (triangle index)
  val normalMemAddrWidth = 16
  // Normal data: 3 floats (x, y, z)
  val normalMemDataWidth = 3 * 32  // = 96 bits

  val subgridMetaMemAddrWidth = 32
  val subgridMetaMemTriStartWidth = 22
  val subgridMetaMemTriCountWidth = 10
  val subgridBlockSize = 32
  val subgridBlockShift = 5
  val subgridCompactMetaDepth = 12288

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
  val sdfMaxSteps = 32
  val ddaMaxSteps = 64
  // Conservative handoff to DDA: the scene sweep shows full coverage with no
  // surface overshoot at this threshold for all three SDF hierarchy levels.
  val sdfThreshold1 = 0.01f
  val sdfThreshold2 = 0.02f
  val sdfThreshold3 = 0.04f
  val sdfStepScale = 0.6f

  private def alignUp(value: Int, quantum: Int): Int = {
    require(quantum > 0, s"alignment quantum must be > 0, got $quantum")
    ((value + quantum - 1) / quantum) * quantum
  }

  private val bramDepthAlign = 512

  def triMemDepthFor(numBanks: Int, lineTriangles: Int = triMemCacheLineTriangles): Int = {
    require(numBanks > 0, s"triMem numBanks must be > 0, got $numBanks")
    require(lineTriangles > 0, s"triMem cache line must contain triangles, got $lineTriangles")
    val totalDepth = (TriOriginalNum + lineTriangles - 1) / lineTriangles
    alignUp((totalDepth + numBanks - 1) / numBanks, bramDepthAlign)
  }

  val triMemDepth = triMemDepthFor(1)
  val triMemBankDepth = triMemDepthFor(triMemNumBanks)
  val normalMemDepth = Trinum
  val subgridMetaMemDepth =  DdaRes*DdaRes*DdaRes
  val subgridBlockDirDepth = (subgridMetaMemDepth + subgridBlockSize - 1) / subgridBlockSize
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
  cacheLineTriangles: Int = GlobalConfig.triMemCacheLineTriangles,
  cfg: FloatConfig = FloatConfig.FP32
) {
  require(numPEs > 0, s"TriPE intersector count must be > 0, got $numPEs")
  require(cacheLineTriangles > 0 && (cacheLineTriangles & (cacheLineTriangles - 1)) == 0,
    s"TriPE cache line must contain a power-of-two number of triangles, got $cacheLineTriangles")
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
  StepScale: Float = GlobalConfig.sdfStepScale
) {
  val addrWidth = GlobalConfig.addrWidth
  require(StepScale > 0.0f, s"StepScale must be > 0, got $StepScale")
  val threshold1Bits = java.lang.Float.floatToRawIntBits(threshold1)
  val threshold2Bits = java.lang.Float.floatToRawIntBits(threshold2)
  val threshold3Bits = java.lang.Float.floatToRawIntBits(threshold3)
  val thresholdBits = threshold1Bits
  val stepScaleBits = java.lang.Float.floatToRawIntBits(StepScale)
}
