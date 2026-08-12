package raytrace_utils

import chisel3._
import chisel3.util.log2Ceil

object GlobalConfig {
  val rayIdWidth = 21
  val frameWidth =100
  val frameHeight = 100
  val traceNumWorkers = 4
  // TraceController owns backend contexts statically as four slots per
  // worker.  Keep this in lockstep with backendContextCount (4 * 4 = 16).
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
  val sdfMemDpiLatency = 4
  val subgridMemDpiLatency = 3
  val fsqrtLatency = 11
  val fmulLatency = 4
  val faddLatency = 5
  val fmaLatency = 8
  val fcmpLatency = 1
  val fptointLatency = 1
  val fdivLatency = 10

  val Trinum = 33205
  val TriOriginalNum = 16000
  // Compact triangle indices are the architectural triangle address space.
  // Keep all triangle IDs, batch bases, cache block addresses, and normal
  // addresses derived from this single capacity contract.
  val maxCompactTriangles = 131072
  val triangleAddrWidth = log2Ceil(maxCompactTriangles)
  require(maxCompactTriangles == (1 << triangleAddrWidth),
    "maxCompactTriangles must be a power of two")

  val GlobalSdfRes = 32
  val LocalSdfRes = 4
  // The 32^3 hybrid SDF cache for Killeroo currently activates 16059 cells.
  // Mapping stores a 14-bit cell index, so retain the full 16384-entry range.
  val LocalCell = 16384

  val GlobalDdaRes = 16
  val SubDdaRes = 4
  val DdaRes= GlobalDdaRes*SubDdaRes
  // Signed range must include the two traversal sentinels -1 and DdaRes.
  val ddaCoordWidth = log2Ceil(DdaRes + 1) + 1

  // A cache request addresses a compact triangle cache line, not an
  // individual triangle.  Four triangles share a line, so 131072 compact
  // triangles require 32768 line addresses (15 bits).  Keep the full
  // 17-bit triangleAddrWidth for architectural triangle IDs.
  val triMemBlockAddrWidth = log2Ceil(maxCompactTriangles / triMemCacheLineTriangles)
  val triMemAddrWidth = triMemBlockAddrWidth
  def triMemBankAddrWidthFor(numBanks: Int): Int = {
    require(numBanks > 0 && (numBanks & (numBanks - 1)) == 0,
      s"triangle memory bank count must be a power of two, got $numBanks")
    triMemAddrWidth - log2Ceil(numBanks)
  }
  val triMemBankAddrWidth = triMemBankAddrWidthFor(triMemNumBanks)
  require(triMemBankAddrWidth == 13,
    s"four-bank triangle cache requires 13-bit bank-local addresses, got $triMemBankAddrWidth")
  // Triangle data width per TriPE request block.
  val triMemDataWidth = triMemCacheLineTriangles * 9 * 32
  val triMemCacheSets = 512
  val triMemCacheWays = 1
  val triMemReqQueueDepth = 8
  val triMemMergeQueueDepth = 8
  val triMemMshrEntries = 4

  // Normal memory: address width (triangle index)
  val normalMemAddrWidth = triangleAddrWidth
  // Normal data: 3 floats (x, y, z)
  val normalMemDataWidth = 3 * 32  // = 96 bits

  val subgridMetaMemAddrWidth = 32
  val subgridMetaMemTriStartWidth = triangleAddrWidth
  val subgridMetaMemTriCountWidth = 10
  val subgridBlockSize = 32
  val subgridBlockShift = 5
  val subgridCompactMetaDepth = 12288

  // SDF memory: global and local address widths
  val sdfMemAddrWidth = 32           // External interface address width
  val sdfMemGlobalAddrWidth = 15   // 32^3 = 32768 global entries
  val sdfMemLocalAddrWidth = 20    // 2^20 = 1M local entries
  // SDF samples are stored as custom 9-bit E2M6 (bias 5, subnormal support).
  // Values above the finite range saturate toward zero.  The read-side
  // codec expands them back to FP32 before they enter SdfPE.
  val sdfMemStoredDataWidth = 9
  val sdfMemDataWidth = 32         // FP32 after the SDF-memory read codec
  val sdfMemGlobalRowSamples = 8
  val sdfMemLocalRowSamples = 64
  val sdfMemGlobalWriteStrideWords = 4
  val sdfMemLocalWriteStrideWords = 32
  val sdfMemLocalWriteBase = 0x10000
  val sdfMemMappingWriteBase = 0xC0000

  // SDF memory internal banking (for synthesis BlackBox)
  val sdfMemBankDepth = 4096
  val sdfMemUramCount = 64
  val sdfMemLocalPerCell = LocalSdfRes * LocalSdfRes * LocalSdfRes
  val sdfMemLocalLaneBits = log2Ceil(sdfMemLocalPerCell)
  val sdfMemLocalGridSize = sdfMemLocalPerCell

  val addrWidth = triangleAddrWidth

  // ============================================================
  // SDF PE algorithm parameters
  // ============================================================
  val sdfMaxSteps = 32
  val ddaMaxSteps = 64
  val sdfThreshold1 = 0.01f
  val sdfThreshold2 = 0.02f
  val sdfThreshold3 = 0.04f

  private def alignUp(value: Int, quantum: Int): Int = {
    require(quantum > 0, s"alignment quantum must be > 0, got $quantum")
    ((value + quantum - 1) / quantum) * quantum
  }

  private val bramDepthAlign = 512

  def triMemDepthFor(numBanks: Int, lineTriangles: Int = triMemCacheLineTriangles): Int = {
    require(numBanks > 0, s"triMem numBanks must be > 0, got $numBanks")
    require(lineTriangles > 0, s"triMem cache line must contain triangles, got $lineTriangles")
    val totalDepth = (maxCompactTriangles + lineTriangles - 1) / lineTriangles
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
  nearSurfaceBackoffThreshold: Float = 0.005f,
) {
  val addrWidth = GlobalConfig.addrWidth
  val threshold1Bits = java.lang.Float.floatToRawIntBits(threshold1)
  val threshold2Bits = java.lang.Float.floatToRawIntBits(threshold2)
  val threshold3Bits = java.lang.Float.floatToRawIntBits(threshold3)
  val nearSurfaceBackoffThresholdBits =
    java.lang.Float.floatToRawIntBits(nearSurfaceBackoffThreshold)
  val thresholdBits = threshold1Bits
}
