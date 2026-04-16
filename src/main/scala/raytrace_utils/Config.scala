package raytrace_utils

import chisel3._
import chisel3.util.log2Ceil

object GlobalConfig {
  // ============================================================
  // FPGA top / frame config
  // ============================================================
  val frameWidth = 100
  val frameHeight = 100
  val pixelQueueDepth = 4
  val rayDirFifoDepth = 16
  // ============================================================
  // BlackBox vs DPI mode switch (memory modules)
  // ============================================================
  // 0: DPI-C memory
  // 1: readmemh BlackBox memory
  // 2: synthesized/IP-style memory modules
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

  // ============================================================
  // Float IP BlackBox switch (fudian modules: FADD, FMUL, FCMP, etc.)
  // Used for FPGA IP integration, disabled in Verilator simulation
  // ============================================================
  private var useFloatIPState = false
  def useFloatIP: Boolean = useFloatIPState
  def setUseFloatIP(value: Boolean): Unit = { useFloatIPState = value }
  def withUseFloatIP[T](value: Boolean)(body: => T): T = {
    val prev = useFloatIPState
    useFloatIPState = value
    try body finally useFloatIPState = prev
  }

  // ============================================================
  // Queue depths (centralized)
  // ============================================================
  val commitQueueDepth = 32
  val slotBits = log2Ceil(commitQueueDepth)

  val triBatchQueueDepth = 16
  val sdfWorkQueueDepth = 16
  val sdfRetryQueueDepth = 16
  val sdfFinalQueueDepth = 8
  val simInitToSdfQueueDepth = 16
  val simSdfHitQueueDepth = 32

  // Unused / reserved for future
  val bvhReqQueueDepth = 16
  val bvhLeafQueueDepth = 16
  val bvhMissQueueDepth = 8

  // ============================================================
  // Memory latency (pipeline depth for DPI / BlackBox)
  // ============================================================
  val normalMemDpiLatency = 4
  val triMemDpiLatency = 2
  val sdfMemDpiLatency = 3
  val subgridMemDpiLatency = 1
  val fsqrtLatency = 29

  // ============================================================
  // Grid resolutions
  // ============================================================
  val Trinum = 14204
  // SDF PE grid
  val GlobalSdfRes = 16
  val LocalSdfRes = 4
  val LocalCell = 2048
  // DDA grid
  val GlobalDdaRes = 8
  val SubDdaRes = 1
  val DdaRes= GlobalDdaRes*SubDdaRes
  // ============================================================
  // Memory address widths (key interfaces)
  // ============================================================
  // Triangle memory: address width for compact triangle storage
  val triMemAddrWidth = 32
  // Triangle data: numPEs * 9 floats per batch
  val triMemNumPEs = 4
  val triMemDataWidth = triMemNumPEs * 9 * 32  // = 1152 bits

  // Normal memory: address width (triangle index)
  val normalMemAddrWidth = 16
  // Normal data: 3 floats (x, y, z)
  val normalMemDataWidth = 3 * 32  // = 96 bits

  // Subgrid meta memory: combined address = globalIdx[Global_ADDR_WIDTH] + subIdx[SUB_ADDR_WIDTH]
  val subgridMetaMemAddrWidth = 32
  val subgridMetaMemTriStartWidth = 16
  val subgridMetaMemTriCountWidth = 16

  // SDF memory: global and local address widths
  val sdfMemAddrWidth = 32           // External interface address width
  val sdfMemGlobalAddrWidth = 12   // 2^12 = 4096 global entries
  val sdfMemLocalAddrWidth = 20    // 2^20 = 1M local entries
  val sdfMemDataWidth = 32         // Single FP32 per access

  // SDF memory internal banking (for synthesis BlackBox)
  val sdfMemBankDepth = 4096
  val sdfMemUramCount = 64
  val sdfMemLocalGridSize = 64

  // BVH memory (unused currently, but configured for future)
  val bvhMemAddrWidth = 32
  val bvhMemNodeBytes = 32  // 6 floats bounds + 4 int32 node info
  val bvhMemDataWidth = bvhMemNodeBytes * 8  // = 256 bits

  // ============================================================
  // Global address width (used across all modules)
  // ============================================================
  val addrWidth = 32

  // ============================================================
  // SDF PE algorithm parameters
  // ============================================================
  val sdfMaxSteps = 64
  val sdfThreshold1 = 0.01f
  val sdfThreshold2 = 0.02f
  val sdfThreshold3 = 0.04f
  val sdfStepScale = 0.7f
  val sdfMinStep = -0.500f
  val sdfHitAdvance = 1e-3f
  val sdfHitBackoffN = 1


  val ddaMaxSteps = 8

  val triMemDepth = Trinum/4
  val normalMemDepth = Trinum
  val subgridMetaMemDepth =  DdaRes*DdaRes*DdaRes
  val sdfGlobalMemDepth = GlobalSdfRes*GlobalSdfRes*GlobalSdfRes
  val sdfLocalMemDepth = LocalCell
  val bvhMemDepth = 65536          // nouse
}
case class FloatConfig(
                        expWidth: Int,
                        precision: Int,
                        fmulLatency: Int = 8,
                        faddLatency: Int = 8,
                        fcmpLatency: Int = 2,
                        fptointLatency: Int = 1,
                        fdivLatency: Int = 29,
                        fsqrtLatency: Int = GlobalConfig.fsqrtLatency,
                        useFloatIP: Boolean = GlobalConfig.useFloatIP,
                      ) {
  val totalWidth = expWidth + precision
  val fmacLatency = fmulLatency + faddLatency
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

case class BvhPeConfig(
  stackDepth: Int = 64,
  reqQueueDepth: Int = GlobalConfig.bvhReqQueueDepth,
  leafQueueDepth: Int = GlobalConfig.bvhLeafQueueDepth,
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
