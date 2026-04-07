package raytrace_utils

import chisel3._
import chisel3.util.log2Ceil
object GlobalConfig {
  val commitQueueDepth = 16
  val slotBits  = log2Ceil(commitQueueDepth)
  val useBlackBox = false
  val normalMemDpiLatency = 2
  val triMemDpiLatency = 2
  val sdfMemDpiLatency = 2
  val subgridMemDpiLatency = 2
  val GlobalSdfRes=16
  val LocalSdfRes=4
  val GlobalDdaRes=8
  val SubDdaRes=1

  // Centralized queue depths used across pipeline stages.
  val triBatchQueueDepth = 16
  val sdfWorkQueueDepth = 16
  val sdfRetryQueueDepth = 16
  val sdfFinalQueueDepth = 8
  val simInitToSdfQueueDepth = 16
  val simSdfHitQueueDepth = 16
  //nouse
  val bvhReqQueueDepth = 16
  val bvhLeafQueueDepth = 16
  val bvhMissQueueDepth = 8
}
case class FloatConfig(
                        expWidth: Int,
                        precision: Int,
                        fmulLatency: Int = 8,
                        faddLatency: Int = 7,
                        fcmpLatency: Int = 2,
                        fptointLatency: Int = 6,
                        fdivLatency: Int = 29,
                        useBlackBox: Boolean = GlobalConfig.useBlackBox,
                      ) {
  val totalWidth = expWidth + precision
  val fmacLatency=fmulLatency+faddLatency
  val fdotLatency=fmulLatency+faddLatency+faddLatency
  val fcrossLatency=fmulLatency+faddLatency
  val bias = (1 << (expWidth - 1)) - 1
  val maxExp = (1 << expWidth) - 1
  val sigWidth = precision
  val oneHex = "3F800000"
  val oneBigInt = BigInt(oneHex, 16)
  val addrWidth = 32
}

object FloatConfig {
  // 预定义常用格式（带默认延时）
  // FP32 仅固定位宽，延时使用 FloatConfig 的当前默认参数。
  def FP32 = FloatConfig(8, 24)
  def FP16 = FloatConfig(5, 11, fmulLatency = 2, faddLatency = 1)
}
case class TriPeConfig(
                      numPEs: Int = 4,        // 块大小/PE 数量
                      addrWidth: Int = 32,
                      cfg: FloatConfig = FloatConfig.FP32
                    )

case class BvhPeConfig(
                        addrWidth: Int = 32,
                        stackDepth: Int = 64,
                        reqQueueDepth: Int = GlobalConfig.bvhReqQueueDepth,
                        leafQueueDepth: Int = GlobalConfig.bvhLeafQueueDepth,
                        cfg: FloatConfig = FloatConfig.FP32
                      )

case class SdfPeConfig(
  addrWidth: Int = 32,
  cfg: FloatConfig = FloatConfig.FP32,
  GlobalResX: Int = GlobalConfig.GlobalSdfRes,
  GlobalResY: Int = GlobalConfig.GlobalSdfRes,
  GlobalResZ: Int = GlobalConfig.GlobalSdfRes,
  LocalResX: Int = GlobalConfig.LocalSdfRes,
  LocalResY: Int = GlobalConfig.LocalSdfRes,
  LocalResZ: Int = GlobalConfig.LocalSdfRes,
  DDAGlobalRes: Int  =GlobalConfig.GlobalDdaRes,
  SubRes: Int = GlobalConfig.SubDdaRes,
  maxSteps: Int = 128,
  DDAMaxSteps: Int  =16,
  threshold3: Float = 0.08f,
  threshold2: Float = 0.04f,
  threshold1: Float = 0.02f,
  StepScale: Float  =0.8f,
  minStep: Float = -0.500f,
  hitAdvance: Float = 1e-3f,
  hitBackoffN: Int = 1
) {
  require(hitBackoffN >= 1, s"hitBackoffN must be >= 1, got $hitBackoffN")
  require(StepScale > 0.0f, s"StepScale must be > 0, got $StepScale")
  val threshold1Bits = java.lang.Float.floatToRawIntBits(threshold1)
  val threshold2Bits = java.lang.Float.floatToRawIntBits(threshold2)
  val threshold3Bits = java.lang.Float.floatToRawIntBits(threshold3)
  val thresholdBits  = threshold1Bits
  val stepScaleBits  = java.lang.Float.floatToRawIntBits(StepScale)
  val minStepBits    = java.lang.Float.floatToRawIntBits(minStep)
  val hitAdvanceBits = java.lang.Float.floatToRawIntBits(hitAdvance)
  val hitBackoffBits = java.lang.Float.floatToRawIntBits(hitAdvance * hitBackoffN.toFloat)
}
