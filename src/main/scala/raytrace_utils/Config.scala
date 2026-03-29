package raytrace_utils

import chisel3._
import chisel3.util.log2Ceil
object GlobalConfig {
  val commitQueueDepth = 8
  val slotBits  = log2Ceil(commitQueueDepth)
}
case class FloatConfig(
                        expWidth: Int,
                        precision: Int,
                        fmulLatency: Int = 3,
                        faddLatency: Int = 2,
                        fdivLatency: Int = 6
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
  def FP32 = FloatConfig(8, 24, fmulLatency = 3, faddLatency = 2)
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
                        reqQueueDepth: Int = 16,
                        leafQueueDepth: Int = 16,
                        cfg: FloatConfig = FloatConfig.FP32
                      )

case class SdfPeConfig(
  addrWidth: Int = 32,
  cfg: FloatConfig = FloatConfig.FP32,
  GlobalResX: Int = 16,
  GlobalResY: Int = 16,
  GlobalResZ: Int = 16,
  LocalResX: Int = 16,
  LocalResY: Int = 16,
  LocalResZ: Int = 16,
  maxSteps: Int = 30,
  threshold: Float = 0.04f,
  minStep: Float = 0.000f,
  hitAdvance: Float = 1e-6f
) {
  val thresholdBits  = java.lang.Float.floatToRawIntBits(threshold)
  val minStepBits    = java.lang.Float.floatToRawIntBits(minStep)
  val hitAdvanceBits = java.lang.Float.floatToRawIntBits(hitAdvance)
}
