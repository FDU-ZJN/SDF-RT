package raytrace_utils

case class FloatConfig(
                        expWidth: Int,
                        precision: Int,
                        fmulLatency: Int = 3,
                        faddLatency: Int = 2,
                        fdivLatency: Int = 6
                      ) {
  val totalWidth = expWidth + precision
  val fmacLatency=fmulLatency+faddLatency
  val bias = (1 << (expWidth - 1)) - 1
  val maxExp = (1 << expWidth) - 1
  val sigWidth = precision
  val oneHex = "3F800000"
  val oneBigInt = BigInt(oneHex, 16)
}

object FloatConfig {
  // 预定义常用格式（带默认延时）
  def FP32 = FloatConfig(8, 24, fmulLatency = 3, faddLatency = 2)
  def FP16 = FloatConfig(5, 11, fmulLatency = 2, faddLatency = 1)
}
