package raytrace_utils

import chisel3._

object SimpleFPCompare {
  def isZero(x: UInt, totalWidth: Int): Bool = x(totalWidth - 2, 0) === 0.U

  def ltZero(x: UInt, totalWidth: Int): Bool =
    x(totalWidth - 1) && !isZero(x, totalWidth)

  def gtZero(x: UInt, totalWidth: Int): Bool =
    !x(totalWidth - 1) && !isZero(x, totalWidth)

  // For non-negative IEEE754 values, unsigned bit ordering matches numeric ordering.
  def nonNegLt(a: UInt, b: UInt): Bool = a < b
  def nonNegLe(a: UInt, b: UInt): Bool = a <= b

  def lePositiveConst(x: UInt, c: UInt, totalWidth: Int): Bool =
    ltZero(x, totalWidth) || (!x(totalWidth - 1) && x <= c)

  def gtPositiveConst(x: UInt, c: UInt, totalWidth: Int): Bool =
    !x(totalWidth - 1) && x > c
}
