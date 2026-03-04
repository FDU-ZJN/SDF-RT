package raytrace_utils

import chisel3._
import chisel3.util._
class Vec3(val len: Int = 32) extends Bundle {
  val x = UInt(len.W)
  val y = UInt(len.W)
  val z = UInt(len.W)
}

class Ray(len: Int = 32) extends Bundle {
  val origin = new Vec3(len)
  val dir = new Vec3(len)
}

class Triangle(len: Int = 32) extends Bundle {
  val v0 = new Vec3(len)
  val v1 = new Vec3(len)
  val v2 = new Vec3(len)
}