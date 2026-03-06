package raytrace_utils

import chisel3._

class Vec3(cfg: FloatConfig = FloatConfig.FP32) extends Bundle {
  val x = UInt(cfg.totalWidth.W)
  val y = UInt(cfg.totalWidth.W)
  val z = UInt(cfg.totalWidth.W)
}

class Ray(cfg: FloatConfig = FloatConfig.FP32) extends Bundle {
  val origin = new Vec3(cfg)
  val dir = new Vec3(cfg)
}

class Triangle(cfg: FloatConfig = FloatConfig.FP32) extends Bundle {
  val v0 = new Vec3(cfg)
  val v1 = new Vec3(cfg)
  val v2 = new Vec3(cfg)
  val id = UInt(cfg.addrWidth.W)
}

class TriangleBlock(val c: TriPeConfig) extends Bundle {
  val tris = Vec(c.numPEs, new Triangle(c.cfg)) // 一个块里的多个三角形
  val mask = Vec(c.numPEs, Bool())                   // 哪些三角形是有效的
}
class TriBatch(addrWidth: Int) extends Bundle {
  val base_addr = UInt(addrWidth.W)
  val count     = UInt(16.W)
}