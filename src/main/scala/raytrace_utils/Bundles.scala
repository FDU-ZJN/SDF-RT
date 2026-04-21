package raytrace_utils

import chisel3._

class Vec3(cfg: FloatConfig = FloatConfig.FP32) extends Bundle {
  val x = UInt(cfg.totalWidth.W)
  val y = UInt(cfg.totalWidth.W)
  val z = UInt(cfg.totalWidth.W)
}

// Axis-aligned bounding box in FP format.
class AABB(cfg: FloatConfig = FloatConfig.FP32) extends Bundle {
  val min = new Vec3(cfg)
  val max = new Vec3(cfg)
}

class Ray(cfg: FloatConfig = FloatConfig.FP32) extends Bundle {
  val origin = new Vec3(cfg)
  val dir = new Vec3(cfg)
}

class RayMeta(addrWidth: Int = 32, pixelWidth: Int = 16) extends Bundle {
  val slotId = UInt(addrWidth.W)
  val pixelX = UInt(pixelWidth.W)
  val pixelY = UInt(pixelWidth.W)
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

class TraceResult(cfg: FloatConfig = FloatConfig.FP32, addrWidth: Int = 32) extends Bundle {
  val meta = new RayMeta(addrWidth)
  val hit = Bool()
  val hitId = UInt(addrWidth.W)
  val hitT = UInt(cfg.totalWidth.W)
}

class RenderResult(cfg: FloatConfig = FloatConfig.FP32, addrWidth: Int = 32) extends Bundle {
  val meta = new RayMeta(addrWidth)
  val hit = Bool()
  val hitId = UInt(addrWidth.W)
  val rgb8 = UInt(24.W)
}

class BvhNode(cfg: FloatConfig = FloatConfig.FP32, addrWidth: Int = 32) extends Bundle {
  val bounds = new AABB(cfg)
  val isLeaf = Bool()
  val leftValid = Bool()
  val rightValid = Bool()
  val left = UInt(addrWidth.W)
  val right = UInt(addrWidth.W)
  val triStart = UInt(addrWidth.W)
  val triCount = UInt(16.W)
}

class BvhStartReq(cfg: FloatConfig = FloatConfig.FP32) extends Bundle {
  val ray = new Ray(cfg)
  val meta = new RayMeta(GlobalConfig.slotBits)
  val rootNode = UInt(cfg.addrWidth.W)
}

class RayIssue(cfg: FloatConfig = FloatConfig.FP32, addrWidth: Int = 32) extends Bundle {
  val ray = new Ray(cfg)
  val meta = new RayMeta(addrWidth)
}

class TraceHitUpdate(cfg: FloatConfig = FloatConfig.FP32, addrWidth: Int = 32) extends Bundle {
  val hit = Bool()
  val hitId = UInt(addrWidth.W)
  val hitT = UInt(cfg.totalWidth.W)
}

class SdfRayReq(cfg: FloatConfig = FloatConfig.FP32, addrWidth: Int = 32) extends Bundle {
  val ray = new Ray(cfg)
  val meta = new RayMeta(addrWidth)
  val iter = UInt(16.W)
}

class SdfRayResp(cfg: FloatConfig = FloatConfig.FP32, addrWidth: Int = 32) extends Bundle {
  val ray = new Ray(cfg)
  val meta = new RayMeta(addrWidth)
  val hit = Bool()
  val iter = UInt(16.W)
  val reverseTraversal = Bool()
}

class SdfMemReq(addrWidth: Int = 32) extends Bundle {
  val globalIdx = UInt(addrWidth.W)
  val localIdx = UInt(addrWidth.W)
}

class SdfInitReq(cfg: FloatConfig = FloatConfig.FP32, addrWidth: Int = 32) extends Bundle {
  val rd = new Vec3(cfg)
  val meta = new RayMeta(addrWidth)
}

class SdfBypassResp(addrWidth: Int = 32) extends Bundle {
  val meta = new RayMeta(addrWidth)
}

class DdaTraversalReq(cfg: FloatConfig = FloatConfig.FP32, addrWidth: Int = 32) extends Bundle {
  val ray = new Ray(cfg)
  val meta = new RayMeta(addrWidth)
  val reverseTraversal = Bool()
}

class DdaSubgridMeta(addrWidth: Int = 32) extends Bundle {
  val triStart = UInt(addrWidth.W)
  val triCount = UInt(16.W)
}

class DdaTraversalResult(cfg: FloatConfig = FloatConfig.FP32, addrWidth: Int = 32) extends Bundle {
  val meta = new RayMeta(addrWidth)
  val hit = Bool()
  val hitId = UInt(addrWidth.W)
  val hitT = UInt(cfg.totalWidth.W)
}

class DdaTraceCmd(cfg: FloatConfig = FloatConfig.FP32, addrWidth: Int = 32) extends Bundle {
  val ray = new Ray(cfg)
  val meta = new RayMeta(addrWidth)
  val tri = new TriBatch(addrWidth)
  val subX = SInt((addrWidth + 1).W)
  val subY = SInt((addrWidth + 1).W)
  val subZ = SInt((addrWidth + 1).W)
  val iter = UInt(16.W)
}

class DdaTraceRsp(cfg: FloatConfig = FloatConfig.FP32, addrWidth: Int = 32) extends Bundle {
  val ray = new Ray(cfg)
  val meta = new RayMeta(addrWidth)
  val hit = Bool()
  val hitId = UInt(addrWidth.W)
  val hitT = UInt(cfg.totalWidth.W)
  val subX = SInt((addrWidth + 1).W)
  val subY = SInt((addrWidth + 1).W)
  val subZ = SInt((addrWidth + 1).W)
  val iter = UInt(16.W)
}
