package raytrace_utils

import chisel3._
import chisel3.util.log2Ceil

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
  val dist = UInt(cfg.totalWidth.W)
}

class RayMeta(addrWidth: Int = 32) extends Bundle {
  val rayId = UInt(16.W)
}

class RayIngress(cfg: FloatConfig = FloatConfig.FP32) extends Bundle {
  val origin = new Vec3(cfg)
  val dir = new Vec3(cfg)
  val rayId = UInt(16.W)
}

class Triangle(cfg: FloatConfig = FloatConfig.FP32) extends Bundle {
  val v0 = new Vec3(cfg)
  val e1 = new Vec3(cfg)
  val e2 = new Vec3(cfg)
  val id = UInt(cfg.addrWidth.W)
}

class TriangleBlock(val c: TriPeConfig) extends Bundle {
  val tris = Vec(c.cacheLineTriangles, new Triangle(c.cfg))
  val mask = Vec(c.cacheLineTriangles, Bool())
}
class TriBatch(addrWidth: Int) extends Bundle {
  require(addrWidth >= GlobalConfig.subgridMetaMemTriStartWidth)
  val base_addr = UInt(GlobalConfig.subgridMetaMemTriStartWidth.W)
  val count     = UInt(GlobalConfig.subgridMetaMemTriCountWidth.W)
}

class TraceResult(cfg: FloatConfig = FloatConfig.FP32, addrWidth: Int = 32) extends Bundle {
  val meta = new RayMeta(addrWidth)
  val hit = Bool()
  val hitId = UInt(addrWidth.W)
  val hitT = UInt(cfg.totalWidth.W)
}

class TraceResultWithSlot(cfg: FloatConfig = FloatConfig.FP32, addrWidth: Int = 32) extends Bundle {
  val slotIdx = UInt(GlobalConfig.ddaTraceSlotBits.W)
  val result = new TraceResult(cfg, addrWidth)
}

class RenderResult(cfg: FloatConfig = FloatConfig.FP32, addrWidth: Int = 32) extends Bundle {
  val meta = new RayMeta(addrWidth)
  val hit = Bool()
  val hitId = UInt(addrWidth.W)
  val hitT = UInt(cfg.totalWidth.W)
  val rgb8 = UInt(24.W)
}

class RayIssue(cfg: FloatConfig = FloatConfig.FP32, addrWidth: Int = 32) extends Bundle {
  val ray = new Ray(cfg)
  val meta = new RayMeta(addrWidth)
}

class SdfRayReq(cfg: FloatConfig = FloatConfig.FP32, addrWidth: Int = 32) extends Bundle {
  val ray = new Ray(cfg)
  val meta = new RayMeta(addrWidth)
  val iter = UInt(16.W)
  val prevSdf = UInt(cfg.totalWidth.W)
}

class SdfRayResp(cfg: FloatConfig = FloatConfig.FP32, addrWidth: Int = 32) extends Bundle {
  val ray = new Ray(cfg)
  val meta = new RayMeta(addrWidth)
  val hit = Bool()
  val iter = UInt(16.W)
  val prevSdf = UInt(cfg.totalWidth.W)
}

class SdfMemReq(addrWidth: Int = 32) extends Bundle {
  val globalIdx = UInt(addrWidth.W)
  val localIdx = UInt(addrWidth.W)
}

class InitStageResp(cfg: FloatConfig = FloatConfig.FP32, addrWidth: Int = 32) extends Bundle {
  val hit = Bool()
  val ray = new Ray(cfg)
  val meta = new RayMeta(addrWidth)
}

class DdaTraversalReq(cfg: FloatConfig = FloatConfig.FP32, addrWidth: Int = 32) extends Bundle {
  val ray = new Ray(cfg)
  val meta = new RayMeta(addrWidth)
  val traceSlot = UInt(GlobalConfig.ddaTraceSlotBits.W)
}

class DdaContext(cfg: FloatConfig = FloatConfig.FP32, addrWidth: Int = 32) extends Bundle {
  val ray = new Ray(cfg)
  val meta = new RayMeta(addrWidth)
  val traceSlot = UInt(GlobalConfig.ddaTraceSlotBits.W)
  val initialized = Bool()
  val subX = SInt(GlobalConfig.ddaCoordWidth.W)
  val subY = SInt(GlobalConfig.ddaCoordWidth.W)
  val subZ = SInt(GlobalConfig.ddaCoordWidth.W)
  val iter = UInt(16.W)
  val tMaxX = UInt(cfg.totalWidth.W)
  val tMaxY = UInt(cfg.totalWidth.W)
  val tMaxZ = UInt(cfg.totalWidth.W)
  val tDeltaX = UInt(cfg.totalWidth.W)
  val tDeltaY = UInt(cfg.totalWidth.W)
  val tDeltaZ = UInt(cfg.totalWidth.W)
}

class DdaSubgridMeta(addrWidth: Int = 32) extends Bundle {
  val triStart = UInt(GlobalConfig.subgridMetaMemTriStartWidth.W)
  val triCount = UInt(GlobalConfig.subgridMetaMemTriCountWidth.W)
}

class DdaSubgridMetaReq(addrWidth: Int = 32) extends Bundle {
  val globalIdx = UInt(addrWidth.W)
  val subIdx = UInt(addrWidth.W)
}

class DdaSubgridMetaResp extends Bundle {
  val triStart = UInt(GlobalConfig.subgridMetaMemTriStartWidth.W)
  val triCount = UInt(GlobalConfig.subgridMetaMemTriCountWidth.W)
}

class DdaStepResult(cfg: FloatConfig = FloatConfig.FP32, addrWidth: Int = 32) extends Bundle {
  val ctx = new DdaContext(cfg, addrWidth)
  val done = Bool()
  val watchdog = Bool()
  val emitCmd = Bool()
  val tri = new TriBatch(addrWidth)
  val cellExitT = UInt(cfg.totalWidth.W)
}

class TraceCellCmd(cfg: FloatConfig = FloatConfig.FP32, addrWidth: Int = 32) extends Bundle {
  val tri = new TriBatch(addrWidth)
  val cellExitT = UInt(cfg.totalWidth.W)
}

class TraceContextDesc(cfg: FloatConfig = FloatConfig.FP32, addrWidth: Int = 32) extends Bundle {
  val contextId = UInt(GlobalConfig.ddaTraceSlotBits.W)
  val ray = new Ray(cfg)
  val meta = new RayMeta(addrWidth)
}
