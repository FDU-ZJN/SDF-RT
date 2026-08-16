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
  val dist = UInt(cfg.totalWidth.W)
}

class RayMeta(addrWidth: Int = 32) extends Bundle {
  val rayId = UInt(GlobalConfig.rayIdWidth.W)
}

class RayIngress(cfg: FloatConfig = FloatConfig.FP32) extends Bundle {
  val origin = new Vec3(cfg)
  val dir = new Vec3(cfg)
  val rayId = UInt(GlobalConfig.rayIdWidth.W)
}

class ImmutableRayRecord(cfg: FloatConfig = FloatConfig.FP32, addrWidth: Int = 32) extends Bundle {
  val origin = new Vec3(cfg)
  val dir = new Vec3(cfg)
  val meta = new RayMeta(addrWidth)
}

class FrontendSdfState(cfg: FloatConfig = FloatConfig.FP32) extends Bundle {
  val dist = UInt(cfg.totalWidth.W)
  val iter = UInt(16.W)
  val prevSdf = UInt(cfg.totalWidth.W)
}

class FrontendStateWrite(cfg: FloatConfig = FloatConfig.FP32) extends Bundle {
  val contextId = UInt(GlobalConfig.frontendContextBits.W)
  val state = new FrontendSdfState(cfg)
}

class FrontendInitResp(cfg: FloatConfig = FloatConfig.FP32) extends Bundle {
  val contextId = UInt(GlobalConfig.frontendContextBits.W)
  val hit = Bool()
  val entryT = UInt(cfg.totalWidth.W)
  val transformedDir = new Vec3(cfg)
}

class FrontendSdfResult(cfg: FloatConfig = FloatConfig.FP32) extends Bundle {
  val contextId = UInt(GlobalConfig.frontendContextBits.W)
  val hit = Bool()
  val dist = UInt(cfg.totalWidth.W)
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
  val rayId = UInt(GlobalConfig.rayIdWidth.W)
  val hit = Bool()
  val hitId = UInt(addrWidth.W)
  val hitT = UInt(cfg.totalWidth.W)
  val hitU = UInt(cfg.totalWidth.W)
  val hitV = UInt(cfg.totalWidth.W)
}

class RenderResult(cfg: FloatConfig = FloatConfig.FP32, addrWidth: Int = 32) extends Bundle {
  val rayId = UInt(GlobalConfig.rayIdWidth.W)
  val hit = Bool()
  val hitId = UInt(addrWidth.W)
  val hitT = UInt(cfg.totalWidth.W)
  val hitU = UInt(16.W)
  val hitV = UInt(16.W)
}

class FloatRenderResult(cfg: FloatConfig = FloatConfig.FP32, addrWidth: Int = 32) extends Bundle {
  val rayId = UInt(GlobalConfig.rayIdWidth.W)
  val hit = Bool()
  val hitId = UInt(addrWidth.W)
  val hitT = UInt(cfg.totalWidth.W)
  val hitU = UInt(cfg.totalWidth.W)
  val hitV = UInt(cfg.totalWidth.W)
}

class SdfStepReq(cfg: FloatConfig = FloatConfig.FP32) extends Bundle {
  val contextId = UInt(GlobalConfig.frontendContextBits.W)
  val origin = new Vec3(cfg)
  val dir = new Vec3(cfg)
  val state = new FrontendSdfState(cfg)
}

class SdfStepResp(cfg: FloatConfig = FloatConfig.FP32) extends Bundle {
  val contextId = UInt(GlobalConfig.frontendContextBits.W)
  val state = new FrontendSdfState(cfg)
  val done = Bool()
  val hit = Bool()
}

class SdfMemReq(addrWidth: Int = 32) extends Bundle {
  val globalIdx = UInt(addrWidth.W)
  val localIdx = UInt(addrWidth.W)
}

class DdaTraversalReq(cfg: FloatConfig = FloatConfig.FP32, addrWidth: Int = 32) extends Bundle {
  val frontendContextId = UInt(GlobalConfig.frontendContextBits.W)
  val ray = new Ray(cfg)
  val meta = new RayMeta(addrWidth)
  val traceSlot = UInt(GlobalConfig.ddaTraceSlotBits.W)
}

class DdaTraversalState(cfg: FloatConfig = FloatConfig.FP32) extends Bundle {
  val contextId = UInt(GlobalConfig.backendContextBits.W)
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
  val dirNegX = Bool()
  val dirNegY = Bool()
  val dirNegZ = Bool()
}

class BackendContextInit(cfg: FloatConfig = FloatConfig.FP32, addrWidth: Int = 32) extends Bundle {
  val sourceFrontendId = UInt(GlobalConfig.frontendContextBits.W)
  val contextId = UInt(GlobalConfig.backendContextBits.W)
  val immutable = new ImmutableRayRecord(cfg, addrWidth)
  val traversal = new DdaTraversalState(cfg)
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
  val state = new DdaTraversalState(cfg)
  val done = Bool()
  val emitCmd = Bool()
  val tri = new TriBatch(addrWidth)
}

class TraceContextDesc(cfg: FloatConfig = FloatConfig.FP32, addrWidth: Int = 32) extends Bundle {
  val contextId = UInt(GlobalConfig.ddaTraceSlotBits.W)
  val ray = new Ray(cfg)
  val meta = new RayMeta(addrWidth)
}
