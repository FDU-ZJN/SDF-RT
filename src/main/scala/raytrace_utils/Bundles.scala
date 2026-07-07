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

class TraceResultWithSlot(cfg: FloatConfig = FloatConfig.FP32, addrWidth: Int = 32) extends Bundle {
  val slotIdx = UInt(GlobalConfig.ddaTraceSlotBits.W)
  val result = new TraceResult(cfg, addrWidth)
}

class RenderResult(cfg: FloatConfig = FloatConfig.FP32, addrWidth: Int = 32) extends Bundle {
  val meta = new RayMeta(addrWidth)
  val hit = Bool()
  val hitId = UInt(addrWidth.W)
  val rgb8 = UInt(24.W)
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

class NormalMemReq(addrWidth: Int = 16, tagWidth: Int = 2) extends Bundle {
  val addr = UInt(addrWidth.W)
  val tag = UInt(tagWidth.W)
}

class NormalMemResp(tagWidth: Int = 2) extends Bundle {
  val data = UInt(GlobalConfig.normalMemDataWidth.W)
  val tag = UInt(tagWidth.W)
}

class SdfInitReq(cfg: FloatConfig = FloatConfig.FP32, addrWidth: Int = 32) extends Bundle {
  val rd = new Vec3(cfg)
  val meta = new RayMeta(addrWidth)
}

class SdfBypassResp(addrWidth: Int = 32) extends Bundle {
  val meta = new RayMeta(addrWidth)
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
  val subX = SInt((addrWidth + 1).W)
  val subY = SInt((addrWidth + 1).W)
  val subZ = SInt((addrWidth + 1).W)
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

class DdaTraceJob(cfg: FloatConfig = FloatConfig.FP32, addrWidth: Int = 32, maxCmds: Int = 1024) extends Bundle {
  val ray = new Ray(cfg)
  val meta = new RayMeta(addrWidth)
  val cmdCount = UInt(log2Ceil(maxCmds + 1).W)
  val cmds = Vec(maxCmds, new TriBatch(addrWidth))
}

class DdaTraceJobDesc(cfg: FloatConfig = FloatConfig.FP32, addrWidth: Int = 32, maxCmds: Int = 1024) extends Bundle {
  val ray = new Ray(cfg)
  val meta = new RayMeta(addrWidth)
  val cmdCount = UInt(log2Ceil(maxCmds + 1).W)
  val traceSlot = UInt(GlobalConfig.ddaTraceSlotBits.W)
}

class DdaStepResult(cfg: FloatConfig = FloatConfig.FP32, addrWidth: Int = 32) extends Bundle {
  val ctx = new DdaContext(cfg, addrWidth)
  val done = Bool()
  val emitCmd = Bool()
  val tri = new TriBatch(addrWidth)
}

class DdaTraceCmdWrite(addrWidth: Int = 32, maxCmds: Int = 1024) extends Bundle {
  val slotIdx = UInt(GlobalConfig.ddaTraceSlotBits.W)
  val cmdIdx = UInt(math.max(1, log2Ceil(maxCmds)).W)
  val tri = new TriBatch(addrWidth)
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
