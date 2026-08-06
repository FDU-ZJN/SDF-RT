package Trace

import chisel3._
import chisel3.util._
import raytrace_utils._

class TriMemReq(val c: TriPeConfig, val tagWidth: Int) extends Bundle {
  val addr = UInt(GlobalConfig.triMemAddrWidth.W)
  val mask = UInt(c.cacheLineTriangles.W)
  val tag = UInt(tagWidth.W)
}

class TriMemResp(val c: TriPeConfig, val tagWidth: Int) extends Bundle {
  val block = new TriangleBlock(c)
  val tag = UInt(tagWidth.W)
}

/** Per-line metadata stored alongside the tag array. */
class TriCacheLineMeta extends Bundle {
  val valid = Bool()
  val generation = Bool()
}


class TriMemRefillReq(numBanks: Int) extends Bundle {
  val bank = UInt(math.max(1, log2Ceil(numBanks)).W)
  val addr = UInt(GlobalConfig.triMemAddrWidth.W)
}

class TriMemBackendReq(val c: TriPeConfig, val numBanks: Int, val idWidth: Int) extends Bundle {
  val bank = UInt(math.max(1, log2Ceil(numBanks)).W)
  val addr = UInt(GlobalConfig.triMemAddrWidth.W)
  val id = UInt(idWidth.W)
}

class TriMemBackendResp(val c: TriPeConfig, val idWidth: Int) extends Bundle {
  val id = UInt(idWidth.W)
  val error = Bool()
}

class TriMemRefillWrite(val c: TriPeConfig, val idWidth: Int) extends Bundle {
  val id = UInt(idWidth.W)
  val data = UInt((c.cacheLineTriangles * 9 * c.cfg.totalWidth).W)
}

class TriMemDmaReadData(val dataWidth: Int) extends Bundle {
  val data = UInt(dataWidth.W)
  val last = Bool()
}

class TriMemAllocReq(
  val c: TriPeConfig,
  val numBanks: Int,
  val numSets: Int,
  val idWidth: Int
) extends Bundle {
  val bank = UInt(math.max(1, log2Ceil(numBanks)).W)
  val addr = UInt(GlobalConfig.triMemAddrWidth.W)
  val set = UInt(math.max(1, log2Ceil(numSets)).W)
  val tag = UInt((GlobalConfig.triMemAddrWidth - math.max(1, log2Ceil(numSets))).W)
  val victimWay = UInt(1.W)
}

class TriMemAllocResp(val idWidth: Int) extends Bundle {
  val id = UInt(idWidth.W)
  val merged = Bool()
}

class TriMemRefillDone(
  val c: TriPeConfig,
  val idWidth: Int,
  val numSets: Int = GlobalConfig.triMemCacheSets
) extends Bundle {
  private val setWidth = math.max(1, log2Ceil(numSets))
  private val tagWidth = GlobalConfig.triMemAddrWidth - setWidth
  val id = UInt(idWidth.W)
  val addr = UInt(GlobalConfig.triMemAddrWidth.W)
  val set = UInt(setWidth.W)
  val tag = UInt(tagWidth.W)
  val data = UInt((c.cacheLineTriangles * 9 * c.cfg.totalWidth).W)
}


class TriMemRelease(val idWidth: Int) extends Bundle {
  val id = UInt(idWidth.W)
}
