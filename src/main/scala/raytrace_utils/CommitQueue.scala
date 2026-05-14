package raytrace_utils

import chisel3._
import chisel3.util._
import raytrace_utils.GlobalConfig._
class CommitQueue(cfg: FloatConfig) extends Module {
  val depth = commitQueueDepth
  require(depth > 1, "CommitQueue depth must be greater than 1")
  require(isPow2(depth), "CommitQueue depth must be a power of 2")

  private val countBits = log2Ceil(depth + 1)

  val io = IO(new Bundle {
    val alloc     = Vec(2, Flipped(Decoupled(UInt(cfg.addrWidth.W))))
    val allocSlot = Output(Vec(2, UInt(slotBits.W)))
    val allocFree = Output(UInt(countBits.W))
    val writeback  = Flipped(Decoupled(new RenderResult(cfg, cfg.addrWidth)))
    val writeback2 = Flipped(Decoupled(new RenderResult(cfg, cfg.addrWidth)))
    val writeback3 = Flipped(Decoupled(new RenderResult(cfg, cfg.addrWidth)))
    val writeback4 = Flipped(Decoupled(new RenderResult(cfg, cfg.addrWidth)))
    val out        = Output(Vec(2, Valid(new RenderResult(cfg, cfg.addrWidth))))
  })

  val entries  = Reg(Vec(depth, new RenderResult(cfg, cfg.addrWidth)))
  val reserved = Reg(Vec(depth, Bool()))
  val done     = Reg(Vec(depth, Bool()))

  val allocPtr  = Reg(UInt(slotBits.W))
  val commitPtr = Reg(UInt(slotBits.W))
  val count     = Reg(UInt(countBits.W))

  val allocSpace = depth.U - count
  io.allocFree := allocSpace
  io.alloc(0).ready := allocSpace =/= 0.U
  io.alloc(1).ready := allocSpace > 1.U
  io.allocSlot(0) := allocPtr
  io.allocSlot(1) := allocPtr + 1.U

  val doAlloc0 = io.alloc(0).fire
  val doAlloc1 = io.alloc(1).fire
  val allocCount = PopCount(Seq(doAlloc0, doAlloc1))

  val wbIdx     = io.writeback.bits.meta.slotId(slotBits - 1, 0)
  val wb2Idx    = io.writeback2.bits.meta.slotId(slotBits - 1, 0)
  val wb3Idx    = io.writeback3.bits.meta.slotId(slotBits - 1, 0)
  val wb4Idx    = io.writeback4.bits.meta.slotId(slotBits - 1, 0)

  io.writeback.ready  := true.B
  io.writeback2.ready := true.B
  io.writeback3.ready := true.B
  io.writeback4.ready := true.B

  val doWb2 = io.writeback2.fire
  val doWb4 = io.writeback4.fire && !(doWb2 && (wb2Idx === wb4Idx))
  val doWb3 = io.writeback3.fire &&
    !(doWb2 && (wb2Idx === wb3Idx)) &&
    !(doWb4 && (wb4Idx === wb3Idx))
  val doWb1 = io.writeback.fire &&
    !(doWb2 && (wb2Idx === wbIdx)) &&
    !(doWb4 && (wb4Idx === wbIdx)) &&
    !(doWb3 && (wb3Idx === wbIdx))

  when(doWb2) {
    entries(wb2Idx) := io.writeback2.bits
    done(wb2Idx)    := true.B
  }
  when(doWb4) {
    entries(wb4Idx) := io.writeback4.bits
    done(wb4Idx)    := true.B
  }
  when(doWb3) {
    entries(wb3Idx) := io.writeback3.bits
    done(wb3Idx)    := true.B
  }
  when(doWb1) {
    entries(wbIdx) := io.writeback.bits
    done(wbIdx)    := true.B
  }

  when(doAlloc0) {
    reserved(allocPtr) := true.B
    done(allocPtr)     := false.B
  }
  when(doAlloc1) {
    reserved(allocPtr + doAlloc0.asUInt) := true.B
    done(allocPtr + doAlloc0.asUInt)     := false.B
  }
  when(allocCount =/= 0.U) {
    allocPtr := allocPtr + allocCount
  }

  val commitIdxs = Wire(Vec(2, UInt(slotBits.W)))
  val commitValids = Wire(Vec(2, Bool()))
  for (i <- 0 until 2) {
    commitIdxs(i) := commitPtr + i.U
    val laneReady = reserved(commitIdxs(i)) && done(commitIdxs(i))
    if (i == 0) {
      commitValids(i) := laneReady
    } else {
      commitValids(i) := commitValids(i - 1) && laneReady
    }
    io.out(i).valid := commitValids(i)
    io.out(i).bits := entries(commitIdxs(i))
  }

  val commitCount = PopCount(commitValids)

  when(commitCount =/= 0.U) {
    for (i <- 0 until 2) {
      when(commitValids(i)) {
        reserved(commitIdxs(i)) := false.B
        done(commitIdxs(i)) := false.B
      }
    }
    commitPtr := commitPtr + commitCount
  }

  count := count + allocCount - commitCount
}
