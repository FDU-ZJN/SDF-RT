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
    val alloc     = Flipped(Decoupled(UInt(cfg.addrWidth.W)))
    val allocSlot = Output(UInt(slotBits.W))
    val writeback  = Flipped(Decoupled(new RenderResult(cfg, cfg.addrWidth)))
    val writeback2 = Flipped(Decoupled(new RenderResult(cfg, cfg.addrWidth)))
    val writeback3 = Flipped(Decoupled(new RenderResult(cfg, cfg.addrWidth)))
    val out        = Decoupled(new RenderResult(cfg, cfg.addrWidth))
  })

  val entries  = RegInit(VecInit(Seq.fill(depth)(0.U.asTypeOf(new RenderResult(cfg, cfg.addrWidth)))))
  val reserved = RegInit(VecInit(Seq.fill(depth)(false.B)))
  val done     = RegInit(VecInit(Seq.fill(depth)(false.B)))

  val allocPtr  = RegInit(0.U(slotBits.W))
  val commitPtr = RegInit(0.U(slotBits.W))
  val count     = RegInit(0.U(countBits.W))

  val commitIdx   = commitPtr
  val commitValid = reserved(commitIdx) && done(commitIdx)
  val doCommit    = commitValid && io.out.ready

  val doAlloc = io.alloc.fire

  io.alloc.ready := count =/= depth.U
  io.allocSlot   := allocPtr

  io.out.valid := commitValid
  io.out.bits  := entries(commitIdx)

  val wbIdx     = io.writeback.bits.meta.slotId(slotBits - 1, 0)
  val wb2Idx    = io.writeback2.bits.meta.slotId(slotBits - 1, 0)
  val wb3Idx    = io.writeback3.bits.meta.slotId(slotBits - 1, 0)

  io.writeback.ready  := true.B
  io.writeback2.ready := true.B
  io.writeback3.ready := true.B

  val doWb2 = io.writeback2.fire
  val doWb3 = io.writeback3.fire && !(doWb2 && (wb2Idx === wb3Idx))
  val doWb1 = io.writeback.fire &&
    !(doWb2 && (wb2Idx === wbIdx)) &&
    !(doWb3 && (wb3Idx === wbIdx))

  when(doWb2) {
    entries(wb2Idx) := io.writeback2.bits
    done(wb2Idx)    := true.B
  }
  when(doWb3) {
    entries(wb3Idx) := io.writeback3.bits
    done(wb3Idx)    := true.B
  }
  when(doWb1) {
    entries(wbIdx) := io.writeback.bits
    done(wbIdx)    := true.B
  }

  when(doAlloc) {
    reserved(allocPtr) := true.B
    done(allocPtr)     := false.B
    allocPtr           := allocPtr + 1.U
  }
  when(doCommit) {
    reserved(commitPtr) := false.B
    done(commitPtr)     := false.B
    commitPtr           := commitPtr + 1.U
  }

  switch(Cat(doAlloc, doCommit)) {
    is("b10".U) { count := count + 1.U }
    is("b01".U) { count := count - 1.U }
    is("b11".U) { count := count }
  }
}
