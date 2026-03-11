package sdf_rt

import chisel3._
import chisel3.util._
import raytrace_utils._

class CommitQueue(cfg: FloatConfig, depth: Int = 8) extends Module {
  require(depth > 1, "CommitQueue depth must be greater than 1")

  private val slotBits = log2Ceil(depth)
  private val countBits = log2Ceil(depth + 1)

  val io = IO(new Bundle {
    val allocValid = Input(Bool())
    val allocReady = Output(Bool())
    val allocSlot = Output(UInt(cfg.addrWidth.W))

    val writebackValid = Input(Bool())
    val writeback = Input(new RenderResult(cfg, cfg.addrWidth))

    val outValid = Output(Bool())
    val outResult = Output(new RenderResult(cfg, cfg.addrWidth))
  })

  val entries = Reg(Vec(depth, new RenderResult(cfg, cfg.addrWidth)))
  val reserved = RegInit(VecInit(Seq.fill(depth)(false.B)))
  val done = RegInit(VecInit(Seq.fill(depth)(false.B)))
  val allocPtr = RegInit(0.U(slotBits.W))
  val commitPtr = RegInit(0.U(slotBits.W))
  val count = RegInit(0.U(countBits.W))

  val commitIdx = commitPtr
  val commitValid = reserved(commitIdx) && done(commitIdx)
  val doAlloc = io.allocValid && io.allocReady
  val doCommit = commitValid

  io.allocReady := count =/= depth.U
  io.allocSlot := allocPtr
  io.outValid := commitValid
  io.outResult := entries(commitIdx)

  when(io.writebackValid) {
    val wbIdx = io.writeback.meta.slotId(slotBits - 1, 0)
    entries(wbIdx) := io.writeback
    done(wbIdx) := true.B
  }

  when(doAlloc) {
    reserved(allocPtr) := true.B
    done(allocPtr) := false.B
    allocPtr := allocPtr + 1.U
  }

  when(doCommit) {
    reserved(commitPtr) := false.B
    done(commitPtr) := false.B
    commitPtr := commitPtr + 1.U
  }

  switch(Cat(doAlloc, doCommit)) {
    is("b10".U) { count := count + 1.U }
    is("b01".U) { count := count - 1.U }
  }
}