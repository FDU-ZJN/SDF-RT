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
  val allocReady0Reg = RegInit(true.B)
  val allocReady1Reg = RegInit(true.B)
  io.alloc(0).ready := allocReady0Reg
  io.alloc(1).ready := allocReady1Reg
  io.allocSlot(0) := allocPtr
  io.allocSlot(1) := allocPtr + 1.U

  val doAlloc0 = io.alloc(0).fire
  val doAlloc1 = io.alloc(1).fire
  val allocCount = PopCount(Seq(doAlloc0, doAlloc1))

  val renderWbQ = Module(new Queue(new RenderResult(cfg, cfg.addrWidth), 2))
  val init0WbQ = Module(new Queue(new RenderResult(cfg, cfg.addrWidth), 2))
  val sdfWbQ = Module(new Queue(new RenderResult(cfg, cfg.addrWidth), 2))
  val init1WbQ = Module(new Queue(new RenderResult(cfg, cfg.addrWidth), 2))

  renderWbQ.io.enq <> io.writeback
  init0WbQ.io.enq <> io.writeback2
  sdfWbQ.io.enq <> io.writeback3
  init1WbQ.io.enq <> io.writeback4

  val nonInitGroupValid = sdfWbQ.io.deq.valid || renderWbQ.io.deq.valid

  val physWbValid = Wire(Vec(2, Bool()))
  val physWbBits = Wire(Vec(2, new RenderResult(cfg, cfg.addrWidth)))
  val physWbIdx = Wire(Vec(2, UInt(slotBits.W)))

  physWbValid(0) := Mux(nonInitGroupValid, sdfWbQ.io.deq.valid, init0WbQ.io.deq.valid)
  physWbBits(0) := Mux(nonInitGroupValid, sdfWbQ.io.deq.bits, init0WbQ.io.deq.bits)
  physWbIdx(0) := physWbBits(0).meta.slotId(slotBits - 1, 0)

  physWbValid(1) := Mux(nonInitGroupValid, renderWbQ.io.deq.valid, init1WbQ.io.deq.valid)
  physWbBits(1) := Mux(nonInitGroupValid, renderWbQ.io.deq.bits, init1WbQ.io.deq.bits)
  physWbIdx(1) := physWbBits(1).meta.slotId(slotBits - 1, 0)

  val doPhysWb0 = physWbValid(0)
  val doPhysWb1 = physWbValid(1)

  init0WbQ.io.deq.ready := !nonInitGroupValid && doPhysWb0
  init1WbQ.io.deq.ready := !nonInitGroupValid && doPhysWb1
  sdfWbQ.io.deq.ready := nonInitGroupValid && doPhysWb0
  renderWbQ.io.deq.ready := nonInitGroupValid && doPhysWb1

  val physWbValidReg = RegInit(VecInit(Seq.fill(2)(false.B)))
  val physWbBitsReg = Reg(Vec(2, new RenderResult(cfg, cfg.addrWidth)))
  val physWbIdxReg = Reg(Vec(2, UInt(slotBits.W)))

  for (i <- 0 until 2) {
    when(physWbValidReg(i)) {
      assert(reserved(physWbIdxReg(i)), "CommitQueue writeback to unreserved slot")
      assert(!done(physWbIdxReg(i)), "CommitQueue duplicate writeback to completed slot")
      entries(physWbIdxReg(i)) := physWbBitsReg(i)
      done(physWbIdxReg(i))    := true.B
    }
  }

  when(physWbValidReg(0) && physWbValidReg(1)) {
    assert(physWbIdxReg(0) =/= physWbIdxReg(1), "CommitQueue two physical writebacks target the same slot")
  }

  physWbValidReg(0) := doPhysWb0
  physWbBitsReg(0) := physWbBits(0)
  physWbIdxReg(0) := physWbIdx(0)
  physWbValidReg(1) := doPhysWb1
  physWbBitsReg(1) := physWbBits(1)
  physWbIdxReg(1) := physWbIdx(1)

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

  val countNext = count + allocCount - commitCount
  count := countNext
  val allocSpaceNext = depth.U - countNext
  allocReady0Reg := allocSpaceNext =/= 0.U
  allocReady1Reg := allocSpaceNext > 1.U
}
