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
    val writeback5 = Flipped(Decoupled(new RenderResult(cfg, cfg.addrWidth)))
    val out        = Output(Valid(Vec(2, new RenderResult(cfg, cfg.addrWidth))))
  })

  val entries  = Reg(Vec(depth, new RenderResult(cfg, cfg.addrWidth)))
  val reserved = RegInit(VecInit(Seq.fill(depth)(false.B)))
  val done     = RegInit(VecInit(Seq.fill(depth)(false.B)))

  val allocPtr  = RegInit(0.U(slotBits.W))
  val commitPtr = RegInit(0.U(slotBits.W))
  val count     = RegInit(0.U(countBits.W))

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
  val sdfWbQ0 = Module(new Queue(new RenderResult(cfg, cfg.addrWidth), 2))
  val init1WbQ = Module(new Queue(new RenderResult(cfg, cfg.addrWidth), 2))
  val sdfWbQ1 = Module(new Queue(new RenderResult(cfg, cfg.addrWidth), 2))

  renderWbQ.io.enq <> io.writeback
  init0WbQ.io.enq <> io.writeback2
  sdfWbQ0.io.enq <> io.writeback3
  init1WbQ.io.enq <> io.writeback4
  sdfWbQ1.io.enq <> io.writeback5

  val physWbValid = Wire(Vec(2, Bool()))
  val physWbBits = Wire(Vec(2, new RenderResult(cfg, cfg.addrWidth)))
  val physWbIdx = Wire(Vec(2, UInt(slotBits.W)))

  private val wbSourceCount = 5

  val wbValid = Wire(Vec(wbSourceCount, Bool()))
  val wbBits = Wire(Vec(wbSourceCount, new RenderResult(cfg, cfg.addrWidth)))
  val wbReady = WireInit(VecInit(Seq.fill(wbSourceCount)(false.B)))

  wbValid(0) := renderWbQ.io.deq.valid
  wbBits(0) := renderWbQ.io.deq.bits
  wbValid(1) := init0WbQ.io.deq.valid
  wbBits(1) := init0WbQ.io.deq.bits
  wbValid(2) := sdfWbQ0.io.deq.valid
  wbBits(2) := sdfWbQ0.io.deq.bits
  wbValid(3) := init1WbQ.io.deq.valid
  wbBits(3) := init1WbQ.io.deq.bits
  wbValid(4) := sdfWbQ1.io.deq.valid
  wbBits(4) := sdfWbQ1.io.deq.bits

  val init0Valid = wbValid(1)
  val init1Valid = wbValid(3)
  val sdf0Valid = wbValid(2)
  val sdf1Valid = wbValid(4)
  val initAny = init0Valid || init1Valid
  val sdfAny = sdf0Valid || sdf1Valid
  val zeroResult = 0.U.asTypeOf(new RenderResult(cfg, cfg.addrWidth))

  physWbValid(0) := false.B
  physWbBits(0) := zeroResult
  physWbIdx(0) := 0.U
  physWbValid(1) := false.B
  physWbBits(1) := zeroResult
  physWbIdx(1) := 0.U

  when(initAny) {
    physWbValid(0) := true.B
    physWbBits(0) := Mux(init0Valid, wbBits(1), wbBits(3))
    wbReady(1) := init0Valid
    wbReady(3) := !init0Valid && init1Valid

    when(init0Valid && init1Valid) {
      physWbValid(1) := true.B
      physWbBits(1) := wbBits(3)
      wbReady(3) := true.B
    }
  }.elsewhen(sdfAny) {
    physWbValid(0) := true.B
    physWbBits(0) := Mux(sdf0Valid, wbBits(2), wbBits(4))
    wbReady(2) := sdf0Valid
    wbReady(4) := !sdf0Valid && sdf1Valid

    when(sdf0Valid && sdf1Valid) {
      physWbValid(1) := true.B
      physWbBits(1) := wbBits(4)
      wbReady(4) := true.B
    }
  }.elsewhen(wbValid(0)) {
    physWbValid(0) := true.B
    physWbBits(0) := wbBits(0)
    wbReady(0) := true.B
  }

  physWbIdx(0) := physWbBits(0).meta.slotId(slotBits - 1, 0)
  physWbIdx(1) := physWbBits(1).meta.slotId(slotBits - 1, 0)

  val doPhysWb0 = physWbValid(0)
  val doPhysWb1 = physWbValid(1)

  renderWbQ.io.deq.ready := wbReady(0)
  init0WbQ.io.deq.ready := wbReady(1)
  sdfWbQ0.io.deq.ready := wbReady(2)
  init1WbQ.io.deq.ready := wbReady(3)
  sdfWbQ1.io.deq.ready := wbReady(4)

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
  val commitReady = Wire(Vec(2, Bool()))
  for (i <- 0 until 2) {
    commitIdxs(i) := commitPtr + i.U
    commitReady(i) := reserved(commitIdxs(i)) && done(commitIdxs(i))
    io.out.bits(i) := entries(commitIdxs(i))
  }

  val commitPairValid = commitReady(0) && commitReady(1)
  val commitCount = Mux(commitPairValid, 2.U, 0.U)
  io.out.valid := commitPairValid

  when(commitPairValid) {
    for (i <- 0 until 2) {
      reserved(commitIdxs(i)) := false.B
      done(commitIdxs(i)) := false.B
    }
    commitPtr := commitPtr + commitCount
  }

  val countNext = count + allocCount - commitCount
  count := countNext
  val allocSpaceNext = depth.U - countNext
  allocReady0Reg := allocSpaceNext =/= 0.U
  allocReady1Reg := allocSpaceNext > 1.U
}
