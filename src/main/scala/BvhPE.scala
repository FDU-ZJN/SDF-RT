package sdf_rt

import chisel3._
import chisel3.util._
import raytrace_utils._
import raytrace_utils.fudian._

class BvhPE(val c: BvhPeConfig) extends Module {
  private val spWidth = log2Ceil(c.stackDepth + 1)
  private val countWidth = 20
  private val aabbLatency = 4 + c.cfg.faddLatency + c.cfg.fdivLatency + c.cfg.fmulLatency
  private val fpInf = "h7f7fffff".U(c.cfg.totalWidth.W)

  val io = IO(new Bundle {
    val start = Input(Bool())
    val rootNode = Input(UInt(c.addrWidth.W))
    val ray_in = Input(new Ray(c.cfg))

    val hit_update_valid = Input(Bool())
    val hit_update_t = Input(UInt(c.cfg.totalWidth.W))

    val node_req = Decoupled(UInt(c.addrWidth.W))
    val node_resp = Flipped(Decoupled(new BvhNode(c.cfg, c.addrWidth)))

    val leaf_out = Decoupled(new TriBatch(c.addrWidth))

    val busy = Output(Bool())
    val done = Output(Bool())
    val stack_level = Output(UInt(spWidth.W))
  })

  val rayReg = Reg(new Ray(c.cfg))
  val active = RegInit(false.B)
  val bestT = RegInit(fpInf)
  val hasBest = RegInit(false.B)

  val stack = Reg(Vec(c.stackDepth, UInt(c.addrWidth.W)))
  val sp = RegInit(0.U(spWidth.W))

  // Tracks unresolved traversal work units (nodes) through stack, request, and AABB pipeline.
  val outstandingNodes = RegInit(0.S(countWidth.W))

  val reqQ = Module(new Queue(UInt(c.addrWidth.W), c.reqQueueDepth))
  val leafQ = Module(new Queue(new TriBatch(c.addrWidth), c.leafQueueDepth))

  io.node_req <> reqQ.io.deq
  io.leaf_out <> leafQ.io.deq

  reqQ.io.enq.valid := false.B
  reqQ.io.enq.bits := 0.U
  leafQ.io.enq.valid := false.B
  leafQ.io.enq.bits := 0.U.asTypeOf(new TriBatch(c.addrWidth))

  val aabb = Module(new RayAABBIntersection(c.cfg))
  aabb.io.ray := rayReg
  aabb.io.aabb := io.node_resp.bits.bounds
  aabb.io.in_valid := io.node_resp.fire

  io.node_resp.ready := true.B

  val nodeCtxPipe = ShiftRegister(io.node_resp.bits, aabbLatency)
  val nodeCtxValid = ShiftRegister(io.node_resp.fire, aabbLatency)

  val cmpHitBest = Module(new FCMP(c.cfg))
  cmpHitBest.io.a := io.hit_update_t
  cmpHitBest.io.b := bestT
  cmpHitBest.io.signaling := false.B

  when(io.start && !active) {
    active := true.B
    rayReg := io.ray_in
    sp := 1.U
    stack(0) := io.rootNode
    bestT := fpInf
    hasBest := false.B
    outstandingNodes := 1.S
  }

  when(active && io.hit_update_valid) {
    when(!hasBest || cmpHitBest.io.lt) {
      bestT := io.hit_update_t
      hasBest := true.B
    }
  }

  val canPop = active && (sp =/= 0.U) && reqQ.io.enq.ready
  val poppedNode = Wire(UInt(c.addrWidth.W))
  val stackIdx = (sp - 1.U)(log2Ceil(c.stackDepth) - 1, 0)
  poppedNode := Mux(sp === 0.U, 0.U, stack(stackIdx))

  reqQ.io.enq.valid := canPop
  reqQ.io.enq.bits := poppedNode

  val cmpPrune = Module(new FCMP(c.cfg))
  cmpPrune.io.a := aabb.io.tNear
  cmpPrune.io.b := bestT
  cmpPrune.io.signaling := false.B

  val nodeAccepted = nodeCtxValid && aabb.io.hit && (!hasBest || cmpPrune.io.lt)
  val isLeaf = nodeCtxPipe.isLeaf

  val pushLeft = nodeAccepted && !isLeaf && nodeCtxPipe.leftValid
  val pushRight = nodeAccepted && !isLeaf && nodeCtxPipe.rightValid
  val pushCount = pushLeft.asUInt +& pushRight.asUInt

  val popCount = canPop.asUInt
  val baseSp = sp - popCount
  val baseIdx = baseSp(log2Ceil(c.stackDepth) - 1, 0)
  val rightIdx = (baseSp + pushLeft.asUInt)(log2Ceil(c.stackDepth) - 1, 0)
  when(pushLeft) {
    stack(baseIdx) := nodeCtxPipe.left
  }
  when(pushRight) {
    stack(rightIdx) := nodeCtxPipe.right
  }

  when(canPop || nodeCtxValid) {
    sp := baseSp + pushCount
  }

  val leafAccepted = nodeAccepted && isLeaf
  leafQ.io.enq.valid := leafAccepted
  leafQ.io.enq.bits.base_addr := nodeCtxPipe.triStart
  leafQ.io.enq.bits.count := nodeCtxPipe.triCount

  assert(!(leafAccepted && !leafQ.io.enq.ready), "BvhPE leaf queue overflow")
  assert((baseSp + pushCount) <= c.stackDepth.U, "BvhPE stack overflow")

  val nodeDelta = Wire(SInt(countWidth.W))
  nodeDelta := 0.S
  when(nodeCtxValid) {
    when(nodeAccepted && !isLeaf) {
      nodeDelta := pushCount.zext - 1.S
    }.otherwise {
      nodeDelta := -1.S
    }
  }

  val nextOutstanding = outstandingNodes + nodeDelta
  when(nodeCtxValid) {
    outstandingNodes := nextOutstanding
  }

  val donePulse = active && nodeCtxValid && (nextOutstanding === 0.S)
  when(donePulse) {
    active := false.B
    sp := 0.U
  }

  io.busy := active
  io.done := donePulse
  io.stack_level := sp
}
