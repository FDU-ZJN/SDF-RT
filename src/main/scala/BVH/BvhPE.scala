package BVH

import chisel3._
import chisel3.util._
import raytrace_utils._
import raytrace_utils.fudian._

class BvhPE(val c: BvhPeConfig) extends Module {
  private val spWidth    = log2Ceil(c.stackDepth + 1)
  private val countWidth = 20
  private val aabbLatency =
    4 + c.cfg.faddLatency + c.cfg.fdivLatency + c.cfg.fmulLatency + (4 * c.cfg.fcmpLatency)
  private val fpInf = "h7f7fffff".U(c.cfg.totalWidth.W)

  val io = IO(new Bundle {
    val start    = Input(Bool())
    val rootNode = Input(UInt(c.addrWidth.W))
    val ray_in   = Input(new Ray(c.cfg))
    val start_meta = Input(new RayMeta(c.addrWidth))

    val hit_update_valid = Input(Bool())
    val hit_update_t     = Input(UInt(c.cfg.totalWidth.W))

    val node_req  = Decoupled(UInt(c.addrWidth.W))
    val node_resp = Flipped(Decoupled(new BvhNode(c.cfg, c.addrWidth)))

    val leaf_out = Decoupled(new TriBatch(c.addrWidth))

    val no_leaf_done = Output(Bool())
    val done_meta = Output(new RayMeta(c.addrWidth))
    val start_ready = Output(Bool())
    val busy        = Output(Bool())
    val done        = Output(Bool())
    val stack_level = Output(UInt(spWidth.W))
    val ray_passthrough = Output(new Ray(c.cfg))
    val first_leaf_pulse = Output(Bool())
  })

  // ── 全局状态寄存器 ──────────────────────────────────────────────────────────
  val rayReg  = Reg(new Ray(c.cfg))
  val active  = RegInit(false.B)
  val bestT   = RegInit(fpInf)
  val hasBest = RegInit(false.B)
  val startMetaReg = Reg(new RayMeta(c.addrWidth))
  val sawValidLeaf = RegInit(false.B)

  // ── 栈模块实例化 ────────────────────────────────────────────────────────────
  val bvhStack = Module(new BvhStack(c.addrWidth, c.stackDepth))

  // ── 未完成节点计数（覆盖栈、请求队列、AABB 流水线中的节点）─────────────────
  val outstandingNodes = RegInit(0.S(countWidth.W))

  // ── 请求队列 ────────────────────────────────────────────────────────────────
  val reqQ = Module(new Queue(UInt(c.addrWidth.W), c.reqQueueDepth))
  io.node_req <> reqQ.io.deq

  // ── AABB 相交测试 ───────────────────────────────────────────────────────────
  val aabb = Module(new RayAABBIntersection(c.cfg))
  aabb.io.ray      := rayReg
  aabb.io.aabb     := io.node_resp.bits.bounds
  aabb.io.in_valid := io.node_resp.fire

  io.node_resp.ready := true.B

  // 将节点上下文和有效位与 AABB 结果对齐（同步延迟流水线）
  val nodeCtxPipe  = ShiftRegister(io.node_resp.bits, aabbLatency)
  val nodeCtxValid = ShiftRegister(io.node_resp.fire, aabbLatency)


  // ── 启动逻辑 ────────────────────────────────────────────────────────────────
  when(io.start && !active) {
    active  := true.B
    rayReg  := io.ray_in
    startMetaReg := io.start_meta
    bestT   := fpInf
    hasBest := false.B
    sawValidLeaf := false.B
    outstandingNodes := 1.S
  }

  // ── 外部命中更新 ────────────────────────────────────────────────────────────
  when(active && io.hit_update_valid) {
      bestT   := io.hit_update_t
      hasBest := true.B

  }
  val canPop = active && !bvhStack.io.empty && reqQ.io.enq.ready

  reqQ.io.enq.valid := canPop
  reqQ.io.enq.bits  := bvhStack.io.topData
  when(reqQ.io.enq.valid) {
    assert(reqQ.io.enq.ready, "BvhPE reqQ overflow")
  }
  val cmpPrune = Module(new FCMP(c.cfg))
  cmpPrune.io.a        := aabb.io.tNear
  cmpPrune.io.b        := bestT
  cmpPrune.io.signaling := false.B

  // ── 节点接受与分类 ──────────────────────────────────────────────────────────
  val nodeAccepted = nodeCtxValid && aabb.io.hit
//  && (!hasBest || cmpPrune.io.lt)
  val isLeaf       = nodeCtxPipe.isLeaf

  val pushLeft  = nodeAccepted && !isLeaf && nodeCtxPipe.leftValid
  val pushRight = nodeAccepted && !isLeaf && nodeCtxPipe.rightValid
  val pushCount = pushLeft.asUInt +& pushRight.asUInt

  // ── 连接栈控制信号 ──────────────────────────────────────────────────────────
  bvhStack.io.pop       := canPop
  bvhStack.io.pushLeft  := pushLeft
  bvhStack.io.pushRight := pushRight
  bvhStack.io.leftData  := nodeCtxPipe.left
  bvhStack.io.rightData := nodeCtxPipe.right

  // 启动时通过 pushLeft 压入根节点（仅在第一拍有效）
  when(io.start && !active) {
    bvhStack.io.pushLeft  := true.B
    bvhStack.io.leftData  := io.rootNode
    bvhStack.io.pushRight := false.B
    bvhStack.io.pop       := false.B
  }
  // ── 叶子节点输出 ────────────────────────────────────────────────────────────
  val leafAccepted      = nodeAccepted && isLeaf&&nodeCtxPipe.triCount>0.U
  val firstLeafPulse = leafAccepted && !sawValidLeaf
  when(leafAccepted) {
    sawValidLeaf := true.B
  }
  io.first_leaf_pulse := firstLeafPulse
  io.leaf_out.valid     := leafAccepted
  io.leaf_out.bits.base_addr := nodeCtxPipe.triStart
  io.leaf_out.bits.count     := nodeCtxPipe.triCount

  // ── 未完成节点计数更新 ──────────────────────────────────────────────────────
  val popDelta = Wire(SInt(countWidth.W))
  popDelta := Mux(canPop, 1.S, 0.S)

  // 2. AABB 完成阶段：根据实际情况修正计数
  val aabbDelta = Wire(SInt(countWidth.W))
  aabbDelta := 0.S
  when(nodeCtxValid) {
    when(nodeAccepted && !isLeaf) {
      aabbDelta := pushCount.zext - 2.S
    }.otherwise {
      aabbDelta := -2.S
    }
  }

  // 3. 计算下一个周期的计数值
  val nextOutstanding = outstandingNodes + popDelta + aabbDelta

  // 更新寄存器（注意：现在需要在 pop 或 aabb 结果有效时都进行更新）
  when(canPop || nodeCtxValid) {
    outstandingNodes := nextOutstanding
  }

  // ── 完成脉冲 ────────────────────────────────────────────────────────────────
  val donePulse = active && nodeCtxValid && (nextOutstanding === 0.S)
  io.no_leaf_done := donePulse && !sawValidLeaf && !leafAccepted
  io.done_meta := startMetaReg
  when(donePulse) {
    active := false.B
  }
  io.start_ready := !active
  io.busy        := active
  io.done        := donePulse
  io.stack_level := bvhStack.io.level

  // ── 透射光线寄存器 ───────────────────────────────────────────────────────────
  val rayPassthroughReg = Reg(new Ray(c.cfg))

  when(io.start && !active) {
    rayPassthroughReg := io.ray_in
  }

  io.ray_passthrough := rayPassthroughReg
}
