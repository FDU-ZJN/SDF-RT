package sdf_rt

import raytrace_utils._
import raytrace_utils.fudian._
import chisel3._
import chisel3.util._

class TriPE(val c: TriPeConfig) extends Module {

  val io = IO(new Bundle {
    val ray_in       = Input(new Ray(c.cfg))
    val ray_valid    = Input(Bool())

    val tri_batch_in    = Input(new TriBatch(c.addrWidth))
    val tri_batch_valid = Input(Bool())
    val end_exec        = Input(Bool())

    val mem_req  = Decoupled(UInt(c.addrWidth.W))
    val mem_resp = Flipped(Decoupled(new TriangleBlock(c)))

    val output_ready = Output(Bool())
    val out_best_hit = Output(Bool())
    val hit_id       = Output(UInt(c.addrWidth.W))
    val out_done     = Output(Valid(Bool()))
  })

  // ============================================================
  // 1. 任务调度
  // ============================================================

  val batch_queue = Module(new Queue(new TriBatch(c.addrWidth), 8))
  batch_queue.io.enq.bits  := io.tri_batch_in
  batch_queue.io.enq.valid := io.tri_batch_valid

  val current_batch = Reg(new TriBatch(c.addrWidth))
  val block_offset  = RegInit(0.U(16.W))
  val batch_active  = RegInit(false.B)
  val no_more_batches = RegInit(false.B)

  val s_IDLE :: s_BUSY :: s_FINISHING :: Nil = Enum(3)
  val state = RegInit(s_IDLE)

  when(state === s_IDLE && io.ray_valid) {
    state := s_BUSY
    no_more_batches := false.B
  }

  when(io.end_exec) {
    no_more_batches := true.B
  }

  batch_queue.io.deq.ready := !batch_active && (state === s_BUSY)

  when(batch_queue.io.deq.fire) {
    current_batch := batch_queue.io.deq.bits
    block_offset  := 0.U
    batch_active  := true.B
  }

  val shiftAmt = log2Up(c.numPEs)

  io.mem_req.valid := batch_active
  io.mem_req.bits  := current_batch.base_addr + (block_offset << shiftAmt)

  when(io.mem_req.fire) {
    block_offset := block_offset + 1.U

    val totalBlocks =
      (current_batch.count + (c.numPEs - 1).U) / c.numPEs.U

    when(block_offset === totalBlocks - 1.U) {
      batch_active := false.B
    }
  }

  // ============================================================
  // 2. PE 阵列
  // ============================================================

  val pes = Seq.fill(c.numPEs)(Module(new RayTriangleIntersection(c.cfg)))

  val pe_best_t   = RegInit(VecInit(Seq.fill(c.numPEs)(
    0x7F7FFFFF.U(c.cfg.totalWidth.W)
  )))
  val pe_best_id  = Reg(Vec(c.numPEs, UInt(c.addrWidth.W)))
  val pe_has_hit  = RegInit(VecInit(Seq.fill(c.numPEs)(false.B)))

  // 新 ray 清空历史 best
  when(state === s_IDLE && io.ray_valid) {
    for(i <- 0 until c.numPEs) {
      pe_best_t(i)  := 0x7F7FFFFF.U
      pe_best_id(i) := 0.U
      pe_has_hit(i) := false.B
    }
  }

  val ray_reg = RegEnable(io.ray_in, io.ray_valid && state === s_IDLE)

  io.mem_resp.ready := true.B

  for (i <- 0 until c.numPEs) {

    pes(i).io.ray := ray_reg
    pes(i).io.tri := io.mem_resp.bits.tris(i)

    pes(i).io.in_valid :=
      io.mem_resp.fire && io.mem_resp.bits.mask(i)

    // 本地比较
    val fcmp = Module(new FCMP(c.cfg))
    fcmp.io.a := pes(i).io.t
    fcmp.io.b := pe_best_t(i)
    fcmp.io.signaling := false.B

    when(pes(i).io.out_valid && pes(i).io.hit) {
      when(fcmp.io.lt || !pe_has_hit(i)) {
        pe_best_t(i)  := pes(i).io.t
        pe_best_id(i) := pes(i).io.id
        pe_has_hit(i) := true.B
      }
    }
  }

  // ============================================================
  // 3. inflight 计数 + 状态转换
  // ============================================================

  val inflight_cnt = RegInit(0.U(10.W))

  val incoming_count =
    PopCount(io.mem_resp.bits.mask.asUInt)

  val outgoing_count =
    PopCount(pes.map(_.io.out_valid))

  inflight_cnt :=
    inflight_cnt +
      Mux(io.mem_resp.fire, incoming_count, 0.U) -
      outgoing_count

  when(state === s_BUSY &&
    no_more_batches &&
    !batch_queue.io.deq.valid &&
    !batch_active) {
    state := s_FINISHING
  }

  when(state === s_FINISHING && inflight_cnt === 0.U) {
    state := s_IDLE
  }

  // ============================================================
  // 4. 全局 argmin(t)
  // ============================================================

  val pairs =
    (0 until c.numPEs).map(i =>
      (pe_best_t(i), pe_best_id(i), pe_has_hit(i))
    )

  val (global_best_t, global_best_id, global_has_hit) =
    pairs.reduce { (a, b) =>

      val cmp = Module(new FCMP(c.cfg))
      cmp.io.a := a._1
      cmp.io.b := b._1
      cmp.io.signaling := false.B

      val a_better =
        a._3 && (!b._3 || cmp.io.lt)

      (
        Mux(a_better, a._1, b._1),
        Mux(a_better, a._2, b._2),
        a._3 || b._3
      )
    }

  // ============================================================
  // 5. 输出
  // ============================================================

  io.output_ready := (state === s_IDLE)

  io.out_best_hit := global_has_hit
  io.hit_id       := global_best_id

  val done_pulse =
    (RegNext(state) === s_FINISHING &&
      state === s_IDLE)

  io.out_done.valid := done_pulse
  io.out_done.bits  := global_has_hit
}