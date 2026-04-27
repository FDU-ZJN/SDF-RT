package DDA.Trace

import chisel3._
import chisel3.util._
import raytrace_utils._
import raytrace_utils.fudian._
import raytrace_utils.PipeUtils._

class TriPE(val c: TriPeConfig) extends Module {

  val io = IO(new Bundle {
    val ray_in: Ray = Input(new Ray(c.cfg))
    val ray_meta: RayMeta = Input(new RayMeta(c.addrWidth))
    val ray_valid: Bool = Input(Bool())

    val tri_batch_in: TriBatch = Input(new TriBatch(c.addrWidth))
    val tri_batch_valid: Bool = Input(Bool())
    val end_exec: Bool = Input(Bool())

    val mem_req: DecoupledIO[UInt] = Decoupled(UInt(c.addrWidth.W))
    val mem_req_mask: DecoupledIO[UInt] = Decoupled(UInt(c.numPEs.W))  // per-lane valid mask
    val mem_resp: DecoupledIO[TriangleBlock] = Flipped(Decoupled(new TriangleBlock(c)))

    val start_ready: Bool = Output(Bool())
    val output_ready: Bool = Output(Bool())
    val out_best_hit: Bool = Output(Bool())
    val hit_id: UInt = Output(UInt(c.addrWidth.W))
    val t_best: UInt = Output(UInt(c.cfg.totalWidth.W))
    val out_meta: RayMeta = Output(new RayMeta(c.addrWidth))
    val out_done: Bool = Output(Bool())
  })

  // ============================================================
  // 1. 任务调度
  // ============================================================

  private val batch_queue = Module(new Queue(new TriBatch(c.addrWidth), GlobalConfig.triBatchQueueDepth))
  batch_queue.io.enq.bits := io.tri_batch_in
  batch_queue.io.enq.valid := io.tri_batch_valid
  when(batch_queue.io.enq.valid) {
    assert(batch_queue.io.enq.ready, "TriPE batch_queue overflow")
  }

  private val current_batch: TriBatch = RegInit(0.U.asTypeOf(new TriBatch(c.addrWidth)))
  private val ray_meta_reg: RayMeta = RegInit(0.U.asTypeOf(new RayMeta(c.addrWidth)))
  private val block_offset: UInt = RegInit(0.U(16.W))
  private val batch_active: Bool = RegInit(false.B)
  private val batch_in_progress: Bool = RegInit(false.B)
  private val no_more_batches: Bool = RegInit(false.B)
  private val done_pulse_reg: Bool = RegInit(false.B)

  private val pe_best_t: Vec[UInt] = RegInit(VecInit(Seq.fill(c.numPEs)(
    0x7F7FFFFF.U(c.cfg.totalWidth.W)
  )))
  private val pe_best_id: Vec[UInt] = RegInit(VecInit(Seq.fill(c.numPEs)(0.U(c.addrWidth.W))))
  private val pe_has_hit: Vec[Bool] = RegInit(VecInit(Seq.fill(c.numPEs)(false.B)))

  private val s_IDLE :: s_BUSY :: Nil = Enum(2)
  private val state: UInt = RegInit(s_IDLE)

  when(state === s_IDLE && io.ray_valid) {
    state := s_BUSY
    ray_meta_reg := io.ray_meta
    no_more_batches := false.B
  }

  when(io.end_exec) {
    no_more_batches := true.B
  }

  batch_queue.io.deq.ready := !batch_in_progress && (state === s_BUSY)

  when(batch_queue.io.deq.fire) {
    current_batch := batch_queue.io.deq.bits
    block_offset := 0.U
    batch_active := true.B
    batch_in_progress := true.B
    for (i <- 0 until c.numPEs) {
      pe_best_t(i) := 0x7F7FFFFF.U
      pe_best_id(i) := 0.U
      pe_has_hit(i) := false.B
    }
  }

  private val shiftAmt: Int = log2Up(c.numPEs)

  // Align base_addr to block boundary (multiple of 4)
  private val alignedBase: UInt = (current_batch.base_addr >> shiftAmt).asUInt << shiftAmt.U

  // Calculate total blocks needed to cover [base_addr, base_addr + count)
  private val lastTri: UInt = current_batch.base_addr + current_batch.count - 1.U
  private val totalBlocks: UInt = ((lastTri >> shiftAmt).asUInt - (alignedBase >> shiftAmt).asUInt) + 1.U

  // Calculate per-lane mask for current aligned address
  // lane i corresponds to triangle: aligned_addr + i
  // mask[i] = 1 iff base_addr <= (aligned_addr + i) < base_addr + count
  private val alignedAddr: UInt = alignedBase + (block_offset << shiftAmt).asUInt
  private val reqMask: UInt = {
    val masks = (0 until c.numPEs).map { lane =>
      val triIdx = alignedAddr + lane.U
      (triIdx >= current_batch.base_addr) && (triIdx < current_batch.base_addr + current_batch.count)
    }
    Cat(masks.reverse)
  }

  io.mem_req.valid := batch_active
  io.mem_req.bits := alignedAddr
  io.mem_req_mask.valid := batch_active
  io.mem_req_mask.bits := reqMask

  when(io.mem_req.fire) {
    block_offset := block_offset + 1.U

    when(block_offset === totalBlocks - 1.U) {
      batch_active := false.B
    }
  }

  // ============================================================
  // 2. PE 阵列
  // ============================================================

  private val pes = Seq.fill(c.numPEs)(Module(new RayTriangleIntersection(c.cfg)))

  private val ray_reg: Ray = pipeUInt(io.ray_in.asUInt, 1, 0.U).asTypeOf(new Ray(c.cfg))

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
        pe_best_t(i) := pes(i).io.t
        pe_best_id(i) := pes(i).io.id
        pe_has_hit(i) := true.B
      }
    }
  }

  // ============================================================
  // 3. inflight 计数 + 状态转换
  // ============================================================

  private val inflight_cnt: UInt = RegInit(0.U(10.W))

  private val incoming_count: UInt = PopCount(io.mem_resp.bits.mask.asUInt)
  private val outgoing_count: UInt = PopCount(pes.map(_.io.out_valid))
  private val inflight_next =
    inflight_cnt +
      Mux(io.mem_resp.fire, incoming_count, 0.U) -
      outgoing_count
  private val batch_done_now =
    batch_in_progress &&
      !batch_active &&
      (inflight_next === 0.U)

  inflight_cnt := inflight_next
  done_pulse_reg := false.B

  when(batch_done_now) {
    batch_in_progress := false.B
    done_pulse_reg := true.B
  }

  when(state === s_BUSY &&
    no_more_batches &&
    !batch_queue.io.deq.valid &&
    !batch_in_progress) {
    state := s_IDLE
  }

  // ============================================================
  // 4. 全局 argmin(t)
  // ============================================================

  private val pairs = (0 until c.numPEs).map(i => (pe_best_t(i), pe_best_id(i), pe_has_hit(i)))
  private val (global_best_t, global_best_id, global_has_hit) = pairs.reduce { (a, b) =>
    val cmp = Module(new FCMP(c.cfg))
    cmp.io.a := a._1
    cmp.io.b := b._1
    cmp.io.signaling := false.B
    val a_better = a._3 && (!b._3 || cmp.io.lt)
    (
      Mux(a_better, a._1, b._1),
      Mux(a_better, a._2, b._2),
      a._3 || b._3
    )
  }

  // ============================================================
  // 5. 输出
  // ============================================================

  io.start_ready := state === s_IDLE
  io.output_ready := batch_queue.io.enq.ready

  io.out_best_hit := global_has_hit
  io.hit_id := global_best_id
  io.t_best := global_best_t
  io.out_meta := ray_meta_reg

  io.out_done := done_pulse_reg
}
