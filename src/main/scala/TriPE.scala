package sdf_rt
import raytrace_utils._
import raytrace_utils.fudian._
import chisel3._
import chisel3.util._


class TriPE(val c: TriPeConfig) extends Module {
  val io = IO(new Bundle {
    val ray_in       = Input(new Ray(c.cfg))
    val ray_valid    = Input(Bool())

    val tri_batch_in = Input(new TriBatch(c.addrWidth))
    val tri_batch_valid = Input(Bool())
    val end_exec     = Input(Bool())

    // 存储器接口：请求一个 Block 的起始地址
    val mem_req      = Decoupled(UInt(c.addrWidth.W))
    val mem_resp     = Flipped(Decoupled(new TriangleBlock(c)))

    val output_ready = Output(Bool())
    val out_best_hit = Output(Bool())
    val out_t        = Output(UInt(c.cfg.totalWidth.W))
    val out_done     = Output(Valid(Bool()))
  })

  // --- 1. 任务调度逻辑 ---
  val batch_queue = Module(new Queue(new TriBatch(c.addrWidth), 8))
  batch_queue.io.enq.bits := io.tri_batch_in
  batch_queue.io.enq.valid := io.tri_batch_valid

  val current_batch = Reg(new TriBatch(c.addrWidth))
  val block_offset  = RegInit(0.U(16.W)) // 注意这里以 Block 为单位增加
  val batch_active  = RegInit(false.B)
  val no_more_batches = RegInit(false.B)

  // 状态机
  val s_IDLE :: s_BUSY :: s_FINISHING :: Nil = Enum(3)
  val state = RegInit(s_IDLE)

  when(state === s_IDLE && io.ray_valid) { state := s_BUSY; no_more_batches := false.B }
  when(io.end_exec) { no_more_batches := true.B }

  // 内存请求：请求大小为 numPEs * sizeof(Triangle)
  batch_queue.io.deq.ready := !batch_active && (state === s_BUSY)
  when(batch_queue.io.deq.fire) {
    current_batch := batch_queue.io.deq.bits
    block_offset  := 0.U
    batch_active  := true.B
  }

  io.mem_req.valid := batch_active
  // 假设地址按字节计算：1个三角形约108字节
  val shiftAmt = log2Up(c.numPEs)
  io.mem_req.bits  := current_batch.base_addr + block_offset<<shiftAmt

  when(io.mem_req.fire) {
    block_offset := block_offset + 1.U
    // 注意：这里的 count 如果不是 block 的整数倍，由最后的 Mask 处理
    when(block_offset === (current_batch.count + (c.numPEs - 1).U) / c.numPEs.U - 1.U) {
      batch_active := false.B
    }
  }

  val pes = Seq.fill(c.numPEs)(Module(new RayTriangleIntersection(c.cfg)))
  val pe_best_t = RegInit(VecInit(Seq.fill(c.numPEs)(0x7F7FFFFF.U(c.cfg.totalWidth.W))))
  val pe_has_hit = RegInit(VecInit(Seq.fill(c.numPEs)(false.B)))

  // 锁定光线参数给所有 PE
  val ray_reg = RegEnable(io.ray_in, io.ray_valid && state === s_IDLE)

  io.mem_resp.ready := true.B // 现在的吞吐量 PE 总是能接受

  for (i <- 0 until c.numPEs) {
    pes(i).io.ray := ray_reg
    pes(i).io.tri   := io.mem_resp.bits.tris(i)

    // 关键：结合 mem_resp 的 valid 和 mask 决定是否喂入 PE
    pes(i).io.in_valid := io.mem_resp.fire && io.mem_resp.bits.mask(i)

    // 每个 PE 内部维护自己的 best_t
    val fcomp = Module(new FCMP(c.cfg))
    fcomp.io.a := pes(i).io.t
    fcomp.io.b := pe_best_t(i)
    fcomp.io.signaling := false.B
    when(pes(i).io.out_valid && pes(i).io.hit) {
      when(fcomp.io.lt || !pe_has_hit(i)) {
        pe_best_t(i)  := pes(i).io.t
        pe_has_hit(i) := true.B
      }
    }
  }

  // --- 3. 全局在途计数与状态转换 ---
  val inflight_cnt = RegInit(0.U(10.W))
  // 每次 mem_resp.fire，计入 mask 中 valid 的数量
  val incoming_count = PopCount(io.mem_resp.bits.mask.asUInt)
  val outgoing_count = PopCount(pes.map(_.io.out_valid))

  inflight_cnt := inflight_cnt + Mux(io.mem_resp.fire, incoming_count, 0.U) - outgoing_count

  when(state === s_BUSY && no_more_batches && !batch_queue.io.deq.valid && !batch_active) {
    state := s_FINISHING
  }
  when(state === s_FINISHING && inflight_cnt === 0.U) {
    state := s_IDLE
  }

  // --- 4. 最终结果归约 (Reduction Tree) ---
  // 这里做一个简单的组合逻辑比较（或者在 DONE 状态下分步比较）
  val final_hit = pe_has_hit.asUInt.orR
  // 选出 pe_best_t 中的最小值
  val global_best_t = pe_best_t.reduce((a, b) => {
    val cmp = Module(new FCMP(c.cfg))
    cmp.io.a := a
    cmp.io.b := b
    cmp.io.signaling := false.B
    Mux(cmp.io.lt, a, b)
  })

  io.output_ready := (state === s_IDLE)
  io.out_best_hit := final_hit
  io.out_t        := global_best_t
  io.out_done.valid := (RegNext(state) === s_FINISHING && state === s_IDLE)
  io.out_done.bits  := final_hit
}