package Trace

import chisel3._
import chisel3.util._
import raytrace_utils._
import raytrace_utils.fudian._

class TriPE(val c: TriPeConfig) extends Module {
  require(c.numPEs == 1, s"TriPE two-level fetch currently requires numPEs=1, got ${c.numPEs}")

  private val triIdWidth = GlobalConfig.triRefIdWidth
  private val refPackFactor = GlobalConfig.triRefPackFactor
  private val refPackShift = log2Ceil(refPackFactor)
  private val refCountWidth = log2Ceil(refPackFactor + 1)
  private val refPrefetchLowWatermark = math.max(1, refPackFactor / 4)

  val io = IO(new Bundle {
    val ray_in: Ray = Input(new Ray(c.cfg))
    val ray_meta: RayMeta = Input(new RayMeta(c.addrWidth))
    val ray_valid: Bool = Input(Bool())

    val tri_batch_in: TriBatch = Input(new TriBatch(c.addrWidth))
    val tri_batch_valid: Bool = Input(Bool())
    val end_exec: Bool = Input(Bool())
    val flush: Bool = Input(Bool())

    val ref_mem_req = Decoupled(UInt(c.addrWidth.W))
    val ref_mem_resp = Flipped(Decoupled(UInt(GlobalConfig.triRefMemDataWidth.W)))

    val mem_req: DecoupledIO[UInt] = Decoupled(UInt(c.addrWidth.W))
    val mem_req_mask: DecoupledIO[UInt] = Decoupled(UInt(c.numPEs.W))
    val mem_resp: DecoupledIO[TriangleBlock] = Flipped(Decoupled(new TriangleBlock(c)))

    val start_ready: Bool = Output(Bool())
    val output_ready: Bool = Output(Bool())
    val out_best_hit: Bool = Output(Bool())
    val hit_id: UInt = Output(UInt(c.addrWidth.W))
    val t_best: UInt = Output(UInt(c.cfg.totalWidth.W))
    val out_meta: RayMeta = Output(new RayMeta(c.addrWidth))
    val out_done: Bool = Output(Bool())
  })

  private val current_batch = RegInit(0.U.asTypeOf(new TriBatch(c.addrWidth)))
  private val ray_reg = RegInit(0.U.asTypeOf(new Ray(c.cfg)))
  private val ray_meta_reg = RegInit(0.U.asTypeOf(new RayMeta(c.addrWidth)))
  private val batch_ref_idx = RegInit(0.U(c.addrWidth.W))
  private val batch_refs_remaining = RegInit(0.U(16.W))
  private val batch_in_progress = RegInit(false.B)
  private val no_more_batches = RegInit(false.B)
  private val done_pulse_reg = RegInit(false.B)
  private val active_epoch = RegInit(false.B)
  private val result_capture_pending = RegInit(false.B)
  private val result_hit_reg = RegInit(false.B)
  private val result_id_reg = RegInit(0.U(c.addrWidth.W))
  private val result_t_reg = RegInit(0x7F7FFFFF.U(c.cfg.totalWidth.W))
  private val result_meta_reg = RegInit(0.U.asTypeOf(new RayMeta(c.addrWidth)))

  private val ref_buffer_a = Reg(Vec(refPackFactor, UInt(triIdWidth.W)))
  private val ref_buffer_b = Reg(Vec(refPackFactor, UInt(triIdWidth.W)))
  private val ref_buf_idx_a = RegInit(0.U(refPackShift.W))
  private val ref_buf_idx_b = RegInit(0.U(refPackShift.W))
  private val ref_buf_count_a = RegInit(0.U(refCountWidth.W))
  private val ref_buf_count_b = RegInit(0.U(refCountWidth.W))

  private val best_t = RegInit(0x7F7FFFFF.U(c.cfg.totalWidth.W))
  private val best_id = RegInit(0.U(c.addrWidth.W))
  private val has_hit = RegInit(false.B)

  private val s_IDLE :: s_BUSY :: Nil = Enum(2)
  private val state = RegInit(s_IDLE)

  when(io.flush) {
    current_batch := 0.U.asTypeOf(new TriBatch(c.addrWidth))
    batch_ref_idx := 0.U
    batch_refs_remaining := 0.U
    batch_in_progress := false.B
    no_more_batches := false.B
    done_pulse_reg := false.B
    result_capture_pending := false.B
    result_hit_reg := false.B
    result_id_reg := 0.U
    result_t_reg := 0x7F7FFFFF.U
    result_meta_reg := 0.U.asTypeOf(new RayMeta(c.addrWidth))
    ref_buf_idx_a := 0.U
    ref_buf_idx_b := 0.U
    ref_buf_count_a := 0.U
    ref_buf_count_b := 0.U
    best_t := 0x7F7FFFFF.U
    best_id := 0.U
    has_hit := false.B
    active_epoch := !active_epoch
    state := s_IDLE
  }.elsewhen(state === s_IDLE && io.ray_valid) {
    state := s_BUSY
    ray_reg := io.ray_in
    ray_meta_reg := io.ray_meta
    no_more_batches := false.B
  }

  when(io.end_exec && !io.flush) {
    no_more_batches := true.B
  }

  private val canAcceptBatch =
    !batch_in_progress && !result_capture_pending && !done_pulse_reg && (state === s_BUSY)

  when(io.tri_batch_valid && canAcceptBatch && !io.flush) {
    current_batch := io.tri_batch_in
    batch_ref_idx := io.tri_batch_in.base_addr
    batch_refs_remaining := io.tri_batch_in.count
    batch_in_progress := true.B
    ref_buf_idx_a := 0.U
    ref_buf_idx_b := 0.U
    ref_buf_count_a := 0.U
    ref_buf_count_b := 0.U
    best_t := 0x7F7FFFFF.U
    best_id := 0.U
    has_hit := false.B
  }

  private val refReqPendingLive = Wire(Bool())
  private val activeBufSel = Wire(Bool())
  private val activeBufIdx = Wire(UInt(refPackShift.W))
  private val activeBufCount = Wire(UInt(refCountWidth.W))
  private val prefetchBufEmpty = Wire(Bool())
  private val requestToBufB = Wire(Bool())
  private val sourceHasData = Wire(Bool())

  activeBufSel := ref_buf_count_a === 0.U && ref_buf_count_b =/= 0.U
  activeBufIdx := Mux(activeBufSel, ref_buf_idx_b, ref_buf_idx_a)
  activeBufCount := Mux(activeBufSel, ref_buf_count_b, ref_buf_count_a)
  sourceHasData := ref_buf_count_a =/= 0.U || ref_buf_count_b =/= 0.U
  prefetchBufEmpty := Mux(activeBufSel, ref_buf_count_a === 0.U, ref_buf_count_b === 0.U)
  requestToBufB := Mux(sourceHasData, !activeBufSel, false.B)

  private val refWordAddr = batch_ref_idx >> refPackShift
  private val refWordOff = batch_ref_idx(refPackShift - 1, 0)
  private val refChunkAvailWide = refPackFactor.U(16.W) - refWordOff
  private val refChunkCountWide = Mux(batch_refs_remaining < refChunkAvailWide, batch_refs_remaining, refChunkAvailWide)
  private val refChunkCount = refChunkCountWide(refCountWidth - 1, 0)
  private val needInitialFill = !sourceHasData
  private val needPrefetch =
    sourceHasData &&
      prefetchBufEmpty &&
      (activeBufCount <= refPrefetchLowWatermark.U)
  io.ref_mem_req.valid := batch_in_progress &&
    (batch_refs_remaining =/= 0.U) &&
    !refReqPendingLive &&
    !io.flush &&
    (needInitialFill || needPrefetch)
  io.ref_mem_req.bits := refWordAddr

  when(io.ref_mem_req.fire) {
    batch_ref_idx := batch_ref_idx + refChunkCount
    batch_refs_remaining := batch_refs_remaining - refChunkCount
  }

  io.ref_mem_resp.ready := true.B

  private val refReqPipeLen = GlobalConfig.triRefMemDpiLatency
  private val ref_req_live_pipe = RegInit(VecInit(Seq.fill(refReqPipeLen)(false.B)))
  private val ref_req_epoch_pipe = RegInit(VecInit(Seq.fill(refReqPipeLen)(false.B)))
  private val ref_req_offset_pipe = RegInit(VecInit(Seq.fill(refReqPipeLen)(0.U(refPackShift.W))))
  private val ref_req_count_pipe = RegInit(VecInit(Seq.fill(refReqPipeLen)(0.U(refCountWidth.W))))
  private val ref_req_target_b_pipe = RegInit(VecInit(Seq.fill(refReqPipeLen)(false.B)))
  when(io.flush) {
    for (i <- 0 until refReqPipeLen) {
      ref_req_live_pipe(i) := false.B
      ref_req_epoch_pipe(i) := false.B
      ref_req_offset_pipe(i) := 0.U
      ref_req_count_pipe(i) := 0.U
      ref_req_target_b_pipe(i) := false.B
    }
  }.otherwise {
    ref_req_live_pipe(0) := io.ref_mem_req.fire
    ref_req_epoch_pipe(0) := Mux(io.ref_mem_req.fire, active_epoch, false.B)
    ref_req_offset_pipe(0) := Mux(io.ref_mem_req.fire, refWordOff, 0.U)
    ref_req_count_pipe(0) := Mux(io.ref_mem_req.fire, refChunkCount, 0.U)
    ref_req_target_b_pipe(0) := Mux(io.ref_mem_req.fire, requestToBufB, false.B)
    for (i <- 1 until refReqPipeLen) {
      ref_req_live_pipe(i) := ref_req_live_pipe(i - 1)
      ref_req_epoch_pipe(i) := ref_req_epoch_pipe(i - 1)
      ref_req_offset_pipe(i) := ref_req_offset_pipe(i - 1)
      ref_req_count_pipe(i) := ref_req_count_pipe(i - 1)
      ref_req_target_b_pipe(i) := ref_req_target_b_pipe(i - 1)
    }
  }
  refReqPendingLive := ref_req_live_pipe.asUInt.orR

  private val ref_resp_live =
    io.ref_mem_resp.fire &&
      ref_req_live_pipe.last &&
      (ref_req_epoch_pipe.last === active_epoch) &&
      !io.flush

  private val refWordIds = Wire(Vec(refPackFactor, UInt(triIdWidth.W)))
  for (i <- 0 until refPackFactor) {
    refWordIds(i) := io.ref_mem_resp.bits((i + 1) * triIdWidth - 1, i * triIdWidth)
  }
  private val compactedIds = Wire(Vec(refPackFactor, UInt(triIdWidth.W)))
  for (i <- 0 until refPackFactor) {
    compactedIds(i) := 0.U
    for (off <- 0 until refPackFactor) {
      if (off + i < refPackFactor) {
        when(ref_req_offset_pipe.last === off.U) {
          compactedIds(i) := refWordIds(off + i)
        }
      }
    }
  }

  when(ref_resp_live) {
    when(ref_req_target_b_pipe.last) {
      for (i <- 0 until refPackFactor) {
        ref_buffer_b(i) := compactedIds(i)
      }
      ref_buf_idx_b := 0.U
      ref_buf_count_b := ref_req_count_pipe.last
    }.otherwise {
      for (i <- 0 until refPackFactor) {
        ref_buffer_a(i) := compactedIds(i)
      }
      ref_buf_idx_a := 0.U
      ref_buf_count_a := ref_req_count_pipe.last
    }
  }

  private val geomTriId = Mux(activeBufSel, ref_buffer_b(activeBufIdx), ref_buffer_a(activeBufIdx))

  class MemReq extends Bundle {
    val addr = UInt(GlobalConfig.triMemAddrWidth.W)
    val epoch = Bool()
  }

  val memReqQ = Module(new Queue(new MemReq, 2))
  memReqQ.io.enq.valid := batch_in_progress && sourceHasData && !io.flush
  memReqQ.io.enq.bits.addr := geomTriId
  memReqQ.io.enq.bits.epoch := active_epoch

  val memReqStale = memReqQ.io.deq.valid && (memReqQ.io.deq.bits.epoch =/= active_epoch)
  val memReqToMemValid = memReqQ.io.deq.valid && !memReqStale && !io.flush
  val memReqToMemReady = io.mem_req.ready && io.mem_req_mask.ready

  io.mem_req.valid := memReqToMemValid
  io.mem_req.bits := memReqQ.io.deq.bits.addr
  io.mem_req_mask.valid := memReqToMemValid
  io.mem_req_mask.bits := 1.U
  memReqQ.io.deq.ready := io.flush || memReqStale || (memReqToMemValid && memReqToMemReady)

  when(memReqQ.io.enq.fire && !io.flush) {
    when(activeBufSel) {
      ref_buf_idx_b := ref_buf_idx_b + 1.U
      ref_buf_count_b := ref_buf_count_b - 1.U
    }.otherwise {
      ref_buf_idx_a := ref_buf_idx_a + 1.U
      ref_buf_count_a := ref_buf_count_a - 1.U
    }
  }

  io.mem_resp.ready := true.B

  private val memReqPipeLen = GlobalConfig.triMemDpiLatency
  private val mem_req_live_pipe = RegInit(VecInit(Seq.fill(memReqPipeLen)(false.B)))
  private val mem_req_epoch_pipe = RegInit(VecInit(Seq.fill(memReqPipeLen)(false.B)))
  private val memReqIssued = memReqQ.io.deq.fire && !memReqStale && !io.flush
  when(io.flush) {
    for (i <- 0 until memReqPipeLen) {
      mem_req_live_pipe(i) := false.B
      mem_req_epoch_pipe(i) := false.B
    }
  }.otherwise {
    mem_req_live_pipe(0) := memReqIssued
    mem_req_epoch_pipe(0) := Mux(memReqIssued, memReqQ.io.deq.bits.epoch, false.B)
    for (i <- 1 until memReqPipeLen) {
      mem_req_live_pipe(i) := mem_req_live_pipe(i - 1)
      mem_req_epoch_pipe(i) := mem_req_epoch_pipe(i - 1)
    }
  }
  private val mem_req_live = mem_req_live_pipe.last
  private val mem_req_epoch = mem_req_epoch_pipe.last
  private val mem_resp_live =
    io.mem_resp.fire &&
      mem_req_live &&
      (mem_req_epoch === active_epoch) &&
      !io.flush

  private val pe = Module(new RayTriangleIntersection(c.cfg))
  pe.io.ray := ray_reg
  pe.io.tri := io.mem_resp.bits.tris(0)
  pe.io.in_valid := mem_resp_live && io.mem_resp.bits.mask(0)

  val fcmp = Module(new FCMP(c.cfg))
  fcmp.io.a := pe.io.t
  fcmp.io.b := best_t
  fcmp.io.signaling := false.B
  when(pe.io.out_valid && pe.io.hit) {
    when(fcmp.io.lt || !has_hit) {
      best_t := pe.io.t
      best_id := pe.io.id
      has_hit := true.B
    }
  }

  private val inflight_cnt = RegInit(0.U(10.W))
  private val incoming_count = Mux(mem_resp_live && io.mem_resp.bits.mask(0), 1.U(10.W), 0.U(10.W))
  private val outgoing_count = Mux(pe.io.out_valid, 1.U(10.W), 0.U(10.W))
  private val inflight_next = inflight_cnt + incoming_count - outgoing_count
  private val batch_source_drained =
    (batch_refs_remaining === 0.U) &&
      (ref_buf_count_a === 0.U) &&
      (ref_buf_count_b === 0.U) &&
      !refReqPendingLive &&
      !memReqQ.io.deq.valid &&
      !mem_req_live_pipe.asUInt.orR
  private val batch_done_now =
    batch_in_progress &&
      batch_source_drained &&
      (inflight_next === 0.U)

  when(io.flush) {
    inflight_cnt := 0.U
    done_pulse_reg := false.B
  }.otherwise {
    inflight_cnt := inflight_next
    done_pulse_reg := false.B
  }

  when(batch_done_now && !io.flush) {
    batch_in_progress := false.B
    result_capture_pending := true.B
  }

  when(state === s_BUSY &&
    no_more_batches &&
    !io.tri_batch_valid &&
    !batch_in_progress &&
    !io.flush) {
    state := s_IDLE
  }

  when(result_capture_pending && !io.flush) {
    result_capture_pending := false.B
    result_hit_reg := has_hit
    result_id_reg := best_id
    result_t_reg := best_t
    result_meta_reg := ray_meta_reg
    done_pulse_reg := true.B
  }

  io.start_ready := state === s_IDLE
  io.output_ready := canAcceptBatch
  io.out_best_hit := result_hit_reg
  io.hit_id := result_id_reg
  io.t_best := result_t_reg
  io.out_meta := result_meta_reg
  io.out_done := done_pulse_reg
}
