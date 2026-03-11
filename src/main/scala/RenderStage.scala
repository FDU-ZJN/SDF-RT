package sdf_rt
import chisel3._
import chisel3.util._
import raytrace_utils._
import raytrace_utils.fudian._

class RenderStage(cfg: FloatConfig) extends Module {
  val io = IO(new Bundle {
    val in_result = Input(new TraceResult(cfg, cfg.addrWidth))
    val in_valid = Input(Bool())
    val in_ready = Output(Bool())

    // 输出到下游（例如 FrameBuffer 或像素写回）
    val out_result = Output(new RenderResult(cfg, cfg.addrWidth))
    val out_valid = Output(Bool())
  })
  // 1. 实例化核心计算单元 (PE)
  val pe = Module(new RenderPE(cfg))

  // 2. 实例化法线存储器 (DPI)
  val mem = Module(new NormalMemDPI(cfg.addrWidth))
  mem.io.clk   := clock
  mem.io.reset := reset

  val reqQ = Module(new Queue(new TraceResult(cfg, cfg.addrWidth), 4))
  reqQ.io.enq.bits := io.in_result
  reqQ.io.enq.valid := io.in_valid
  io.in_ready := reqQ.io.enq.ready

  val zeroNormal = 0.U.asTypeOf(new Vec3(cfg))
  val zeroMeta = 0.U.asTypeOf(new RayMeta(cfg.addrWidth))
  val pendingHitValid = RegInit(false.B)
  val pendingHit = Reg(new TraceResult(cfg, cfg.addrWidth))

  val launchValid = WireDefault(false.B)
  val launchHit = WireDefault(false.B)
  val launchId = WireDefault(0.U(cfg.addrWidth.W))
  val launchMeta = WireDefault(zeroMeta)
  val launchNormal = WireDefault(zeroNormal)

  mem.io.addr := 0.U
  mem.io.en := false.B
  reqQ.io.deq.ready := false.B

  when(pendingHitValid && mem.io.valid) {
    launchValid := true.B
    launchHit := true.B
    launchId := pendingHit.hitId
    launchMeta := pendingHit.meta
    pendingHitValid := false.B
  }.elsewhen(!pendingHitValid && reqQ.io.deq.valid) {
    reqQ.io.deq.ready := true.B
    launchId := reqQ.io.deq.bits.hitId
    launchMeta := reqQ.io.deq.bits.meta
    when(reqQ.io.deq.bits.hit) {
      mem.io.addr := reqQ.io.deq.bits.hitId
      mem.io.en := true.B
      when(reqQ.io.deq.fire) {
        pendingHitValid := true.B
        pendingHit := reqQ.io.deq.bits
      }
    }.otherwise {
      launchValid := reqQ.io.deq.fire
      launchHit := false.B
    }
  }

  pe.io.in_meta := launchMeta
  pe.io.hit_id := launchId
  pe.io.in_hit := launchHit

  // B. 处理内存返回的数据并送回 PE
  // NormalMemDPI 返回的是 96 位原始数据，我们需要将其解包为 Vec3 浮点向量
  val normal_from_mem = Wire(new Vec3(cfg))
  normal_from_mem.x := mem.io.data(31, 0)
  normal_from_mem.y := mem.io.data(63, 32)
  normal_from_mem.z := mem.io.data(95, 64)

  launchNormal := Mux(launchHit, normal_from_mem, zeroNormal)
  pe.io.in_normal := launchNormal
  pe.io.in_valid  := launchValid

  // C. 输出最终计算结果
  io.out_result := pe.io.out_result
  io.out_valid := pe.io.out_valid
}
