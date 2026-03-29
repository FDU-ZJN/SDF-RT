import DDA.DDA
import Render.RenderStage
import SDF.{InitStage, SdfStage}
import chisel3._
import chisel3.util._
import raytrace_utils._

class SimTop extends Module {
  val c = TriPeConfig()
  val io = IO(new Bundle {
    val setup_valid = Input(Bool())
    val setup_origin = Input(new Vec3(c.cfg))
    val setup_grid_min = Input(new Vec3(c.cfg))
    val setup_grid_max = Input(new Vec3(c.cfg))
    val setup_finish = Output(Bool())
    val rd_in = Input(new Vec3(c.cfg))
    val rd_valid = Input(Bool())
    val out_ready = Output(Bool())
    val out_rgb = Output(new Vec3(c.cfg))
    val out_id = Output(UInt(c.addrWidth.W))
    val out_valid = Output(Bool())
  })

  val initStage = Module(new InitStage(c.cfg, c.addrWidth))
  val sdfStage = Module(new SdfStage(c.cfg, c.addrWidth))
  val ddaStage = Module(new DDA(c.cfg, c.addrWidth, globalRes = 16, subRes = 16, maxTraversalSteps = 256))
  val renderStage = Module(new RenderStage(c.cfg))
  val commitQueue = Module(new CommitQueue(c.cfg))

  // Init->SDF decouple queue; depth is sized to absorb InitStage fixed-latency burst safely.
  val initToSdfDepth = 32
  val initToSdfQ = Module(new Queue(new RayIssue(c.cfg, c.addrWidth), initToSdfDepth))

  sdfStage.io.setup_valid := io.setup_valid
  sdfStage.io.setup_origin := io.setup_origin
  sdfStage.io.setup_grid_min := io.setup_grid_min
  sdfStage.io.setup_grid_max := io.setup_grid_max
  io.setup_finish := sdfStage.io.setup_finish

  initStage.io.setup_origin := io.setup_origin
  initStage.io.setup_grid_min := io.setup_grid_min
  initStage.io.setup_grid_max := io.setup_grid_max

  // Conservative admission control for InitStage input:
  // assume all in-flight rays may become SDF-hit and need Init->SDF queue space.
  val initOutLatency = 4 + (3 * c.cfg.faddLatency) + (2 * c.cfg.fmulLatency) + c.cfg.fdivLatency
  val inflightW = log2Ceil(initToSdfDepth + initOutLatency + 4)
  val initInflight = RegInit(0.U(inflightW.W))

  val initSdfOutFire = initStage.io.to_sdf.fire
  val initBypassOutFire = initStage.io.to_bypass.fire
  val initOutFireAny = initSdfOutFire || initBypassOutFire

  val sdfQCountExt = Wire(UInt(inflightW.W))
  sdfQCountExt := initToSdfQ.io.count
  val sdfQFree = initToSdfDepth.U(inflightW.W) - sdfQCountExt

  val canReserveSdfQ = sdfQFree > initInflight
  val conservativeInitReady = initStage.io.in.ready && commitQueue.io.alloc.ready && initStage.io.to_bypass.ready && canReserveSdfQ
  val inputFire = io.rd_valid && conservativeInitReady

  when(inputFire && !initOutFireAny) {
    initInflight := initInflight + 1.U
  }.elsewhen(!inputFire && initOutFireAny) {
    initInflight := initInflight - 1.U
  }

  initStage.io.in.valid := io.rd_valid && conservativeInitReady
  initStage.io.in.bits.rd := io.rd_in
  initStage.io.in.bits.meta.slotId := commitQueue.io.allocSlot
  initStage.io.in.bits.meta.pixelX := 0.U
  initStage.io.in.bits.meta.pixelY := 0.U

  commitQueue.io.alloc.valid := inputFire
  commitQueue.io.alloc.bits := 0.U

  initToSdfQ.io.enq <> initStage.io.to_sdf
  sdfStage.io.issue_in <> initToSdfQ.io.deq

  ddaStage.io.grid_min := sdfStage.io.grid_min
  ddaStage.io.inv_voxel := sdfStage.io.inv_voxel

  val sdfHitQ = Module(new Queue(new DdaTraversalReq(c.cfg, c.addrWidth), 16))
  sdfHitQ.io.enq.valid := sdfStage.io.out_valid && sdfStage.io.out_hit
  sdfHitQ.io.enq.bits.ray := sdfStage.io.out_ray
  sdfHitQ.io.enq.bits.meta := sdfStage.io.out_meta
  sdfHitQ.io.enq.bits.reverseTraversal := sdfStage.io.out_reverseTraversal
  when(sdfHitQ.io.enq.valid) {
    assert(sdfHitQ.io.enq.ready, "SimTop sdfHitQ overflow")
  }

  ddaStage.io.in <> sdfHitQ.io.deq

  renderStage.io.in.valid := ddaStage.io.out.valid
  renderStage.io.in.bits.meta := ddaStage.io.out.bits.meta
  renderStage.io.in.bits.hit := ddaStage.io.out.bits.hit
  renderStage.io.in.bits.hitId := ddaStage.io.out.bits.hitId
  renderStage.io.in.bits.hitT := ddaStage.io.out.bits.hitT
  ddaStage.io.out.ready := renderStage.io.in.ready

  val zeroFp = 0.U(c.cfg.totalWidth.W)
  val oneFp = java.lang.Float.floatToRawIntBits(1.0f).U(c.cfg.totalWidth.W)
  val deepBlueFp = java.lang.Float.floatToRawIntBits(0.5f).U(c.cfg.totalWidth.W)

  commitQueue.io.writeback.valid := renderStage.io.out.valid
  commitQueue.io.writeback.bits := renderStage.io.out.bits
  renderStage.io.out.ready := commitQueue.io.writeback.ready

  val wb2 = Wire(new RenderResult(c.cfg, c.addrWidth))
  wb2.meta := initStage.io.to_bypass.bits.meta
  wb2.hit := false.B
  wb2.hitId := 0.U
  wb2.rgb.x := zeroFp
  wb2.rgb.y := oneFp
  wb2.rgb.z := zeroFp
  commitQueue.io.writeback2.valid := initStage.io.to_bypass.valid
  commitQueue.io.writeback2.bits := wb2
  initStage.io.to_bypass.ready := commitQueue.io.writeback2.ready

  val wb3 = Wire(new RenderResult(c.cfg, c.addrWidth))
  wb3.meta := sdfStage.io.out_meta
  wb3.hit := false.B
  wb3.hitId := 0.U
  wb3.rgb.x := zeroFp
  wb3.rgb.y := zeroFp
  wb3.rgb.z := deepBlueFp
  commitQueue.io.writeback3.valid := sdfStage.io.out_valid && !sdfStage.io.out_hit
  commitQueue.io.writeback3.bits := wb3

  commitQueue.io.out.ready := true.B

  io.out_ready := conservativeInitReady
  io.out_rgb := commitQueue.io.out.bits.rgb
  io.out_id := commitQueue.io.out.bits.hitId
  io.out_valid := commitQueue.io.out.valid
}

object SimTopGen extends App {
  emitVerilog(new SimTop(), Array("--target-dir", "build"))
}