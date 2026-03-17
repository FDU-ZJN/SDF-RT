import SDF.{InitStage, SdfStage}
import chisel3._
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
    val out_valid = Output(Bool())
  })

  val initStage = Module(new InitStage(c.cfg, c.addrWidth))
  val sdfStage = Module(new SdfStage(c.cfg, c.addrWidth))
  val commitQueue = Module(new CommitQueue(c.cfg))

  sdfStage.io.setup_valid := io.setup_valid
  sdfStage.io.setup_origin := io.setup_origin
  sdfStage.io.setup_grid_min := io.setup_grid_min
  sdfStage.io.setup_grid_max := io.setup_grid_max
  io.setup_finish := sdfStage.io.setup_finish

  initStage.io.setup_origin := io.setup_origin
  initStage.io.setup_grid_min := io.setup_grid_min
  initStage.io.setup_grid_max := io.setup_grid_max

  val inputReady = initStage.io.in.ready && commitQueue.io.alloc.ready
  val inputFire = io.rd_valid && inputReady

  initStage.io.in.valid := io.rd_valid && commitQueue.io.alloc.ready
  initStage.io.in.bits.rd := io.rd_in
  initStage.io.in.bits.meta.slotId := commitQueue.io.allocSlot
  initStage.io.in.bits.meta.pixelX := 0.U
  initStage.io.in.bits.meta.pixelY := 0.U

  commitQueue.io.alloc.valid := inputFire
  commitQueue.io.alloc.bits := 0.U

  sdfStage.io.issue_in.valid := initStage.io.to_sdf.valid
  sdfStage.io.issue_in.bits := initStage.io.to_sdf.bits
  initStage.io.to_sdf.ready := sdfStage.io.issue_in.ready

  val zeroFp = 0.U(c.cfg.totalWidth.W)

  val wb = Wire(new RenderResult(c.cfg, c.addrWidth))
  wb.meta := sdfStage.io.out_meta
  wb.hit := sdfStage.io.out_rgb.x =/= zeroFp
  wb.hitId := 0.U
  wb.rgb := sdfStage.io.out_rgb
  commitQueue.io.writeback.valid := sdfStage.io.out_valid
  commitQueue.io.writeback.bits := wb

  val wb2 = Wire(new RenderResult(c.cfg, c.addrWidth))
  wb2.meta := initStage.io.to_bypass.bits.meta
  wb2.hit := false.B
  wb2.hitId := 0.U
  wb2.rgb.x := zeroFp
  wb2.rgb.y := zeroFp
  wb2.rgb.z := zeroFp
  commitQueue.io.writeback2.valid := initStage.io.to_bypass.valid
  commitQueue.io.writeback2.bits := wb2
  initStage.io.to_bypass.ready := commitQueue.io.writeback2.ready

  commitQueue.io.out.ready := true.B

  io.out_ready := inputReady
  io.out_rgb := commitQueue.io.out.bits.rgb
  io.out_valid := commitQueue.io.out.valid
}
object SimTopGen extends App {
  emitVerilog(new SimTop(), Array("--target-dir", "build"))
}