import DDA.DDA
import Render.RenderStage
import SDF.{InitStage, SetupUnit, SdfStage, SdfMemWriteIO}
import chisel3._
import chisel3.util._
import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Paths}
import raytrace_utils._
import scala.io.Source

class SimTop extends Module {
  val c = TriPeConfig(cfg = FloatConfig.FP32.copy())
  private val sdfCfg = SdfPeConfig(cfg = c.cfg)
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
    
    // SDF memory write port for PS initialization
    val sdf_mem_wr = Flipped(new SdfMemWriteIO)
  })

  val initStage = Module(new InitStage(c.cfg, c.addrWidth))
  val setupUnit = Module(new SetupUnit(c.cfg, sdfCfg))
  val sdfStage = Module(new SdfStage(c.cfg, c.addrWidth))
  val ddaStage = Module(new DDA(c.cfg, c.addrWidth, globalRes = sdfCfg.DDAGlobalRes, subRes = sdfCfg.SubRes, maxTraversalSteps = sdfCfg.DDAMaxSteps))
  val renderStage = Module(new RenderStage(c.cfg))
  val commitQueue = Module(new CommitQueue(c.cfg))

  // Init->SDF decouple queue; depth is sized to absorb InitStage fixed-latency burst safely.
  val initToSdfDepth = GlobalConfig.simInitToSdfQueueDepth
  val initToSdfQ = Module(new Queue(new RayIssue(c.cfg, c.addrWidth), initToSdfDepth))

  setupUnit.io.setup_valid := io.setup_valid
  setupUnit.io.setup_origin := io.setup_origin
  setupUnit.io.setup_grid_min := io.setup_grid_min
  setupUnit.io.setup_grid_max := io.setup_grid_max
  io.setup_finish := setupUnit.io.setup_finish

  initStage.io.setup_origin := setupUnit.io.origin
  initStage.io.setup_grid_min := setupUnit.io.gridMin
  initStage.io.setup_grid_max := setupUnit.io.gridMax

  sdfStage.io.grid_min := setupUnit.io.gridMin
  sdfStage.io.inv_voxel := setupUnit.io.invVoxel
  sdfStage.io.sdf_mem_wr <> io.sdf_mem_wr

  // Conservative admission control for InitStage input:
  // assume all in-flight rays may become SDF-hit and need Init->SDF queue space.
  val initOutLatency = 4 + (3 * c.cfg.faddLatency) + (2 * c.cfg.fmulLatency) + c.cfg.fdivLatency + (4 * c.cfg.fcmpLatency)
  val inflightW = log2Ceil(initToSdfDepth + initOutLatency + 4)
  val initInflight = RegInit(0.U(inflightW.W))

  val initSdfOutFire = initStage.io.to_sdf.fire
  val initBypassOutFire = initStage.io.to_bypass.fire
  val initOutFireAny = initSdfOutFire || initBypassOutFire

  val sdfQCountExt = Wire(UInt(inflightW.W))
  sdfQCountExt := initToSdfQ.io.count
  val sdfQFree = initToSdfDepth.U(inflightW.W) - sdfQCountExt

  val setupReady = setupUnit.io.setup_finish
  val canReserveSdfQ = sdfQFree > initInflight
  val conservativeInitReady = setupReady && initStage.io.in.ready && commitQueue.io.alloc.ready && initStage.io.to_bypass.ready && canReserveSdfQ
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
  when(initToSdfQ.io.enq.valid) {
    assert(initToSdfQ.io.enq.ready, "SimTop initToSdfQ overflow")
  }
  sdfStage.io.issue_in <> initToSdfQ.io.deq

  ddaStage.io.grid_min := setupUnit.io.gridMin
  ddaStage.io.inv_sub_voxel := setupUnit.io.invSubVoxel

  val sdfHitQ = Module(new Queue(new DdaTraversalReq(c.cfg, c.addrWidth), GlobalConfig.simSdfHitQueueDepth))
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
  private val bbFileListMarker = "// ----- 8< ----- FILE \"firrtl_black_box_resource_files.f\" ----- 8< -----"

  private def stripFirrtlBbFileList(targetDir: String): Unit = {
    val outPath = Paths.get(targetDir, "SimTop.sv")
    if (!Files.exists(outPath)) return

    val src = Source.fromFile(outPath.toFile, "UTF-8")
    val text = try src.mkString finally src.close()
    val markerIdx = text.indexOf(bbFileListMarker)
    if (markerIdx >= 0) {
      val trimmed = text.substring(0, markerIdx).trim + "\n"
      Files.write(outPath, trimmed.getBytes(StandardCharsets.UTF_8))
    }
  }

  def generateSimTopVerilog(memImplMode: Int, useFloatIP: Boolean, targetDir: String): Unit = {
    GlobalConfig.withMemImplMode(memImplMode) {
      GlobalConfig.withUseFloatIP(useFloatIP) {
        emitVerilog(new SimTop, Array("--target-dir", targetDir))
      }
    }
    // Remove trailing firrtl blackbox resource file-list payload from combined output.
    stripFirrtlBbFileList(targetDir)
  }

  // Emit both variants into build/ subdirectories.
  // memImplMode: 0=DPI-C, 1=readmemh BlackBox, 2=IP-style memory modules
  generateSimTopVerilog(memImplMode = 0, useFloatIP = false, "build/noblackbox")
  generateSimTopVerilog(memImplMode = 1, useFloatIP = false, "build/useblackbox")
  
  // Generate FPGA_TOP for FPGA deployment
  FpgaTopGen.generateFpgaTopVerilog("build/fpga")
}