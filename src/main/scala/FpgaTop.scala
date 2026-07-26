import chisel3._
import chisel3.util._
import raytrace_utils._
import Backend.Backend
import SDF.{FrontendContextManager, InitStage, SdfMemWriteIO, SdfStage, SetupUnit}
import Trace.TriMemDmaReadData

import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Paths}
import java.time.Instant
import scala.io.Source

class FpgaTop extends Module {
  val c = TriPeConfig(cfg = FloatConfig.FP32.copy())
  private val sdfCfg = SdfPeConfig(cfg = c.cfg)
  private val missT = "h7F800000".U(32.W)

  val io = IO(new Bundle {
    val setup_valid = Input(Bool())
    val setup_origin = Input(new Vec3(c.cfg))
    val setup_grid_min = Input(new Vec3(c.cfg))
    val setup_grid_max = Input(new Vec3(c.cfg))
    val setup_ready = Output(Bool())

    val ray_in_valid = Input(Bool())
    val ray_in = Input(new RayIngress(c.cfg))
    val ray_in_last = Input(Bool())
    val ray_in_ready = Output(Bool())

    val result_valid = Output(Bool())
    val result = Output(new RenderResult(c.cfg, c.addrWidth))
    val result_last = Output(Bool())
    val result_ready = Input(Bool())
    val batch_done = Output(Bool())
    val protocol_error = Output(Bool())

    val sdf_mem_wr = Flipped(new SdfMemWriteIO)
    val triangle_base_address = Input(UInt(64.W))
    val tri_dma_cmd = Decoupled(UInt(80.W))
    val tri_dma_data = Flipped(Decoupled(new TriMemDmaReadData(128)))
    val tri_dma_status = Flipped(Decoupled(UInt(8.W)))
    val tri_dma_read_error = Output(Bool())
    val tri_dma_malformed_line_count = Output(UInt(32.W))
    val tri_dma_status_error_count = Output(UInt(32.W))
    val tri_dma_tag_mismatch_count = Output(UInt(32.W))
    val tri_axi_issued_count = Output(UInt(32.W))
    val tri_axi_completed_count = Output(UInt(32.W))
    val tri_axi_outstanding_count = Output(UInt(8.W))
  })

  val setupUnit = Module(new SetupUnit(c.cfg, sdfCfg))
  val initStage = Module(new InitStage(c.cfg, c.addrWidth))
  val sdfStage = Module(new SdfStage(c.cfg, c.addrWidth))
  val frontendContexts = Module(new FrontendContextManager(c.cfg, c.addrWidth,
    readPorts = 4, releasePorts = 3, stateWritePorts = 2))
  val backend = Module(new Backend(c, sdfCfg.DDAGlobalRes, sdfCfg.SubRes, sdfCfg.DDAMaxSteps))
  val earlyQ = Module(new BufferedRRArbiter(new RenderResult(c.cfg, c.addrWidth), 2, GlobalConfig.resultArbiterFifoDepth))
  val traceQ = Module(new BufferedRRArbiter(new RenderResult(c.cfg, c.addrWidth), GlobalConfig.traceNumWorkers, GlobalConfig.resultArbiterFifoDepth))
  val resultQ = Module(new BufferedRRArbiter(new RenderResult(c.cfg, c.addrWidth), 2, GlobalConfig.resultArbiterFifoDepth))

  val setupPending = RegInit(false.B)
  val setupOrigin = Reg(new Vec3(c.cfg))
  val setupGridMin = Reg(new Vec3(c.cfg))
  val setupGridMax = Reg(new Vec3(c.cfg))
  when(io.setup_valid && !setupPending) {
    setupOrigin := io.setup_origin
    setupGridMin := io.setup_grid_min
    setupGridMax := io.setup_grid_max
    setupPending := true.B
  }.elsewhen(setupUnit.io.setup_finish) {
    setupPending := false.B
  }
  setupUnit.io.setup_valid := setupPending
  setupUnit.io.setup_origin := setupOrigin
  setupUnit.io.setup_grid_min := setupGridMin
  setupUnit.io.setup_grid_max := setupGridMax
  io.setup_ready := !setupPending

  initStage.io.setup_origin := setupUnit.io.origin
  initStage.io.setup_grid_min := setupUnit.io.gridMin
  initStage.io.setup_grid_max := setupUnit.io.gridMax
  sdfStage.io.grid_min := setupUnit.io.gridMin
  sdfStage.io.inv_voxel := setupUnit.io.invVoxel
  sdfStage.io.inv_sub_voxel := setupUnit.io.invSubVoxel
  sdfStage.io.sdf_mem_wr <> io.sdf_mem_wr

  val batchClosed = RegInit(false.B)
  val acceptedCount = RegInit(0.U(32.W))
  val emittedCount = RegInit(0.U(32.W))
  val protocolError = RegInit(false.B)
  val inputCanAccept = frontendContexts.io.ingress.ready && !batchClosed && setupUnit.io.setup_finish
  io.ray_in_ready := inputCanAccept
  val inputFire = io.ray_in_valid && io.ray_in_ready
  when(io.ray_in_valid && batchClosed) { protocolError := true.B }
  when(inputFire) {
    acceptedCount := acceptedCount + 1.U
    when(io.ray_in_last) { batchClosed := true.B }
  }
  frontendContexts.io.ingress.valid := io.ray_in_valid && inputCanAccept
  frontendContexts.io.ingress.bits := io.ray_in
  initStage.io.in <> frontendContexts.io.init_issue
  frontendContexts.io.read_id(0) := initStage.io.context_id
  initStage.io.immutable := frontendContexts.io.immutable(0)

  def missResult(rayId: UInt): RenderResult = {
    val r = Wire(new RenderResult(c.cfg, c.addrWidth))
    r.meta.rayId := rayId
    r.hit := false.B
    r.hitId := 0.U
    r.hitT := missT
    r.rgb8 := 0.U
    r
  }
  val initOut = initStage.io.out
  frontendContexts.io.read_id(1) := initOut.bits.contextId
  earlyQ.io.enq(0).valid := initOut.valid && !initOut.bits.hit
  earlyQ.io.enq(0).bits := missResult(frontendContexts.io.immutable(1).meta.rayId)
  sdfStage.io.issue_in.valid := initOut.valid && initOut.bits.hit
  sdfStage.io.issue_in.bits := initOut.bits.contextId
  initOut.ready := Mux(initOut.bits.hit, sdfStage.io.issue_in.ready, earlyQ.io.enq(0).ready)

  frontendContexts.io.state_write(0).valid := initOut.fire && initOut.bits.hit
  frontendContexts.io.state_write(0).bits.contextId := initOut.bits.contextId
  frontendContexts.io.state_write(0).bits.state.dist := initOut.bits.entryT
  frontendContexts.io.state_write(0).bits.state.iter := 0.U
  frontendContexts.io.state_write(0).bits.state.prevSdf := 0.U

  frontendContexts.io.read_id(2) := sdfStage.io.sdf_context_id
  sdfStage.io.sdf_immutable := frontendContexts.io.immutable(2)
  sdfStage.io.sdf_state := frontendContexts.io.sdf_state(2)
  frontendContexts.io.state_write(1) := sdfStage.io.state_write
  frontendContexts.io.read_id(3) := sdfStage.io.final_context_id
  sdfStage.io.final_immutable := frontendContexts.io.immutable(3)
  sdfStage.io.final_state := frontendContexts.io.sdf_state(3)

  val sdfOut = sdfStage.io.miss_out
  earlyQ.io.enq(1).valid := sdfOut.valid
  earlyQ.io.enq(1).bits := missResult(frontendContexts.io.immutable(3).meta.rayId)
  sdfOut.ready := earlyQ.io.enq(1).ready

  frontendContexts.io.release(0).valid := initOut.fire && !initOut.bits.hit
  frontendContexts.io.release(0).bits := initOut.bits.contextId
  frontendContexts.io.release(1).valid := sdfOut.fire
  frontendContexts.io.release(1).bits := sdfOut.bits.contextId
  frontendContexts.io.release(2).valid := sdfStage.io.dda_context_out.fire
  frontendContexts.io.release(2).bits := sdfStage.io.dda_context_out.bits.sourceFrontendId

  sdfStage.io.dda_context_id <> backend.io.init_alloc
  backend.io.context_in <> sdfStage.io.dda_context_out

  backend.io.triangleBaseAddress := io.triangle_base_address
  io.tri_dma_cmd <> backend.io.triDmaCmd
  backend.io.triDmaData <> io.tri_dma_data
  backend.io.triDmaStatus <> io.tri_dma_status
  io.tri_dma_read_error := backend.io.triDmaReadError
  io.tri_dma_malformed_line_count := backend.io.triDmaMalformedLineCount
  io.tri_dma_status_error_count := backend.io.triDmaStatusErrorCount
  io.tri_dma_tag_mismatch_count := backend.io.triDmaTagMismatchCount
  io.tri_axi_issued_count := backend.io.triAxiIssuedCount
  io.tri_axi_completed_count := backend.io.triAxiCompletedCount
  io.tri_axi_outstanding_count := backend.io.triAxiOutstandingCount
  for (worker <- 0 until GlobalConfig.traceNumWorkers) {
    val result = backend.io.result_out(worker)
    traceQ.io.enq(worker).valid := result.valid
    traceQ.io.enq(worker).bits.meta.rayId := result.bits.meta.rayId
    traceQ.io.enq(worker).bits.hit := result.bits.hit
    traceQ.io.enq(worker).bits.hitId := result.bits.hitId
    traceQ.io.enq(worker).bits.hitT := result.bits.hitT
    traceQ.io.enq(worker).bits.rgb8 := 0.U
    result.ready := traceQ.io.enq(worker).ready
  }

  resultQ.io.enq(0) <> earlyQ.io.deq
  resultQ.io.enq(1) <> traceQ.io.deq
  
  io.result_valid := resultQ.io.deq.valid
  io.result := resultQ.io.deq.bits
  val finalResult = batchClosed && (emittedCount + 1.U === acceptedCount)
  io.result_last := resultQ.io.deq.valid && finalResult
  resultQ.io.deq.ready := io.result_ready
  val resultFire = resultQ.io.deq.fire
  val donePulse = WireDefault(false.B)
  when(resultFire) {
    emittedCount := emittedCount + 1.U
    when(finalResult) {
      batchClosed := false.B
      acceptedCount := 0.U
      emittedCount := 0.U
      donePulse := true.B
    }
  }
  io.batch_done := donePulse
  io.protocol_error := protocolError
}

object FpgaTopGen {
  private val bbFileListMarker = "// ----- 8< ----- FILE \"firrtl_black_box_resource_files.f\" ----- 8< -----"
  private def stripFirrtlBbFileList(targetDir: String): Unit = {
    val outPath = Paths.get(targetDir, "FpgaTop.sv")
    if (!Files.exists(outPath)) return
    val src = Source.fromFile(outPath.toFile, "UTF-8")
    val text = try src.mkString finally src.close()
    val idx = text.indexOf(bbFileListMarker)
    val stripped = if (idx >= 0) text.substring(0, idx).trim + "\n" else text
    val marker = s"// FpgaTopGen build UTC: ${Instant.now}; regenerate with: sbt \\\"runMain FpgaTopGen\\\"\n"
    Files.write(outPath, (marker + stripped).getBytes(StandardCharsets.UTF_8))
  }
  def generateFpgaTopVerilog(targetDir: String = "build/fpga", withMemImplMode: Int = 0, withUseFloatIP: Boolean = true, withTriangleDmaRefill: Boolean = false): Unit = {
    GlobalConfig.withMemImplMode(withMemImplMode) {
      GlobalConfig.withUseFloatIP(withUseFloatIP) {
        GlobalConfig.withTriangleDmaRefill(withTriangleDmaRefill) {
          emitVerilog(new FpgaTop, Array("--target-dir", targetDir))
          stripFirrtlBbFileList(targetDir)
        }
      }
    }
  }
  def main(args: Array[String]): Unit = {
    generateFpgaTopVerilog(targetDir = "build/vivado", withMemImplMode = 2)
    generateFpgaTopVerilog(withUseFloatIP = false, withTriangleDmaRefill = true)
  }
}
