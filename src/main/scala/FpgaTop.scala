import chisel3._
import chisel3.util._
import raytrace_utils._
import DDA.DDA
import SDF.{InitStage, SdfMemWriteIO, SdfStage, SetupUnit}
import Trace.{TraceController, TriMemDmaReadData}

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
    val ray_in = Input(Vec(2, new RayIngress(c.cfg)))
    val ray_in_mask = Input(UInt(2.W))
    val ray_in_last = Input(Bool())
    val ray_in_ready = Output(Bool())

    val result_valid = Output(Bool())
    val result = Output(Vec(2, new RenderResult(c.cfg, c.addrWidth)))
    val result_mask = Output(UInt(2.W))
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
  val initStages = Seq.fill(2)(Module(new InitStage(c.cfg, c.addrWidth)))
  val sdfStage = Module(new SdfStage(c.cfg, c.addrWidth))
  val ddaStage = Module(new DDA(c.cfg, c.addrWidth, sdfCfg.DDAGlobalRes, sdfCfg.SubRes, sdfCfg.DDAMaxSteps))
  val traceController = Module(new TraceController(c))
  val earlyQ = Module(new FourToTwoArbiter(new RenderResult(c.cfg, c.addrWidth), GlobalConfig.resultArbiterFifoDepth))
  val traceQ = Module(new FourToTwoArbiter(new RenderResult(c.cfg, c.addrWidth), GlobalConfig.resultArbiterFifoDepth))
  val resultQ = Module(new FourToTwoArbiter(new RenderResult(c.cfg, c.addrWidth), GlobalConfig.resultArbiterFifoDepth))

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

  for (lane <- 0 until 2) {
    initStages(lane).io.setup_origin := setupUnit.io.origin
    initStages(lane).io.setup_grid_min := setupUnit.io.gridMin
    initStages(lane).io.setup_grid_max := setupUnit.io.gridMax
  }
  sdfStage.io.grid_min := setupUnit.io.gridMin
  sdfStage.io.inv_voxel := setupUnit.io.invVoxel
  sdfStage.io.inv_sub_voxel := setupUnit.io.invSubVoxel
  sdfStage.io.sdf_mem_wr <> io.sdf_mem_wr

  val batchClosed = RegInit(false.B)
  val acceptedCount = RegInit(0.U(32.W))
  val emittedCount = RegInit(0.U(32.W))
  val protocolError = RegInit(false.B)
  val inputMaskCount = PopCount(io.ray_in_mask)
  val inputStageReady = (0 until 2).map(lane => !io.ray_in_mask(lane) || initStages(lane).io.in.ready).reduce(_ && _)
  val inputCanAccept = inputStageReady && !batchClosed && setupUnit.io.setup_finish
  io.ray_in_ready := inputCanAccept
  val inputFire = io.ray_in_valid && io.ray_in_ready
  when(io.ray_in_valid && io.ray_in_mask === 0.U) { protocolError := true.B }
  when(io.ray_in_valid && batchClosed) { protocolError := true.B }
  when(inputFire) {
    acceptedCount := acceptedCount + inputMaskCount
    when(io.ray_in_last) { batchClosed := true.B }
  }
  for (lane <- 0 until 2) {
    initStages(lane).io.in.valid := io.ray_in_valid && io.ray_in_mask(lane) && inputCanAccept
    initStages(lane).io.in.bits.ray.origin := io.ray_in(lane).origin
    initStages(lane).io.in.bits.ray.dir := io.ray_in(lane).dir
    initStages(lane).io.in.bits.ray.dist := 0.U
    initStages(lane).io.in.bits.meta.rayId := io.ray_in(lane).rayId
  }

  def missResult(rayId: UInt): RenderResult = {
    val r = Wire(new RenderResult(c.cfg, c.addrWidth))
    r.meta.rayId := rayId
    r.hit := false.B
    r.hitId := 0.U
    r.hitT := missT
    r.rgb8 := 0.U
    r
  }
  for (lane <- 0 until 2) {
    val initOut = initStages(lane).io.out
    earlyQ.io.enq(lane).valid := initOut.valid && !initOut.bits.hit
    earlyQ.io.enq(lane).bits := missResult(initOut.bits.meta.rayId)
    sdfStage.io.issue_in(lane).valid := initOut.valid && initOut.bits.hit
    sdfStage.io.issue_in(lane).bits.ray := initOut.bits.ray
    sdfStage.io.issue_in(lane).bits.meta := initOut.bits.meta
    initOut.ready := Mux(initOut.bits.hit, sdfStage.io.issue_in(lane).ready, earlyQ.io.enq(lane).ready)

    val sdfOut = sdfStage.io.miss_out(lane)
    earlyQ.io.enq(lane + 2).valid := sdfOut.valid
    earlyQ.io.enq(lane + 2).bits := missResult(sdfOut.bits.meta.rayId)
    sdfOut.ready := earlyQ.io.enq(lane + 2).ready
  }

  sdfStage.io.dda_context_id <> ddaStage.io.init_alloc
  ddaStage.io.context_in <> sdfStage.io.dda_context_out

  traceController.io.trace_claim <> ddaStage.io.trace_claim
  ddaStage.io.trace_cmd_context := traceController.io.trace_cmd_context
  ddaStage.io.trace_cmd_active := traceController.io.trace_cmd_active
  for (lane <- 0 until GlobalConfig.traceNumWorkers * 2) {
    traceController.io.trace_cmd(lane) <> ddaStage.io.trace_cmd(lane)
  }
  traceController.io.trace_dda_done := ddaStage.io.trace_dda_done
  ddaStage.io.trace_stop := traceController.io.trace_stop
  ddaStage.io.trace_release := traceController.io.trace_release
  traceController.io.triangleBaseAddress := io.triangle_base_address
  io.tri_dma_cmd <> traceController.io.triDmaCmd
  traceController.io.triDmaData <> io.tri_dma_data
  traceController.io.triDmaStatus <> io.tri_dma_status
  io.tri_dma_read_error := traceController.io.triDmaReadError
  io.tri_dma_malformed_line_count := traceController.io.triDmaMalformedLineCount
  io.tri_dma_status_error_count := traceController.io.triDmaStatusErrorCount
  io.tri_dma_tag_mismatch_count := traceController.io.triDmaTagMismatchCount
  io.tri_axi_issued_count := traceController.io.triAxiIssuedCount
  io.tri_axi_completed_count := traceController.io.triAxiCompletedCount
  io.tri_axi_outstanding_count := traceController.io.triAxiOutstandingCount
  for (worker <- 0 until 4) {
    val result = traceController.io.result_out(worker)
    traceQ.io.enq(worker).valid := result.valid
    traceQ.io.enq(worker).bits.meta.rayId := result.bits.meta.rayId
    traceQ.io.enq(worker).bits.hit := result.bits.hit
    traceQ.io.enq(worker).bits.hitId := result.bits.hitId
    traceQ.io.enq(worker).bits.hitT := result.bits.hitT
    traceQ.io.enq(worker).bits.rgb8 := 0.U
    result.ready := traceQ.io.enq(worker).ready
  }

	val earlyDispatch = earlyQ.io.deq.valid && earlyQ.io.deq.ready
	val traceDispatch = traceQ.io.deq.valid && traceQ.io.deq.ready
	for (lane <- 0 until 2) {
	  resultQ.io.enq(lane).valid := earlyDispatch && earlyQ.io.deq.bits.validMask(lane)
	  resultQ.io.enq(lane).bits := earlyQ.io.deq.bits.bits(lane)
	  resultQ.io.enq(lane + 2).valid := traceDispatch && traceQ.io.deq.bits.validMask(lane)
	  resultQ.io.enq(lane + 2).bits := traceQ.io.deq.bits.bits(lane)
	 }
  earlyQ.io.deq.ready := resultQ.io.enq(0).ready && resultQ.io.enq(1).ready
  traceQ.io.deq.ready := resultQ.io.enq(2).ready && resultQ.io.enq(3).ready
  
  io.result_valid := resultQ.io.deq.valid
  io.result_mask := resultQ.io.deq.bits.validMask
  io.result := resultQ.io.deq.bits.bits
  val resultCount = PopCount(resultQ.io.deq.bits.validMask)
  val finalResult = batchClosed && (emittedCount + resultCount === acceptedCount)
  io.result_last := resultQ.io.deq.valid && finalResult
  resultQ.io.deq.ready := io.result_ready
  val resultFire = resultQ.io.deq.fire
  val donePulse = WireDefault(false.B)
  when(resultFire) {
    emittedCount := emittedCount + resultCount
    when(finalResult) {
      batchClosed := false.B
      acceptedCount := 0.U
      emittedCount := 0.U
      donePulse := true.B
    }
  }
  io.batch_done := donePulse
  io.protocol_error := protocolError || ddaStage.io.watchdog_error
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
