package raytrace_utils

import chisel3._
import chisel3.util._
import raytrace_utils.fudian._


class RayDirCalc(
  cfg: FloatConfig = FloatConfig.FP32,
  width: Int = GlobalConfig.frameWidth,
  height: Int = GlobalConfig.frameHeight
) extends Module {
  val io = IO(new Bundle {
    val in_valid  = Input(Bool())
    val in_ready  = Output(Bool())
    val pixel_x   = Input(UInt(16.W))
    val pixel_y   = Input(UInt(16.W))
    val out_valid = Output(Bool())
    val out_ready = Input(Bool())
    val dir_x     = Output(UInt(cfg.totalWidth.W))
    val dir_y     = Output(UInt(cfg.totalWidth.W))
    val dir_z     = Output(UInt(cfg.totalWidth.W))
    val out_pixel_x = Output(UInt(16.W))
    val out_pixel_y = Output(UInt(16.W))
  })

  val rm = RNE

  // =========================================================================
  // FP32 constants computed at elaboration time
  // =========================================================================
  private def f32(v: Float): BigInt =
    BigInt(java.lang.Float.floatToRawIntBits(v) & 0xFFFFFFFFL)

  val twoFp    = f32(2.0f).U(cfg.totalWidth.W)
  val invHFp   = f32(1.0f / height.toFloat).U(cfg.totalWidth.W)
  val negInvHFp = f32(-1.0f / height.toFloat).U(cfg.totalWidth.W)
  val negWFp   = f32((-width).toFloat).U(cfg.totalWidth.W)
  val negHFp   = f32((-height).toFloat).U(cfg.totalWidth.W)
  val neg1_8Fp = f32(-1.8f).U(cfg.totalWidth.W)
  val z2Fp     = f32(3.24f).U(cfg.totalWidth.W)   // (-1.8)² = 3.24, pre-computed

  // Pipeline always accepts
  io.in_ready := true.B

  val intToFP_x = Module(new IntToFP(cfg.expWidth, cfg.precision))
  intToFP_x.io.int  := Cat(0.U(48.W), io.pixel_x)
  intToFP_x.io.sign := false.B
  intToFP_x.io.long := false.B
  intToFP_x.io.rm   := rm

  val intToFP_y = Module(new IntToFP(cfg.expWidth, cfg.precision))
  intToFP_y.io.int  := Cat(0.U(48.W), io.pixel_y)
  intToFP_y.io.sign := false.B
  intToFP_y.io.long := false.B
  intToFP_y.io.rm   := rm

  val xFp = intToFP_x.io.result
  val yFp = intToFP_y.io.result

  // =========================================================================
  // S1: FMUL — 2*x, 2*y  (latency +8, cumulative 8)
  // =========================================================================
  val mul2x = Module(new FMUL(cfg))
  mul2x.io.a := xFp
  mul2x.io.b := twoFp
  mul2x.io.rm := rm

  val mul2y = Module(new FMUL(cfg))
  mul2y.io.a := yFp
  mul2y.io.b := twoFp
  mul2y.io.rm := rm

  // =========================================================================
  // S2: FADD — 2x - w, 2y - h  (latency +7, cumulative 15)
  // =========================================================================
  val subW = Module(new FADD(cfg))
  subW.io.a := mul2x.io.result
  subW.io.b := negWFp
  subW.io.rm := rm

  val subH = Module(new FADD(cfg))
  subH.io.a := mul2y.io.result
  subH.io.b := negHFp
  subH.io.rm := rm

  // =========================================================================
  // S3: FMUL — u = (2x-w)/h, v = -(2y-h)/h  (latency +8, cumulative 23)
  //       Division replaced by multiplication with pre-computed 1/h.
  // =========================================================================
  val mulU = Module(new FMUL(cfg))
  mulU.io.a := subW.io.res
  mulU.io.b := invHFp
  mulU.io.rm := rm

  val mulV = Module(new FMUL(cfg))
  mulV.io.a := subH.io.res
  mulV.io.b := negInvHFp
  mulV.io.rm := rm

  val u = mulU.io.result  // available at cumulative 23
  val v = mulV.io.result

  // =========================================================================
  // S4: FMUL — u², v²  (latency +8, cumulative 31)
  //       z² = 3.24 is pre-computed constant (available from t=0)
  // =========================================================================
  val mulU2 = Module(new FMUL(cfg))
  mulU2.io.a := u
  mulU2.io.b := u
  mulU2.io.rm := rm

  val mulV2 = Module(new FMUL(cfg))
  mulV2.io.a := v
  mulV2.io.b := v
  mulV2.io.rm := rm

  // =========================================================================
  // S5: FADD — u² + v²  (latency +7, cumulative 38)
  // =========================================================================
  val addUV2 = Module(new FADD(cfg))
  addUV2.io.a := mulU2.io.result
  addUV2.io.b := mulV2.io.result
  addUV2.io.rm := rm

  // =========================================================================
  // S6: FADD — sumSq = (u²+v²) + z²  (latency +7, cumulative 45)
  //       z² is constant, always valid; no alignment delay needed
  // =========================================================================
  val addSum = Module(new FADD(cfg))
  addSum.io.a := addUV2.io.res
  addSum.io.b := z2Fp
  addSum.io.rm := rm

  val sumSq = addSum.io.res  // available at cumulative 45

  // =========================================================================
  // S7: FSQRT — invLen = 1 / sqrt(sumSq)  (latency +cfg.fsqrtLatency)
  // =========================================================================
  val fsqrt = Module(new FSQRT(cfg))
  fsqrt.io.in := sumSq

  val invLen = fsqrt.io.out  // available at cumulative 74

  // Latency bookkeeping derived from the configured floating-point units.
  //
  // u/v become available after S3:
  //   S1 FMUL + S2 FADD + S3 FMUL
  val uvLatency = cfg.fmulLatency + cfg.faddLatency + cfg.fmulLatency
  // invLen becomes available after S7:
  //   S1 FMUL + S2 FADD + S3 FMUL + S4 FMUL + S5 FADD + S6 FADD + S7 FSQRT
  val invLenLatency = 3 * cfg.fmulLatency + 3 * cfg.faddLatency + cfg.fsqrtLatency
  require(invLenLatency >= uvLatency, "RayDirCalc timing error: invLen must arrive after u/v")

  // =========================================================================
  // S8: FMUL — dir = (u, v, z) * invLen  (latency +8, cumulative 82)
  //       u, v produced at t=23 must be delayed to t=74 (51 cycles)
  //       z is constant, delayed 74 cycles for alignment
  // =========================================================================
  val uDelayed = PipeUtils.pipeData(u, invLenLatency - uvLatency)
  val vDelayed = PipeUtils.pipeData(v, invLenLatency - uvLatency)
  val zDelayed = PipeUtils.pipeData(neg1_8Fp, invLenLatency)

  val mulDirX = Module(new FMUL(cfg))
  mulDirX.io.a := uDelayed
  mulDirX.io.b := invLen
  mulDirX.io.rm := rm

  val mulDirY = Module(new FMUL(cfg))
  mulDirY.io.a := vDelayed
  mulDirY.io.b := invLen
  mulDirY.io.rm := rm

  val mulDirZ = Module(new FMUL(cfg))
  mulDirZ.io.a := zDelayed
  mulDirZ.io.b := invLen
  mulDirZ.io.rm := rm

  // =========================================================================
  // Output
  // =========================================================================
  val totalLatency = invLenLatency + cfg.fmulLatency
  require(totalLatency == 4 * cfg.fmulLatency + 3 * cfg.faddLatency + cfg.fsqrtLatency,
    "RayDirCalc timing error: total latency mismatch")

  io.dir_x := mulDirX.io.result
  io.dir_y := mulDirY.io.result
  io.dir_z := mulDirZ.io.result

  // out_valid tracks in_valid through the pipeline
  io.out_valid := PipeUtils.pipeData(io.in_valid, totalLatency)

  // Pass through pixel coordinates with matching delay
  io.out_pixel_x := PipeUtils.pipeData(io.pixel_x, totalLatency)
  io.out_pixel_y := PipeUtils.pipeData(io.pixel_y, totalLatency)
}
