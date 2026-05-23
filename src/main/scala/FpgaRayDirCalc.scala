package raytrace_utils

import chisel3._
import chisel3.util._
import raytrace_utils.fudian._


class RayDirCalc(
  cfg: FloatConfig = FloatConfig.FP32,
  width: Int = GlobalConfig.frameWidth,
  height: Int = GlobalConfig.frameHeight,
  laneId: Int = 0,
  numLanes: Int = 1
) extends Module {
  val io = IO(new Bundle {
    val clear     = Input(Bool())
    val in_valid  = Input(Bool())
    val in_ready  = Output(Bool())
    val out_valid = Output(Bool())
    val out_ready = Input(Bool())
    val dir_x     = Output(UInt(cfg.totalWidth.W))
    val dir_y     = Output(UInt(cfg.totalWidth.W))
    val dir_z     = Output(UInt(cfg.totalWidth.W))
  })

  // =========================================================================
  // FP32 constants computed at elaboration time
  // =========================================================================
  private def f32(v: Float): BigInt =
    BigInt(java.lang.Float.floatToRawIntBits(v) & 0xFFFFFFFFL)

  require(height % 5 == 0, "RayDirCalc integer normalization requires height to be divisible by 5 for z=-1.8")
  require(math.max(width, height) <= 65535, "RayDirCalc integer normalization requires 16-bit numerator magnitude")

  private val zScaledInt = -(9 * height / 5)
  private val zScaledAbs = math.abs(zScaledInt)
  private val zScaledSquared = BigInt(zScaledAbs) * BigInt(zScaledAbs)
  val zScaledFp = f32(zScaledInt.toFloat).U(cfg.totalWidth.W)

  private def unsignedToFp32(value: UInt): UInt = {
    val width = value.getWidth
    val isZero = value === 0.U
    val msbOH = PriorityEncoderOH(Reverse(value))
    val msbFromTop = OHToUInt(msbOH)
    val msbIdx = (width - 1).U - msbFromTop
    val exp = (127.U(8.W) + msbIdx)(7, 0)
    val aligned = (value << ((width - 1).U - msbIdx))(width - 1, 0)
    val frac = if (width >= 24) aligned(width - 2, width - 24) else Cat(aligned(width - 2, 0), 0.U((24 - width).W))

    Mux(isZero, 0.U(32.W), Cat(0.U(1.W), exp, frac))
  }

  // Pipeline always accepts
  io.in_ready := true.B

  require(laneId >= 0 && laneId < numLanes, s"laneId $laneId must be in [0, $numLanes)")

  private val pixelCoordW = log2Ceil(math.max(width, height) + numLanes + 1)
  private val startX = (laneId % width).U(pixelCoordW.W)
  private val startY = (laneId / width).U(pixelCoordW.W)
  private val widthU = width.U(pixelCoordW.W)
  private val laneStride = numLanes.U(pixelCoordW.W)

  val pixelXReg = RegInit(startX)
  val pixelYReg = RegInit(startY)
  when(io.clear) {
    pixelXReg := startX
    pixelYReg := startY
  }.elsewhen(io.in_valid) {
    val nextX = pixelXReg + laneStride
    when(nextX >= widthU) {
      pixelXReg := nextX - widthU
      pixelYReg := pixelYReg + 1.U
    }.otherwise {
      pixelXReg := nextX
    }
  }

  private val numerW = pixelCoordW + 1
  require(numerW <= 16, "RayDirCalc IMUL path requires numerators to fit into 16 bits")
  private val widthExt = width.U(numerW.W)
  private val heightExt = height.U(numerW.W)
  val x2Int = (pixelXReg << 1).pad(numerW)
  val y2Int = (pixelYReg << 1).pad(numerW)
  val numerXSign = x2Int < widthExt
  val numerYSign = y2Int > heightExt
  val numerXMag = Wire(UInt(numerW.W))
  val numerYMag = Wire(UInt(numerW.W))
  numerXMag := Mux(numerXSign, widthExt - x2Int, x2Int - widthExt)
  numerYMag := Mux(numerYSign, y2Int - heightExt, heightExt - y2Int)

  val numerXSignReg = Reg(Bool())
  val numerYSignReg = Reg(Bool())
  val numerXMagReg = Reg(UInt(numerW.W))
  val numerYMagReg = Reg(UInt(numerW.W))
  numerXSignReg := numerXSign
  numerYSignReg := numerYSign
  numerXMagReg := numerXMag
  numerYMagReg := numerYMag

  // =========================================================================
  // S1: integer numerator — A=(2*x-w), B=(h-2*y)  (latency +1, cumulative 1)
  // =========================================================================
  val numerLatency = 1

  // =========================================================================
  // S2: IMUL — A², B²  (latency +1, cumulative 2)
  // =========================================================================
  val mulA2 = Module(new IMUL(cfg.useFloatIP))
  val numerXMag16 = numerXMagReg.pad(16)
  mulA2.io.a := numerXMag16
  mulA2.io.b := numerXMag16

  val mulB2 = Module(new IMUL(cfg.useFloatIP))
  val numerYMag16 = numerYMagReg.pad(16)
  mulB2.io.a := numerYMag16
  mulB2.io.b := numerYMag16

  val numerFpLatency = numerLatency + 1
  val aMagFp = unsignedToFp32(numerXMagReg)
  val bMagFp = unsignedToFp32(numerYMagReg)
  val aFp = RegNext(Cat(numerXSignReg, aMagFp(cfg.totalWidth - 2, 0)))
  val bFp = RegNext(Cat(numerYSignReg, bMagFp(cfg.totalWidth - 2, 0)))

  // =========================================================================
  // S3: integer sumSq = A² + B² + (z*h)², then register and IntToFP
  // =========================================================================
  val sumSqInt = Wire(UInt(32.W))
  sumSqInt := mulA2.io.p + mulB2.io.p + zScaledSquared.U(32.W)

  val sumSqIntReg = Reg(UInt(32.W))
  sumSqIntReg := sumSqInt

  val sumSqLatency = numerLatency + 3
  val sumSq = RegNext(unsignedToFp32(sumSqIntReg))

  val fsqrt = Module(new FSQRT(cfg))
  fsqrt.io.in := sumSq

  val invLen = fsqrt.io.out

  val invLenLatency = sumSqLatency + cfg.fsqrtLatency
  require(invLenLatency >= numerFpLatency, "RayDirCalc timing error: invLen must arrive after numerators")

  val aDelayed = PipeUtils.pipeData(aFp, invLenLatency - numerFpLatency)
  val bDelayed = PipeUtils.pipeData(bFp, invLenLatency - numerFpLatency)
  val zDelayed = PipeUtils.pipeData(zScaledFp, invLenLatency)

  val mulDirX = Module(new FMUL(cfg))
  mulDirX.io.a := aDelayed
  mulDirX.io.b := invLen

  val mulDirY = Module(new FMUL(cfg))
  mulDirY.io.a := bDelayed
  mulDirY.io.b := invLen

  val mulDirZ = Module(new FMUL(cfg))
  mulDirZ.io.a := zDelayed
  mulDirZ.io.b := invLen

  // =========================================================================
  // Output
  // =========================================================================
  val totalLatency = invLenLatency + cfg.fmulLatency
  require(totalLatency == sumSqLatency + cfg.fsqrtLatency + cfg.fmulLatency,
    "RayDirCalc timing error: total latency mismatch")

  io.dir_x := mulDirX.io.result
  io.dir_y := mulDirY.io.result
  io.dir_z := mulDirZ.io.result

  // out_valid tracks in_valid through the pipeline
  io.out_valid := PipeUtils.pipeData(io.in_valid, totalLatency)
}
