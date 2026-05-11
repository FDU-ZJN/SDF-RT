package raytrace_utils
import chisel3._
import chisel3.util._
import raytrace_utils.fudian._

// Global pipe utility functions with explicit initialization values
object PipeUtils {
  def pipeData[T <: Data](x: T, n: Int): T = {
    require(n >= 0, s"pipeData delay must be >= 0, got $n")
    if (n > 0) ShiftRegister(x, n)
    else x
  }

  def pipeUInt(x: UInt, n: Int, init: UInt = 0.U): UInt = {
    pipeData(x, n)
  }
  def pipeBool(x: Bool, n: Int, init: Bool = false.B): Bool = {
    pipeData(x, n)
  }
  def pipeSInt(x: SInt, n: Int, init: SInt = 0.S): SInt = {
    pipeData(x, n)
  }
}

class DotProductUnit(cfg: FloatConfig = FloatConfig.FP32) extends Module {
  val io = IO(new Bundle {
    val a = Input(new Vec3(cfg))
    val b = Input(new Vec3(cfg))
    val res = Output(UInt(cfg.totalWidth.W))
  })

  // 1. 并行执行三个乘法 (Stage 1-3: 3 拍)
  val mul_x = Module(new FMUL(cfg))
  val mul_y = Module(new FMUL(cfg))
  val mul_z = Module(new FMUL(cfg))

  mul_x.io.a := io.a.x; mul_x.io.b := io.b.x
  mul_y.io.a := io.a.y; mul_y.io.b := io.b.y
  mul_z.io.a := io.a.z; mul_z.io.b := io.b.z

  // 2. 第一层加法 (Stage 4-5: 2 拍)
  val add_xy = Module(new FADD(cfg))
  add_xy.io.a := mul_x.io.result
  add_xy.io.b := mul_y.io.result

  // 3. 路径对齐：Z 的乘法结果需要多等 2 拍，直到 xy 加法完成
  val mul_z_delayed = PipeUtils.pipeData(mul_z.io.result, cfg.faddLatency)

  // 4. 第二层加法 (Stage 6-7: 2 拍)
  val add_final = Module(new FADD(cfg))
  add_final.io.a := add_xy.io.res
  add_final.io.b := mul_z_delayed
  io.res := add_final.io.res
}
class CrossProductUnit(cfg: FloatConfig = FloatConfig.FP32) extends Module {
  val io = IO(new Bundle {
    val a = Input(new Vec3(cfg))
    val b = Input(new Vec3(cfg))

    val res = Output(new Vec3(cfg))
  })

  def subtractMul(a: UInt, b: UInt, c: UInt, d: UInt): UInt = {
    val mul1 = Module(new FMUL(cfg))
    val mul2 = Module(new FMUL(cfg))
    val sub  = Module(new FADD(cfg))

    mul1.io.a := a; mul1.io.b := b
    mul2.io.a := c; mul2.io.b := d

    val mul2_neg = Cat(!mul2.io.result(cfg.totalWidth - 1), mul2.io.result(cfg.totalWidth - 2, 0))

    sub.io.a := mul1.io.result
    sub.io.b := mul2_neg
    sub.io.res
  }

  // 计算三轴结果
  val rx = subtractMul(io.a.y, io.b.z, io.a.z, io.b.y) // Cy*Bz - Cz*By
  val ry = subtractMul(io.a.z, io.b.x, io.a.x, io.b.z) // Cz*Bx - Cx*Bz
  val rz = subtractMul(io.a.x, io.b.y, io.a.y, io.b.x) // Cx*By - Cy*Bx

  io.res.x := rx
  io.res.y := ry
  io.res.z := rz
}
