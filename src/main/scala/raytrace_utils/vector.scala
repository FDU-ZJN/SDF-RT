package raytrace_utils
import chisel3._
import chisel3.util._
import raytrace_utils.fudian._
class DotProductUnit(cfg: FloatConfig = FloatConfig.FP32) extends Module {
  val io = IO(new Bundle {
    val a = Input(new Vec3(cfg))
    val b = Input(new Vec3(cfg))
    val rm = Input(UInt(3.W))
    val res = Output(UInt(cfg.totalWidth.W))
    val fflags = Output(UInt(5.W))
  })

  // 1. 并行执行三个乘法 (Stage 1-3: 3 拍)
  val mul_x = Module(new FMUL(cfg))
  val mul_y = Module(new FMUL(cfg))
  val mul_z = Module(new FMUL(cfg))

  mul_x.io.a := io.a.x; mul_x.io.b := io.b.x;mul_x.io.rm:=io.rm
  mul_y.io.a := io.a.y; mul_y.io.b := io.b.y;mul_y.io.rm:=io.rm
  mul_z.io.a := io.a.z; mul_z.io.b := io.b.z;mul_z.io.rm:=io.rm

  // 2. 第一层加法 (Stage 4-5: 2 拍)
  val add_xy = Module(new FADD(cfg))
  add_xy.io.a := mul_x.io.result
  add_xy.io.b := mul_y.io.result
  add_xy.io.rm := io.rm

  // 3. 路径对齐：Z 的乘法结果需要多等 2 拍，直到 xy 加法完成
  val mul_z_delayed = ShiftRegister(mul_z.io.result, cfg.faddLatency)
  val flags_z_delayed = ShiftRegister(mul_z.io.fflags, cfg.faddLatency)

  // 4. 第二层加法 (Stage 6-7: 2 拍)
  val add_final = Module(new FADD(cfg))
  add_final.io.a := add_xy.io.res
  add_final.io.b := mul_z_delayed
  add_final.io.rm := io.rm
  io.res := add_final.io.res
  // 别忘了合并所有的异常标志位
  io.fflags := add_final.io.fflags | add_xy.io.fflags | flags_z_delayed
}
class CrossProductUnit(cfg: FloatConfig = FloatConfig.FP32) extends Module {
  val io = IO(new Bundle {
    val a = Input(new Vec3(cfg))
    val b = Input(new Vec3(cfg))
    val rm = Input(UInt(3.W))

    val res = Output(new Vec3(cfg))
    val fflags = Output(UInt(5.W))
  })

  def subtractMul(a: UInt, b: UInt, c: UInt, d: UInt): (UInt, UInt) = {
    val mul1 = Module(new FMUL(cfg))
    val mul2 = Module(new FMUL(cfg))
    val sub  = Module(new FADD(cfg))

    mul1.io.a := a; mul1.io.b := b; mul1.io.rm := io.rm
    mul2.io.a := c; mul2.io.b := d; mul2.io.rm := io.rm

    val mul2_neg = Cat(!mul2.io.result(cfg.totalWidth - 1), mul2.io.result(cfg.totalWidth - 2, 0))

    sub.io.a := mul1.io.result
    sub.io.b := mul2_neg
    sub.io.rm := io.rm

    (sub.io.res, sub.io.fflags | mul1.io.fflags | mul2.io.fflags)
  }

  // 计算三轴结果
  val (rx, fx) = subtractMul(io.a.y, io.b.z, io.a.z, io.b.y) // Cy*Bz - Cz*By
  val (ry, fy) = subtractMul(io.a.z, io.b.x, io.a.x, io.b.z) // Cz*Bx - Cx*Bz
  val (rz, fz) = subtractMul(io.a.x, io.b.y, io.a.y, io.b.x) // Cx*By - Cy*Bx

  io.res.x := rx
  io.res.y := ry
  io.res.z := rz

  // 合并所有轴产生的异常标志
  io.fflags := fx | fy | fz
}

