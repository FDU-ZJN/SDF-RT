package raytrace_utils

import chisel3._
import chisel3.util._

class FDIV(cfg: FloatConfig = FloatConfig.FP32) extends Module {
  val expWidth   = cfg.expWidth    // e.g., 8
  val sigWidth   = cfg.precision   // 包含隐式位的总精度, e.g., 24
  val fracWidth  = sigWidth - 1    // 显式尾数位宽, e.g., 23
  val totalWidth = cfg.totalWidth  // e.g., 32

  // 计算 Bias: 2^(expWidth-1) - 1
  val bias = (1 << (expWidth - 1)) - 1

  val io = IO(new Bundle {
    val a, b      = Input(UInt(totalWidth.W))
    val result    = Output(UInt(totalWidth.W))
    val out_valid = Output(Bool())
    val in_valid  = Input(Bool())
  })

  // --- 1. 浮点结构解析 ---
  val sign_a = io.a(totalWidth - 1)
  val exp_a  = io.a(totalWidth - 2, fracWidth)
  val frac_a = io.a(fracWidth - 1, 0)

  val sign_b = io.b(totalWidth - 1)
  val exp_b  = io.b(totalWidth - 2, fracWidth)
  val frac_b = io.b(fracWidth - 1, 0)

  // 常量定义
  val expMax = ((1 << expWidth) - 1).U(expWidth.W)

  // --- 2. 特殊值标志 ---
  val a_zero = (exp_a === 0.U) && (frac_a === 0.U)
  val b_zero = (exp_b === 0.U) && (frac_b === 0.U)
  val a_inf  = (exp_a === expMax) && (frac_a === 0.U)
  val b_inf  = (exp_b === expMax) && (frac_b === 0.U)
  val a_nan  = (exp_a === expMax) && (frac_a =/= 0.U)
  val b_nan  = (exp_b === expMax) && (frac_b =/= 0.U)

  // 尾数比较逻辑（用于原代码中的 alb）
  val alb    = frac_a < frac_b

  // --- 3. 符号计算 ---
  val result_sign = sign_a ^ sign_b

  // --- 4. 核心逻辑 ---
  val final_exp   = Wire(UInt(expWidth.W))
  val final_frac  = Wire(UInt(fracWidth.W))
  val invalid     = (a_nan || b_nan) || (a_inf && b_inf) || (a_zero && b_zero)
  val div_by_zero = b_zero && !a_zero && !a_nan

  final_exp  := 0.U
  final_frac := 0.U

  when(invalid || div_by_zero) {
    final_exp  := expMax
    // 参数化处理 NaN 的尾数保留逻辑 (假设最高位设为1)
    final_frac := Mux(a_nan, Cat(1.B, frac_a(fracWidth - 2, 0)),
      Mux(b_nan, Cat(1.B, frac_b(fracWidth - 2, 0)), 0.U))
  } .elsewhen(a_inf || b_zero) {
    final_exp  := expMax
    final_frac := 0.U
  } .elsewhen(a_zero || b_inf) {
    final_exp  := 0.U
    final_frac := 0.U
  } .otherwise {
    // --- 指数计算 ---
    // 使用 SInt 处理可能的负数中间结果
    val exp_tmp = Wire(SInt((expWidth + 2).W))
    when(alb) {
      exp_tmp := exp_a.zext - exp_b.zext + (bias - 1).S
    } .otherwise {
      exp_tmp := exp_a.zext - exp_b.zext + bias.S
    }

    // --- 尾数处理 ---
    val ma = Cat(1.B, frac_a)
    val mb = Cat(1.B, frac_b)

    // 模拟原逻辑: (ma << sigWidth) / mb
    val frac_calculation = (ma << sigWidth) / mb

    // 规格化与舍入处理 (保持原逻辑：对最低位进行简单的加1舍入)
    val temp_frac = Wire(UInt(fracWidth.W))
    val norm_bit = frac_calculation(sigWidth) // 对应原代码中的 frac_calculation(24)

    when(norm_bit === 1.B) {
      val rounded = frac_calculation(fracWidth, 1) + frac_calculation(0)
      temp_frac := rounded
    } .otherwise {
      temp_frac := frac_calculation(fracWidth - 1, 0)
    }

    // --- 指数边界处理 ---
    when(exp_tmp >= (expMax.zext)) {
      final_exp  := expMax
      final_frac := 0.U
    } .elsewhen(exp_tmp <= 0.S) {
      final_exp  := 0.U
      final_frac := 0.U
    } .otherwise {
      final_exp  := exp_tmp.asUInt
      final_frac := temp_frac
    }
  }

  // --- 5. 结果输出 ---
  val combined_res = Cat(result_sign, final_exp, final_frac)
  io.result    := ShiftRegister(combined_res, cfg.fdivLatency)
  io.out_valid := ShiftRegister(io.in_valid, cfg.fdivLatency)
}