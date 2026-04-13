package SDF

import chisel3._
import chisel3.util._
import raytrace_utils._
import raytrace_utils.fudian._

class SetupUnit(cfg: FloatConfig, peCfg: SdfPeConfig) extends Module {
  val io = IO(new Bundle {
    val setup_valid = Input(Bool())
    val setup_origin = Input(new Vec3(cfg))
    val setup_grid_min = Input(new Vec3(cfg))
    val setup_grid_max = Input(new Vec3(cfg))

    val origin = Output(new Vec3(cfg))
    val tNear = Output(UInt(cfg.totalWidth.W))
    val tFar = Output(UInt(cfg.totalWidth.W))
    val gridMin = Output(new Vec3(cfg))
    val gridMax = Output(new Vec3(cfg))
    val invVoxel = Output(new Vec3(cfg))
    val invSubVoxel = Output(new Vec3(cfg))
    val setup_finish = Output(Bool())
  })

  def neg(x: UInt): UInt = Cat(!x(cfg.totalWidth - 1), x(cfg.totalWidth - 2, 0))

  val originReg = RegInit(0.U.asTypeOf(new Vec3(cfg)))
  val tNearReg = RegInit(0.U(cfg.totalWidth.W))
  val tFarReg = RegInit("h7F800000".U(cfg.totalWidth.W))
  val gridMinReg = RegInit(0.U.asTypeOf(new Vec3(cfg)))
  val gridMaxReg = RegInit(0.U.asTypeOf(new Vec3(cfg)))
  val invVoxelReg = RegInit(0.U.asTypeOf(new Vec3(cfg)))
  val invSubVoxelReg = RegInit(0.U.asTypeOf(new Vec3(cfg)))
  val setupFinishReg = RegInit(false.B)

  // span = gridMax - gridMin
  val spanXSub = Module(new FADD(cfg))
  val spanYSub = Module(new FADD(cfg))
  val spanZSub = Module(new FADD(cfg))
  spanXSub.io.a := io.setup_grid_max.x
  spanXSub.io.b := neg(io.setup_grid_min.x)
  spanXSub.io.rm := RNE
  spanYSub.io.a := io.setup_grid_max.y
  spanYSub.io.b := neg(io.setup_grid_min.y)
  spanYSub.io.rm := RNE
  spanZSub.io.a := io.setup_grid_max.z
  spanZSub.io.b := neg(io.setup_grid_min.z)
  spanZSub.io.rm := RNE

  // inv_span(axis) = 1 / span(axis)
  val divX = Module(new FDIV(cfg))
  val divY = Module(new FDIV(cfg))
  val divZ = Module(new FDIV(cfg))

  val fullOne = BigInt(0x3F800000L).U(cfg.totalWidth.W) // 1.0f

  val fullResXfp = java.lang.Float.floatToRawIntBits((peCfg.GlobalResX * peCfg.LocalResX).toFloat)
  val fullResYfp = java.lang.Float.floatToRawIntBits((peCfg.GlobalResY * peCfg.LocalResY).toFloat)
  val fullResZfp = java.lang.Float.floatToRawIntBits((peCfg.GlobalResZ * peCfg.LocalResZ).toFloat)

  val fullSubResXfp = java.lang.Float.floatToRawIntBits((peCfg.DDAGlobalRes * peCfg.SubRes).toFloat)
  val fullSubResYfp = java.lang.Float.floatToRawIntBits((peCfg.DDAGlobalRes * peCfg.SubRes).toFloat)
  val fullSubResZfp = java.lang.Float.floatToRawIntBits((peCfg.DDAGlobalRes * peCfg.SubRes).toFloat)

  val divStart = ShiftRegister(io.setup_valid, cfg.faddLatency)

  divX.io.a := fullOne
  divY.io.a := fullOne
  divZ.io.a := fullOne
  divX.io.b := spanXSub.io.res
  divY.io.b := spanYSub.io.res
  divZ.io.b := spanZSub.io.res
  divX.io.in_valid := divStart
  divY.io.in_valid := divStart
  divZ.io.in_valid := divStart

  // inv_voxel = (GlobalRes*LocalRes) * inv_span
  val mulResX = Module(new FMUL(cfg))
  val mulResY = Module(new FMUL(cfg))
  val mulResZ = Module(new FMUL(cfg))
  mulResX.io.a := BigInt(fullResXfp & 0xffffffffL).U(cfg.totalWidth.W)
  mulResY.io.a := BigInt(fullResYfp & 0xffffffffL).U(cfg.totalWidth.W)
  mulResZ.io.a := BigInt(fullResZfp & 0xffffffffL).U(cfg.totalWidth.W)
  mulResX.io.b := divX.io.result
  mulResY.io.b := divY.io.result
  mulResZ.io.b := divZ.io.result
  mulResX.io.rm := RNE
  mulResY.io.rm := RNE
  mulResZ.io.rm := RNE

  // inv_sub_voxel = (GlobalRes*SubRes) * inv_span
  val mulSubResX = Module(new FMUL(cfg))
  val mulSubResY = Module(new FMUL(cfg))
  val mulSubResZ = Module(new FMUL(cfg))
  mulSubResX.io.a := BigInt(fullSubResXfp & 0xffffffffL).U(cfg.totalWidth.W)
  mulSubResY.io.a := BigInt(fullSubResYfp & 0xffffffffL).U(cfg.totalWidth.W)
  mulSubResZ.io.a := BigInt(fullSubResZfp & 0xffffffffL).U(cfg.totalWidth.W)
  mulSubResX.io.b := divX.io.result
  mulSubResY.io.b := divY.io.result
  mulSubResZ.io.b := divZ.io.result
  mulSubResX.io.rm := RNE
  mulSubResY.io.rm := RNE
  mulSubResZ.io.rm := RNE

  val mulDone = ShiftRegister(divX.io.out_valid && divY.io.out_valid && divZ.io.out_valid, cfg.fmulLatency)

  when(io.setup_valid) {
    originReg := io.setup_origin
    gridMinReg := io.setup_grid_min
    gridMaxReg := io.setup_grid_max
    setupFinishReg := false.B
  }

  when(mulDone) {
    invVoxelReg.x := mulResX.io.result
    invVoxelReg.y := mulResY.io.result
    invVoxelReg.z := mulResZ.io.result
    invSubVoxelReg.x := mulSubResX.io.result
    invSubVoxelReg.y := mulSubResY.io.result
    invSubVoxelReg.z := mulSubResZ.io.result
    setupFinishReg := true.B
  }

  io.origin := originReg
  io.tNear := tNearReg
  io.tFar := tFarReg
  io.gridMin := gridMinReg
  io.gridMax := gridMaxReg
  io.invVoxel := invVoxelReg
  io.invSubVoxel := invSubVoxelReg
  io.setup_finish := setupFinishReg
}
