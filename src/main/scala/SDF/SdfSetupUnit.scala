package SDF

import chisel3._
import chisel3.util._
import raytrace_utils._
import raytrace_utils.fudian._

class SdfSetupUnit(cfg: FloatConfig, peCfg: SdfPeConfig) extends Module {
  val io = IO(new Bundle {
    val setup_valid = Input(Bool())
    val setup_origin = Input(new Vec3(cfg))
    val setup_grid_min = Input(new Vec3(cfg))
    val setup_grid_max = Input(new Vec3(cfg))

    val origin = Output(new Vec3(cfg))
    val tNear = Output(UInt(cfg.totalWidth.W))
    val tFar = Output(UInt(cfg.totalWidth.W))
    val gridMin = Output(new Vec3(cfg))
    val invVoxel = Output(new Vec3(cfg))
    val setup_finish = Output(Bool())
  })

  def neg(x: UInt): UInt = Cat(!x(cfg.totalWidth - 1), x(cfg.totalWidth - 2, 0))

  val originReg = RegInit(0.U.asTypeOf(new Vec3(cfg)))
  val tNearReg = RegInit(0.U(cfg.totalWidth.W))
  val tFarReg = RegInit("h7F800000".U(cfg.totalWidth.W))
  val gridMinReg = RegInit(0.U.asTypeOf(new Vec3(cfg)))
  val invVoxelReg = RegInit(0.U.asTypeOf(new Vec3(cfg)))
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

  // inv_voxel(axis) = (GlobalRes*LocalRes) / span(axis)
  val divX = Module(new FDIV(cfg))
  val divY = Module(new FDIV(cfg))
  val divZ = Module(new FDIV(cfg))

  val fullResXfp = java.lang.Float.floatToRawIntBits((peCfg.GlobalResX * peCfg.LocalResX).toFloat)
  val fullResYfp = java.lang.Float.floatToRawIntBits((peCfg.GlobalResY * peCfg.LocalResY).toFloat)
  val fullResZfp = java.lang.Float.floatToRawIntBits((peCfg.GlobalResZ * peCfg.LocalResZ).toFloat)

  val divStart = ShiftRegister(io.setup_valid, cfg.faddLatency)

  divX.io.a := BigInt(fullResXfp & 0xffffffffL).U(cfg.totalWidth.W)
  divY.io.a := BigInt(fullResYfp & 0xffffffffL).U(cfg.totalWidth.W)
  divZ.io.a := BigInt(fullResZfp & 0xffffffffL).U(cfg.totalWidth.W)
  divX.io.b := spanXSub.io.res
  divY.io.b := spanYSub.io.res
  divZ.io.b := spanZSub.io.res
  divX.io.in_valid := divStart
  divY.io.in_valid := divStart
  divZ.io.in_valid := divStart

  when(io.setup_valid) {
    originReg := io.setup_origin
    gridMinReg := io.setup_grid_min
    setupFinishReg := false.B
  }

  when(divX.io.out_valid && divY.io.out_valid && divZ.io.out_valid) {
    invVoxelReg.x := divX.io.result
    invVoxelReg.y := divY.io.result
    invVoxelReg.z := divZ.io.result
    setupFinishReg := true.B
  }

  io.origin := originReg
  io.tNear := tNearReg
  io.tFar := tFarReg
  io.gridMin := gridMinReg
  io.invVoxel := invVoxelReg
  io.setup_finish := setupFinishReg
}
