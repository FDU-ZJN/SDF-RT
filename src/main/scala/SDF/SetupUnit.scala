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
    val gridMinRelOrigin = Output(new Vec3(cfg))
    val gridMaxRelOrigin = Output(new Vec3(cfg))
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
  val gridMinRelOriginReg = RegInit(0.U.asTypeOf(new Vec3(cfg)))
  val gridMaxRelOriginReg = RegInit(0.U.asTypeOf(new Vec3(cfg)))
  val invVoxelReg = RegInit(0.U.asTypeOf(new Vec3(cfg)))
  val invSubVoxelReg = RegInit(0.U.asTypeOf(new Vec3(cfg)))
  val setupFinishReg = RegInit(false.B)

  // span = gridMax - gridMin
  val spanXSub = Module(new FADD(cfg))
  val spanYSub = Module(new FADD(cfg))
  val spanZSub = Module(new FADD(cfg))
  spanXSub.io.a := io.setup_grid_max.x
  spanXSub.io.b := neg(io.setup_grid_min.x)
  spanYSub.io.a := io.setup_grid_max.y
  spanYSub.io.b := neg(io.setup_grid_min.y)
  spanZSub.io.a := io.setup_grid_max.z
  spanZSub.io.b := neg(io.setup_grid_min.z)

  // AABB init constants: grid_{min,max} - origin. These are setup constants
  // and do not need to be recomputed per ray in InitStage.
  val minRelXSub = Module(new FADD(cfg))
  val minRelYSub = Module(new FADD(cfg))
  val minRelZSub = Module(new FADD(cfg))
  val maxRelXSub = Module(new FADD(cfg))
  val maxRelYSub = Module(new FADD(cfg))
  val maxRelZSub = Module(new FADD(cfg))
  minRelXSub.io.a := io.setup_grid_min.x
  minRelXSub.io.b := neg(io.setup_origin.x)
  minRelYSub.io.a := io.setup_grid_min.y
  minRelYSub.io.b := neg(io.setup_origin.y)
  minRelZSub.io.a := io.setup_grid_min.z
  minRelZSub.io.b := neg(io.setup_origin.z)
  maxRelXSub.io.a := io.setup_grid_max.x
  maxRelXSub.io.b := neg(io.setup_origin.x)
  maxRelYSub.io.a := io.setup_grid_max.y
  maxRelYSub.io.b := neg(io.setup_origin.y)
  maxRelZSub.io.a := io.setup_grid_max.z
  maxRelZSub.io.b := neg(io.setup_origin.z)

  // inv_span(axis) = 1 / span(axis)
  val divX = Module(new FRQ(cfg))
  val divY = Module(new FRQ(cfg))
  val divZ = Module(new FRQ(cfg))

  val fullResXfp = java.lang.Float.floatToRawIntBits((peCfg.GlobalResX * peCfg.LocalResX).toFloat)
  val fullResYfp = java.lang.Float.floatToRawIntBits((peCfg.GlobalResY * peCfg.LocalResY).toFloat)
  val fullResZfp = java.lang.Float.floatToRawIntBits((peCfg.GlobalResZ * peCfg.LocalResZ).toFloat)

  val fullSubResXfp = java.lang.Float.floatToRawIntBits((peCfg.DDAGlobalRes * peCfg.SubRes).toFloat)
  val fullSubResYfp = java.lang.Float.floatToRawIntBits((peCfg.DDAGlobalRes * peCfg.SubRes).toFloat)
  val fullSubResZfp = java.lang.Float.floatToRawIntBits((peCfg.DDAGlobalRes * peCfg.SubRes).toFloat)

  val divStart = PipeUtils.pipeData(io.setup_valid, cfg.faddLatency)

  divX.io.in := spanXSub.io.res
  divY.io.in := spanYSub.io.res
  divZ.io.in := spanZSub.io.res
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

  val mulDone = PipeUtils.pipeData(divX.io.out_valid && divY.io.out_valid && divZ.io.out_valid, cfg.fmulLatency)

  when(io.setup_valid) {
    originReg := io.setup_origin
    gridMinReg := io.setup_grid_min
    gridMaxReg := io.setup_grid_max
    setupFinishReg := false.B
  }

  when(mulDone) {
    gridMinRelOriginReg.x := minRelXSub.io.res
    gridMinRelOriginReg.y := minRelYSub.io.res
    gridMinRelOriginReg.z := minRelZSub.io.res
    gridMaxRelOriginReg.x := maxRelXSub.io.res
    gridMaxRelOriginReg.y := maxRelYSub.io.res
    gridMaxRelOriginReg.z := maxRelZSub.io.res
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
  io.gridMinRelOrigin := gridMinRelOriginReg
  io.gridMaxRelOrigin := gridMaxRelOriginReg
  io.invVoxel := invVoxelReg
  io.invSubVoxel := invSubVoxelReg
  io.setup_finish := setupFinishReg
}
