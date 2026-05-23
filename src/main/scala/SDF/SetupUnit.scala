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

  private def neg(x: UInt): UInt = Cat(!x(cfg.totalWidth - 1), x(cfg.totalWidth - 2, 0))

  private def selectVec(v: Vec3, axis: UInt): UInt =
    Mux(axis === 0.U, v.x, Mux(axis === 1.U, v.y, v.z))

  private def writeVec(v: Vec3, axis: UInt, value: UInt): Unit = {
    when(axis === 0.U) {
      v.x := value
    }.elsewhen(axis === 1.U) {
      v.y := value
    }.otherwise {
      v.z := value
    }
  }

  val originReg = Reg(new Vec3(cfg))
  val gridMinReg = Reg(new Vec3(cfg))
  val gridMaxReg = Reg(new Vec3(cfg))
  val invVoxelReg = Reg(new Vec3(cfg))
  val invSubVoxelReg = Reg(new Vec3(cfg))
  val tNearReg = Reg(UInt(cfg.totalWidth.W))
  val tFarReg = Reg(UInt(cfg.totalWidth.W))
  val setupFinishReg = RegInit(false.B)
  val setupArmed = RegInit(true.B)

  val axisReg = RegInit(0.U(2.W))
  val invSpanReg = Reg(UInt(cfg.totalWidth.W))

  val addAReg = Reg(UInt(cfg.totalWidth.W))
  val addBReg = Reg(UInt(cfg.totalWidth.W))
  val divInReg = Reg(UInt(cfg.totalWidth.W))
  val mulAReg = Reg(UInt(cfg.totalWidth.W))
  val mulBReg = Reg(UInt(cfg.totalWidth.W))

  val fullRes = VecInit(Seq(
    java.lang.Float.floatToRawIntBits((peCfg.GlobalResX * peCfg.LocalResX).toFloat).U(cfg.totalWidth.W),
    java.lang.Float.floatToRawIntBits((peCfg.GlobalResY * peCfg.LocalResY).toFloat).U(cfg.totalWidth.W),
    java.lang.Float.floatToRawIntBits((peCfg.GlobalResZ * peCfg.LocalResZ).toFloat).U(cfg.totalWidth.W)
  ))
  val fullSubRes = VecInit(Seq.fill(3) {
    java.lang.Float.floatToRawIntBits((peCfg.DDAGlobalRes * peCfg.SubRes).toFloat).U(cfg.totalWidth.W)
  })

  private def selectFullRes(axis: UInt): UInt =
    Mux(axis === 0.U, fullRes(0), Mux(axis === 1.U, fullRes(1), fullRes(2)))

  private def selectFullSubRes(axis: UInt): UInt =
    Mux(axis === 0.U, fullSubRes(0), Mux(axis === 1.U, fullSubRes(1), fullSubRes(2)))

  val fadd = Module(new FADD(cfg))
  val frq = Module(new FRQ(cfg))
  val fmul = Module(new FMUL(cfg))

  fadd.io.a := addAReg
  fadd.io.b := addBReg
  frq.io.in := divInReg
  fmul.io.a := mulAReg
  fmul.io.b := mulBReg

  val addIssue = WireDefault(false.B)
  val divIssue = WireDefault(false.B)
  val mulIssue = WireDefault(false.B)
  frq.io.in_valid := divIssue

  val addValid = ShiftRegister(addIssue, cfg.faddLatency, false.B, true.B)
  val mulValid = ShiftRegister(mulIssue, cfg.fmulLatency, false.B, true.B)

  val sIdle :: sLoadAdd :: sIssueAdd :: sWaitAdd :: sIssueDiv :: sWaitDiv :: sLoadMulVoxel :: sIssueMulVoxel :: sWaitMulVoxel :: sLoadMulSub :: sIssueMulSub :: sWaitMulSub :: sDone :: Nil = Enum(13)
  val state = RegInit(sIdle)

  private def beginSetup(): Unit = {
    originReg := io.setup_origin
    gridMinReg := io.setup_grid_min
    gridMaxReg := io.setup_grid_max
    invVoxelReg := 0.U.asTypeOf(new Vec3(cfg))
    invSubVoxelReg := 0.U.asTypeOf(new Vec3(cfg))
    setupFinishReg := false.B
    setupArmed := false.B
    axisReg := 0.U
    state := sLoadAdd
  }

  when(!io.setup_valid) {
    setupArmed := true.B
  }

  val startSetup = io.setup_valid && setupArmed && (state === sIdle || state === sDone)

  switch(state) {
    is(sIdle) {
      when(startSetup) {
        beginSetup()
      }
    }

    is(sLoadAdd) {
      addAReg := selectVec(gridMaxReg, axisReg)
      addBReg := neg(selectVec(gridMinReg, axisReg))
      state := sIssueAdd
    }

    is(sIssueAdd) {
      addIssue := true.B
      state := sWaitAdd
    }

    is(sWaitAdd) {
      when(addValid) {
        divInReg := fadd.io.res
        state := sIssueDiv
      }
    }

    is(sIssueDiv) {
      divIssue := true.B
      state := sWaitDiv
    }

    is(sWaitDiv) {
      when(frq.io.out_valid) {
        invSpanReg := frq.io.result
        state := sLoadMulVoxel
      }
    }

    is(sLoadMulVoxel) {
      mulAReg := selectFullRes(axisReg)
      mulBReg := invSpanReg
      state := sIssueMulVoxel
    }

    is(sIssueMulVoxel) {
      mulIssue := true.B
      state := sWaitMulVoxel
    }

    is(sWaitMulVoxel) {
      when(mulValid) {
        writeVec(invVoxelReg, axisReg, fmul.io.result)
        state := sLoadMulSub
      }
    }

    is(sLoadMulSub) {
      mulAReg := selectFullSubRes(axisReg)
      mulBReg := invSpanReg
      state := sIssueMulSub
    }

    is(sIssueMulSub) {
      mulIssue := true.B
      state := sWaitMulSub
    }

    is(sWaitMulSub) {
      when(mulValid) {
        writeVec(invSubVoxelReg, axisReg, fmul.io.result)
        when(axisReg === 2.U) {
          setupFinishReg := true.B
          state := sDone
        }.otherwise {
          axisReg := axisReg + 1.U
          state := sLoadAdd
        }
      }
    }

    is(sDone) {
      when(startSetup) {
        beginSetup()
      }
    }
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
