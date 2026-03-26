package DDA

import Trace._
import chisel3._
import chisel3.util._
import raytrace_utils._

class DDA(
  cfg: FloatConfig = FloatConfig.FP32,
  addrWidth: Int = 32,
  globalRes: Int = 16,
  subRes: Int = 8,
  maxTraversalSteps: Int = 1024
) extends Module {
  require((globalRes & (globalRes - 1)) == 0, s"globalRes must be power-of-two, got $globalRes")
  require((subRes & (subRes - 1)) == 0, s"subRes must be power-of-two, got $subRes")

  val io = IO(new Bundle {
    val in = Flipped(Decoupled(new DdaTraversalReq(cfg, addrWidth)))
    val out = Decoupled(new DdaTraversalResult(cfg, addrWidth))
  })

  private val localShift = log2Ceil(subRes)
  private val globalShift = log2Ceil(globalRes)
  private val globalPlaneShift = globalShift + globalShift
  private val localPlaneShift = localShift + localShift
  private val localMask = (subRes - 1).S((addrWidth + 1).W)
  private val totalSub = globalRes * subRes
  private val totalSubS = totalSub.S((addrWidth + 1).W)
  private val missT = "h7F7FFFFF".U(cfg.totalWidth.W)

  def fpAbs(x: UInt): UInt = Cat(0.U(1.W), x(cfg.totalWidth - 2, 0))

  val trace = Module(new TraceStage())
  val subgridMem = Module(new SubgridMetaMemDPI(addrWidth))

  val sIdle :: sFetchMeta :: sWaitMeta :: sIssueTrace :: sWaitTrace :: sStep :: sDone :: Nil = Enum(7)
  val state = RegInit(sIdle)

  val rayReg = Reg(new Ray(cfg))
  val metaReg = Reg(new RayMeta(addrWidth))
  val resultReg = Reg(new DdaTraversalResult(cfg, addrWidth))

  val subX = RegInit(0.S((addrWidth + 1).W))
  val subY = RegInit(0.S((addrWidth + 1).W))
  val subZ = RegInit(0.S((addrWidth + 1).W))
  val iter = RegInit(0.U(16.W))

  val triStartReg = Reg(UInt(addrWidth.W))
  val triCountReg = Reg(UInt(16.W))

  val inBounds = subX >= 0.S && subY >= 0.S && subZ >= 0.S &&
    subX < totalSubS && subY < totalSubS && subZ < totalSubS

  val globalX = (subX.asUInt >> localShift).pad(addrWidth)(addrWidth - 1, 0)
  val globalY = (subY.asUInt >> localShift).pad(addrWidth)(addrWidth - 1, 0)
  val globalZ = (subZ.asUInt >> localShift).pad(addrWidth)(addrWidth - 1, 0)
  val localX = (subX & localMask).asUInt.pad(addrWidth)(addrWidth - 1, 0)
  val localY = (subY & localMask).asUInt.pad(addrWidth)(addrWidth - 1, 0)
  val localZ = (subZ & localMask).asUInt.pad(addrWidth)(addrWidth - 1, 0)

  val globalYScaled = (globalY << globalShift).asUInt
  val globalZScaled = (globalZ << globalPlaneShift).asUInt
  val localYScaled = (localY << localShift).asUInt
  val localZScaled = (localZ << localPlaneShift).asUInt

  val globalLinear = globalX + globalYScaled + globalZScaled
  val localLinear = localX + localYScaled + localZScaled

  val rdNegX = rayReg.dir.x(cfg.totalWidth - 1)
  val rdNegY = rayReg.dir.y(cfg.totalWidth - 1)
  val rdNegZ = rayReg.dir.z(cfg.totalWidth - 1)

  val rdAbsX = fpAbs(rayReg.dir.x)
  val rdAbsY = fpAbs(rayReg.dir.y)
  val rdAbsZ = fpAbs(rayReg.dir.z)

  val xDominant = (rdAbsX >= rdAbsY) && (rdAbsX >= rdAbsZ)
  val yDominant = !xDominant && (rdAbsY >= rdAbsZ)
  val zDominant = !xDominant && !yDominant

  io.in.ready := state === sIdle

  subgridMem.io.clk := clock
  subgridMem.io.reset := reset
  subgridMem.io.globalIdx := globalLinear
  subgridMem.io.localIdx := localLinear
  subgridMem.io.en := state === sFetchMeta

  trace.io.issue_in.valid := false.B
  trace.io.issue_in.bits.ray := rayReg
  trace.io.issue_in.bits.meta := metaReg
  trace.io.tri_batch_in.valid := false.B
  trace.io.tri_batch_in.bits.base_addr := triStartReg
  trace.io.tri_batch_in.bits.count := triCountReg
  trace.io.end_exec := false.B
  trace.io.result_out.ready := true.B

  io.out.valid := state === sDone
  io.out.bits := resultReg

  switch(state) {
    is(sIdle) {
      when(io.in.fire) {
        rayReg := io.in.bits.ray
        metaReg := io.in.bits.meta
        // SDF stage passes compressed subgrid entry point in origin payload.
        subX := io.in.bits.ray.origin.x(addrWidth - 1, 0).asSInt
        subY := io.in.bits.ray.origin.y(addrWidth - 1, 0).asSInt
        subZ := io.in.bits.ray.origin.z(addrWidth - 1, 0).asSInt
        iter := 0.U
        state := sFetchMeta
      }
    }

    is(sFetchMeta) {
      when(!inBounds || iter >= maxTraversalSteps.U) {
        resultReg.meta := metaReg
        resultReg.hit := false.B
        resultReg.hitId := 0.U
        resultReg.hitT := missT
        state := sDone
      }.otherwise {
        state := sWaitMeta
      }
    }

    is(sWaitMeta) {
      when(subgridMem.io.valid) {
        triStartReg := subgridMem.io.triStart
        triCountReg := subgridMem.io.triCount
        when(subgridMem.io.triCount === 0.U) {
          state := sStep
        }.otherwise {
          state := sIssueTrace
        }
      }
    }

    is(sIssueTrace) {
      trace.io.issue_in.valid := true.B
      trace.io.tri_batch_in.valid := true.B
      trace.io.end_exec := trace.io.issue_in.ready && trace.io.tri_batch_in.ready
      when(trace.io.issue_in.ready && trace.io.tri_batch_in.ready) {
        state := sWaitTrace
      }
    }

    is(sWaitTrace) {
      when(trace.io.result_out.valid) {
        when(trace.io.result_out.bits.hit) {
          resultReg.meta := trace.io.result_out.bits.meta
          resultReg.hit := true.B
          resultReg.hitId := trace.io.result_out.bits.hitId
          resultReg.hitT := trace.io.result_out.bits.hitT
          state := sDone
        }.otherwise {
          state := sStep
        }
      }
    }

    is(sStep) {
      when(xDominant) {
        subX := subX + Mux(rdNegX, (-1).S, 1.S)
      }.elsewhen(yDominant) {
        subY := subY + Mux(rdNegY, (-1).S, 1.S)
      }.elsewhen(zDominant) {
        subZ := subZ + Mux(rdNegZ, (-1).S, 1.S)
      }
      iter := iter + 1.U
      state := sFetchMeta
    }

    is(sDone) {
      when(io.out.ready) {
        state := sIdle
      }
    }
  }
}
