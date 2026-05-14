package SDF

import chisel3._
import chisel3.util._
import raytrace_utils._

class InitStage(cfg: FloatConfig, addrWidth: Int) extends Module {
  val io = IO(new Bundle {
    val setup_origin = Input(new Vec3(cfg))
    val setup_grid_min = Input(new Vec3(cfg))
    val setup_grid_max = Input(new Vec3(cfg))

    val in = Flipped(Decoupled(new SdfInitReq(cfg, addrWidth)))
    val out = Decoupled(new InitStageResp(cfg, addrWidth))
  })

  val aabb = Module(new RayAABBIntersection(cfg))

  val aabbLatency = 4 + cfg.fdivLatency + cfg.fmulLatency + (4 * cfg.fcmpLatency)
  val entryLatency = 1
  val entryAdvanceShift = 14
  val fracWidth = cfg.precision - 1
  require(cfg.totalWidth == 32 && cfg.expWidth == 8 && cfg.precision == 24,
    "InitStage bit-level entry advance currently assumes FP32")
  require(fracWidth >= entryAdvanceShift,
    s"entryAdvanceShift=$entryAdvanceShift exceeds fraction width $fracWidth")

  def entryAdvanceApprox(x: UInt): UInt = {
    val sign = x(cfg.totalWidth - 1)
    val expHi = cfg.totalWidth - 2
    val expLo = fracWidth
    val exp = x(expHi, expLo)
    val frac = x(fracWidth - 1, 0)
    val expMax = Fill(cfg.expWidth, 1.U(1.W))
    val step = (BigInt(1) << (fracWidth - entryAdvanceShift)).U(fracWidth.W)
    val fracSum = Cat(0.U(1.W), frac) + step
    val expInc = fracSum(fracWidth)
    val nextExpWide = Cat(0.U(1.W), exp) + expInc
    val nextExp = nextExpWide(cfg.expWidth - 1, 0)
    val advanced = Cat(sign, nextExp, fracSum(fracWidth - 1, 0))
    val isPositiveZero = (!sign) && (exp === 0.U) && (frac === 0.U)
    val canAdvance = (!sign) && (exp =/= expMax) && (!nextExpWide(cfg.expWidth)) && (nextExp =/= expMax)
    val minAdvance = java.lang.Float.floatToRawIntBits(1.0f / 16384.0f).U(cfg.totalWidth.W)
    Mux(isPositiveZero, minAdvance, Mux(canAdvance, advanced, x))
  }

  val rayWire = Wire(new Ray(cfg))
  rayWire.origin := io.setup_origin
  rayWire.dir := io.in.bits.rd
  rayWire.dist := 0.U

  aabb.io.ray := rayWire
  aabb.io.aabb.min := io.setup_grid_min
  aabb.io.aabb.max := io.setup_grid_max

  io.in.ready := true.B
  val inFire = io.in.fire
  aabb.io.in_valid := inFire

  val rdXAtAabb = PipeUtils.pipeData(io.in.bits.rd.x, aabbLatency)
  val rdYAtAabb = PipeUtils.pipeData(io.in.bits.rd.y, aabbLatency)
  val rdZAtAabb = PipeUtils.pipeData(io.in.bits.rd.z, aabbLatency)

  val roXAtAabb = PipeUtils.pipeData(io.setup_origin.x, aabbLatency)
  val roYAtAabb = PipeUtils.pipeData(io.setup_origin.y, aabbLatency)
  val roZAtAabb = PipeUtils.pipeData(io.setup_origin.z, aabbLatency)

  val slotAtAabb = PipeUtils.pipeData(io.in.bits.meta.slotId, aabbLatency)

  val tEntry = RegNext(entryAdvanceApprox(aabb.io.tNear))

  val outAlignLatency = entryLatency
  val outValid = PipeUtils.pipeBool(aabb.io.out_valid, outAlignLatency, false.B)
  val outHit = PipeUtils.pipeData(aabb.io.hit, outAlignLatency)
  val outSlot = PipeUtils.pipeData(slotAtAabb, entryLatency)
  val outRoX = PipeUtils.pipeData(roXAtAabb, entryLatency)
  val outRoY = PipeUtils.pipeData(roYAtAabb, entryLatency)
  val outRoZ = PipeUtils.pipeData(roZAtAabb, entryLatency)
  val outRdX = PipeUtils.pipeData(rdXAtAabb, entryLatency)
  val outRdY = PipeUtils.pipeData(rdYAtAabb, entryLatency)
  val outRdZ = PipeUtils.pipeData(rdZAtAabb, entryLatency)

  io.out.valid := outValid
  io.out.bits.hit := outHit
  io.out.bits.ray.origin.x := outRoX
  io.out.bits.ray.origin.y := outRoY
  io.out.bits.ray.origin.z := outRoZ
  io.out.bits.ray.dir.x := outRdX
  io.out.bits.ray.dir.y := outRdY
  io.out.bits.ray.dir.z := outRdZ
  io.out.bits.ray.dist := tEntry
  io.out.bits.meta.slotId := outSlot
  io.out.bits.meta.pixelX := 0.U
  io.out.bits.meta.pixelY := 0.U

  when(io.out.valid) {
    assert(io.out.ready, "InitStage expects out.ready to stay high")
  }
}
