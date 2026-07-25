package SDF

import chisel3._
import chisel3.util._
import raytrace_utils._

class InitStage(cfg: FloatConfig, addrWidth: Int) extends Module {
  val io = IO(new Bundle {
    val setup_origin = Input(new Vec3(cfg))
    val setup_grid_min = Input(new Vec3(cfg))
    val setup_grid_max = Input(new Vec3(cfg))

    val in = Flipped(Decoupled(UInt(GlobalConfig.frontendContextBits.W)))
    val context_id = Output(UInt(GlobalConfig.frontendContextBits.W))
    val immutable = Input(new ImmutableRayRecord(cfg, addrWidth))
    val out = Decoupled(new FrontendInitResp(cfg))
  })

  val aabb = Module(new RayAABBIntersection(cfg))

  val aabbLatency = 4 + cfg.fdivLatency + cfg.fmulLatency + (4 * cfg.fcmpLatency)
  val entryLatency = 1
  private val outputQueueDepth = GlobalConfig.frontendContextCount
  private val reservationCapacity = outputQueueDepth
  private val reservationWidth = log2Ceil(reservationCapacity + 1)
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
  io.context_id := io.in.bits
  rayWire.origin := io.immutable.origin
  rayWire.dir := io.immutable.dir
  rayWire.dist := 0.U

  aabb.io.ray := rayWire
  aabb.io.aabb.min := io.setup_grid_min
  aabb.io.aabb.max := io.setup_grid_max

  val outputQ = Module(new Queue(new FrontendInitResp(cfg), outputQueueDepth))
  val reservations = RegInit(0.U(reservationWidth.W))
  io.in.ready := reservations < reservationCapacity.U
  val inFire = io.in.fire
  aabb.io.in_valid := inFire

  val contextAtAabb = PipeUtils.pipeData(io.in.bits, aabbLatency)

  val tEntry = RegNext(entryAdvanceApprox(aabb.io.tNear))

  val outAlignLatency = entryLatency
  val outValid = PipeUtils.pipeBool(aabb.io.out_valid, outAlignLatency, false.B)
  val outHit = PipeUtils.pipeData(aabb.io.hit, outAlignLatency)
  val outContextId = PipeUtils.pipeData(contextAtAabb, entryLatency)

  outputQ.io.enq.valid := outValid
  outputQ.io.enq.bits.hit := outHit
  outputQ.io.enq.bits.contextId := outContextId
  outputQ.io.enq.bits.entryT := tEntry
  assert(!outValid || outputQ.io.enq.ready, "InitStage reservation overflow")
  io.out <> outputQ.io.deq

  val outFire = io.out.fire
  when(inFire || outFire) {
    reservations := reservations + inFire.asUInt - outFire.asUInt
  }
  assert(reservations <= reservationCapacity.U, "InitStage reservation counter overflow")
}
