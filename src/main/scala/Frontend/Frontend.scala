package Frontend

import chisel3._
import chisel3.util._
import SDF.{FrontendContextManager, InitStage, SdfMemWriteIO, SdfStage}
import raytrace_utils._

/** Frontend boundary for ray contexts, grid entry initialization, and SDF traversal. */
class Frontend(c: TriPeConfig = TriPeConfig()) extends Module {
  private val missT = "h7F800000".U(c.cfg.totalWidth.W)

  val io = IO(new Bundle {
    val setup_origin = Input(new Vec3(c.cfg))
    val setup_grid_min = Input(new Vec3(c.cfg))
    val setup_grid_max = Input(new Vec3(c.cfg))
    val inv_voxel = Input(new Vec3(c.cfg))
    val inv_sub_voxel = Input(new Vec3(c.cfg))

    val ingress = Flipped(Decoupled(new RayIngress(c.cfg)))
    val miss_out = Decoupled(new RenderResult(c.cfg, c.addrWidth))

    val backend_alloc = Flipped(Decoupled(UInt(GlobalConfig.ddaTraceSlotBits.W)))
    val context_out = Decoupled(new BackendContextInit(c.cfg, c.addrWidth))
    val sdf_mem_wr = Flipped(new SdfMemWriteIO)
  })

  val contexts = Module(new FrontendContextManager(
    c.cfg,
    c.addrWidth,
    readPorts = 4,
    releasePorts = 3,
    stateWritePorts = 1
  ))
  val init = Module(new InitStage(c.cfg, c.addrWidth))
  val sdf = Module(new SdfStage(c.cfg, c.addrWidth))
  val initMissQ = Module(new Queue(
    new RenderResult(c.cfg, c.addrWidth),
    GlobalConfig.frontendContextCount,
    pipe = true,
    useSyncReadMem = true
  ))
  val sdfMissQ = Module(new Queue(
    new RenderResult(c.cfg, c.addrWidth),
    GlobalConfig.resultArbiterFifoDepth,
    pipe = true,
    useSyncReadMem = true
  ))
  val missArb = Module(new RRArbiter(new RenderResult(c.cfg, c.addrWidth), 2))

  // A context is reserved before Init.  Track that reservation directly
  // rather than recomputing active_count + initMissQ.count in the ingress
  // cone.  Init misses replace a released context with an initMissQ entry, so
  // their net reservation change is zero.
  val initCapacityUsed = RegInit(0.U(log2Ceil(GlobalConfig.frontendContextCount + 1).W))
  val initCapacityAvailable = initCapacityUsed < GlobalConfig.frontendContextCount.U
  contexts.io.ingress.valid := io.ingress.valid && initCapacityAvailable
  contexts.io.ingress.bits := io.ingress.bits
  io.ingress.ready := contexts.io.ingress.ready && initCapacityAvailable

  init.io.setup_origin := io.setup_origin
  init.io.setup_grid_min := io.setup_grid_min
  init.io.setup_grid_max := io.setup_grid_max
  init.io.in <> contexts.io.init_issue
  contexts.io.read_id(0) := init.io.context_id
  init.io.immutable := contexts.io.immutable(0)

  sdf.io.grid_min := io.setup_grid_min
  sdf.io.inv_voxel := io.inv_voxel
  sdf.io.inv_sub_voxel := io.inv_sub_voxel
  sdf.io.sdf_mem_wr <> io.sdf_mem_wr

  def missResult(rayId: UInt): RenderResult = {
    val result = Wire(new RenderResult(c.cfg, c.addrWidth))
    result.meta.rayId := rayId
    result.hit := false.B
    result.hitId := 0.U
    result.hitT := missT
    result.rgb8 := 0.U
    result
  }

  val initOut = init.io.out
  contexts.io.init_complete := initOut
  contexts.io.read_id(1) := initOut.bits.contextId
  initMissQ.io.enq.valid := initOut.valid && !initOut.bits.hit
  initMissQ.io.enq.bits := missResult(contexts.io.immutable(1).meta.rayId)
  when(initOut.valid && !initOut.bits.hit) {
    assert(initMissQ.io.enq.ready, "reserved Init miss buffer overflow")
  }
  sdf.io.issue_in <> contexts.io.sdf_issue

  contexts.io.read_id(2) := sdf.io.sdf_context_id
  sdf.io.sdf_immutable := contexts.io.immutable(2)
  sdf.io.sdf_state := contexts.io.sdf_state(2)
  contexts.io.state_write(0) := sdf.io.state_write
  contexts.io.read_id(3) := sdf.io.final_context_id
  sdf.io.final_immutable := contexts.io.immutable(3)
  sdf.io.final_state := contexts.io.sdf_state(3)

  val sdfOut = sdf.io.miss_out
  sdfMissQ.io.enq.valid := sdfOut.valid
  sdfMissQ.io.enq.bits := missResult(contexts.io.immutable(3).meta.rayId)
  sdfOut.ready := sdfMissQ.io.enq.ready

  contexts.io.release(0).valid := initOut.valid && !initOut.bits.hit
  contexts.io.release(0).bits := initOut.bits.contextId
  contexts.io.release(1).valid := sdfOut.fire
  contexts.io.release(1).bits := sdfOut.bits.contextId
  contexts.io.release(2).valid := sdf.io.dda_context_out.fire
  contexts.io.release(2).bits := sdf.io.dda_context_out.bits.sourceFrontendId

  sdf.io.dda_context_id <> io.backend_alloc
  io.context_out <> sdf.io.dda_context_out
  missArb.io.in(0) <> initMissQ.io.deq
  missArb.io.in(1) <> sdfMissQ.io.deq
  io.miss_out <> missArb.io.out

  val capacityReleaseCount = PopCount(Seq(
    initMissQ.io.deq.fire,
    contexts.io.release(1).valid,
    contexts.io.release(2).valid
  ))
  val capacityWithIngress = initCapacityUsed + contexts.io.ingress.fire.asUInt
  assert(capacityWithIngress >= capacityReleaseCount,
    "Frontend init reservation counter underflow")
  initCapacityUsed := capacityWithIngress - capacityReleaseCount
}
