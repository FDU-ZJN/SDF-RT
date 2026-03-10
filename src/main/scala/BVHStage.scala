package sdf_rt
import chisel3._
import chisel3.util._
import raytrace_utils._

class BVHStage(val c: BvhPeConfig) extends Module {
  val io = IO(new Bundle {
    val start = Input(Bool())
    val rootNode = Input(UInt(c.addrWidth.W))
    val ray_in = Input(new Ray(c.cfg))
    val hit_update_valid = Input(Bool())
    val hit_update_t = Input(UInt(c.cfg.totalWidth.W))
    val leaf_out = Decoupled(new TriBatch(c.addrWidth))
    val busy = Output(Bool())
    val done = Output(Bool())
    val stack_level = Output(UInt(log2Ceil(c.stackDepth + 1).W))
  })

  val pe = Module(new BvhPE(c))
  val mem = Module(new BVHMenDPI(addrWidth = c.addrWidth, nodeBytes = 40))

  // Connect PE IO
  pe.io.start := io.start
  pe.io.rootNode := io.rootNode
  pe.io.ray_in := io.ray_in
  pe.io.hit_update_valid := io.hit_update_valid
  pe.io.hit_update_t := io.hit_update_t

  io.leaf_out <> pe.io.leaf_out
  io.busy := pe.io.busy
  io.done := pe.io.done
  io.stack_level := pe.io.stack_level

  // Connect PE node_req to memory
  mem.io.clk := clock
  mem.io.reset := reset
  mem.io.addr := pe.io.node_req.bits
  mem.io.en := pe.io.node_req.valid
  pe.io.node_req.ready := mem.io.valid

  // Unpack memory data to BvhNode
  val nodeRaw = mem.io.data
  val nodeResp = Wire(new BvhNode(c.cfg, c.addrWidth))
  nodeResp.bounds.min.x := nodeRaw(31, 0)
  nodeResp.bounds.min.y := nodeRaw(63, 32)
  nodeResp.bounds.min.z := nodeRaw(95, 64)
  nodeResp.bounds.max.x := nodeRaw(127, 96)
  nodeResp.bounds.max.y := nodeRaw(159, 128)
  nodeResp.bounds.max.z := nodeRaw(191, 160)

  nodeResp.left := nodeRaw(223, 192)
  nodeResp.right := nodeRaw(255, 224)
  nodeResp.triStart := nodeRaw(287, 256)
  nodeResp.triCount := nodeRaw(319,288)
  nodeResp.isLeaf := nodeResp.triCount===0.U
  nodeResp.leftValid := nodeResp.left.asSInt>=0.S
  nodeResp.rightValid := nodeResp.right.asSInt>=0.S

  pe.io.node_resp.valid := mem.io.valid
  pe.io.node_resp.bits := nodeResp
}
