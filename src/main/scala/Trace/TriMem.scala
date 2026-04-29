package Trace

import chisel3._
import chisel3.util._
import raytrace_utils._

class TriangleMemWrapper(val c: TriPeConfig) extends Module {
  val io = IO(new Bundle {
    val req      = Flipped(Decoupled(UInt(GlobalConfig.triMemAddrWidth.W)))
    val req_mask = Flipped(Decoupled(UInt(c.numPEs.W)))
    val resp     = Decoupled(new TriangleBlock(c))
  })
  val dpi_mem = Module(new TriangleMemDPI(c, latency = GlobalConfig.triMemDpiLatency))

  dpi_mem.io.clk := clock
  dpi_mem.io.reset := reset
  io.req.ready := dpi_mem.io.req_ready
  io.req_mask.ready := dpi_mem.io.req_ready
  dpi_mem.io.addr := io.req.bits
  dpi_mem.io.req_valid := io.req.valid
  dpi_mem.io.req_mask := io.req_mask.bits

  val block_data = Wire(new TriangleBlock(c))
  val bitsPerTri = 3 * 3 * c.cfg.totalWidth
  for(i <- 0 until  c.numPEs) {
    val hi = bitsPerTri*(i+1) - 1
    val lo = bitsPerTri*i
    val triBits = dpi_mem.io.data(hi, lo)
    block_data.tris(i).v0.x := triBits(31, 0)
    block_data.tris(i).v0.y := triBits(63, 32)
    block_data.tris(i).v0.z := triBits(95, 64)
    block_data.tris(i).v1.x := triBits(127, 96)
    block_data.tris(i).v1.y := triBits(159, 128)
    block_data.tris(i).v1.z := triBits(191, 160)
    block_data.tris(i).v2.x := triBits(223, 192)
    block_data.tris(i).v2.y := triBits(255, 224)
    block_data.tris(i).v2.z := triBits(287, 256)
    block_data.tris(i).id := dpi_mem.io.addr_q + i.U(GlobalConfig.triMemAddrWidth.W)
    block_data.mask(i) := dpi_mem.io.valid_mask(i)
  }

  io.resp.valid := dpi_mem.io.valid
  io.resp.bits  := block_data
}

class TriangleMemMultiPort(
  val c: TriPeConfig,
  val numPorts: Int,
  val numBanks: Int = GlobalConfig.triMemNumBanks
) extends Module {
  require(numPorts > 0, "TriangleMemMultiPort needs at least one port")
  require(numBanks > 0, "TriangleMemMultiPort needs at least one bank")

  private val srcW = math.max(1, log2Ceil(numPorts))
  private val bankSelW = math.max(1, log2Ceil(numBanks))

  class BankReq extends Bundle {
    val addr = UInt(GlobalConfig.triMemAddrWidth.W)
    val mask = UInt(c.numPEs.W)
    val src = UInt(srcW.W)
  }

  val io = IO(new Bundle {
    val req = Vec(numPorts, Flipped(Decoupled(UInt(GlobalConfig.triMemAddrWidth.W))))
    val req_mask = Vec(numPorts, Flipped(Decoupled(UInt(c.numPEs.W))))
    val resp = Vec(numPorts, Decoupled(new TriangleBlock(c)))
  })

  private def decodeBlock(data: UInt, addrQ: UInt, mask: UInt): TriangleBlock = {
    val block = Wire(new TriangleBlock(c))
    val bitsPerTri = 3 * 3 * c.cfg.totalWidth
    for (i <- 0 until c.numPEs) {
      val hi = bitsPerTri * (i + 1) - 1
      val lo = bitsPerTri * i
      val triBits = data(hi, lo)
      block.tris(i).v0.x := triBits(31, 0)
      block.tris(i).v0.y := triBits(63, 32)
      block.tris(i).v0.z := triBits(95, 64)
      block.tris(i).v1.x := triBits(127, 96)
      block.tris(i).v1.y := triBits(159, 128)
      block.tris(i).v1.z := triBits(191, 160)
      block.tris(i).v2.x := triBits(223, 192)
      block.tris(i).v2.y := triBits(255, 224)
      block.tris(i).v2.z := triBits(287, 256)
      block.tris(i).id := addrQ + i.U(GlobalConfig.triMemAddrWidth.W)
      block.mask(i) := mask(i)
    }
    block
  }

  val banks = Seq.fill(numBanks)(Module(new TriangleMemDPI(c, latency = GlobalConfig.triMemDpiLatency)))
  val arbs = Seq.fill(numBanks)(Module(new RRArbiter(new BankReq, numPorts)))

  val targetedBank = Wire(Vec(numPorts, UInt(bankSelW.W)))
  for (p <- 0 until numPorts) {
    targetedBank(p) := (if (numBanks == 1) 0.U else io.req(p).bits(bankSelW - 1, 0))
  }

  val reqReady = Wire(Vec(numPorts, Bool()))
  val reqMaskReady = Wire(Vec(numPorts, Bool()))
  reqReady := VecInit(Seq.fill(numPorts)(false.B))
  reqMaskReady := VecInit(Seq.fill(numPorts)(false.B))

  for (b <- 0 until numBanks) {
    for (p <- 0 until numPorts) {
      val targetsBank = targetedBank(p) === b.U
      arbs(b).io.in(p).valid := io.req(p).valid && io.req_mask(p).valid && targetsBank
      arbs(b).io.in(p).bits.addr := io.req(p).bits
      arbs(b).io.in(p).bits.mask := io.req_mask(p).bits
      arbs(b).io.in(p).bits.src := p.U

      when(targetsBank) {
        reqReady(p) := arbs(b).io.in(p).ready && banks(b).io.req_ready
        reqMaskReady(p) := arbs(b).io.in(p).ready && banks(b).io.req_ready
      }
    }

    banks(b).io.clk := clock
    banks(b).io.reset := reset
    banks(b).io.addr := arbs(b).io.out.bits.addr
    banks(b).io.req_valid := arbs(b).io.out.valid
    banks(b).io.req_mask := arbs(b).io.out.bits.mask
    arbs(b).io.out.ready := banks(b).io.req_ready
  }

  for (p <- 0 until numPorts) {
    io.req(p).ready := reqReady(p)
    io.req_mask(p).ready := reqMaskReady(p)
  }

  val respValid = Wire(Vec(numPorts, Bool()))
  val respBits = Wire(Vec(numPorts, new TriangleBlock(c)))
  for (p <- 0 until numPorts) {
    val defaultBlock = Wire(new TriangleBlock(c))
    for (i <- 0 until c.numPEs) {
      defaultBlock.tris(i) := 0.U.asTypeOf(new Triangle(c.cfg))
      defaultBlock.mask(i) := false.B
    }
    respValid(p) := false.B
    respBits(p) := defaultBlock
  }

  for (b <- 0 until numBanks) {
    val srcPipe = RegInit(VecInit(Seq.fill(GlobalConfig.triMemDpiLatency)(0.U(srcW.W))))
    when(reset.asBool) {
      for (i <- 0 until GlobalConfig.triMemDpiLatency) {
        srcPipe(i) := 0.U
      }
    }.otherwise {
      srcPipe(0) := Mux(arbs(b).io.out.fire, arbs(b).io.out.bits.src, 0.U)
      for (i <- 1 until GlobalConfig.triMemDpiLatency) {
        srcPipe(i) := srcPipe(i - 1)
      }
    }

    val bankBlock = decodeBlock(banks(b).io.data, banks(b).io.addr_q, banks(b).io.valid_mask)
    when(banks(b).io.valid) {
      respValid(srcPipe.last) := true.B
      respBits(srcPipe.last) := bankBlock
    }
  }

  for (p <- 0 until numPorts) {
    io.resp(p).valid := respValid(p)
    io.resp(p).bits := respBits(p)
  }
}
