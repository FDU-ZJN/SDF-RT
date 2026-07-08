package Trace

import chisel3._
import chisel3.util._
import raytrace_utils._

class TriMemReq(val c: TriPeConfig, val tagWidth: Int) extends Bundle {
  val addr = UInt(GlobalConfig.triMemAddrWidth.W)
  val mask = UInt(c.numPEs.W)
  val tag = UInt(tagWidth.W)
}

class TriMemResp(val c: TriPeConfig, val tagWidth: Int) extends Bundle {
  val block = new TriangleBlock(c)
  val tag = UInt(tagWidth.W)
}

class TriCacheDataArrayBB(
  val dataWidth: Int,
  val depth: Int
) extends BlackBox(
      Map(
        "DATA_WIDTH" -> dataWidth,
        "DEPTH" -> depth,
        "ADDR_WIDTH" -> math.max(1, log2Ceil(depth))
      )
    )
    with HasBlackBoxResource {
  val addrWidth = math.max(1, log2Ceil(depth))
  val io = IO(new Bundle {
    val clk = Input(Clock())
    val reset = Input(Reset())
    val rd_en = Input(Bool())
    val rd_addr = Input(UInt(addrWidth.W))
    val rd_data = Output(UInt(dataWidth.W))
    val wr_en = Input(Bool())
    val wr_addr = Input(UInt(addrWidth.W))
    val wr_data = Input(UInt(dataWidth.W))
  })
  addResource("/TriCacheDataArray.sv")
}

class TriCacheDataArray(
  val dataWidth: Int,
  val depth: Int,
  val useXpm: Boolean
) extends Module {
  val addrWidth = math.max(1, log2Ceil(depth))
  val io = IO(new Bundle {
    val rd_en = Input(Bool())
    val rd_addr = Input(UInt(addrWidth.W))
    val rd_data = Output(UInt(dataWidth.W))
    val wr_en = Input(Bool())
    val wr_addr = Input(UInt(addrWidth.W))
    val wr_data = Input(UInt(dataWidth.W))
  })

  if (useXpm) {
    val ram = Module(new TriCacheDataArrayBB(dataWidth, depth))
    ram.io.clk := clock
    ram.io.reset := reset
    ram.io.rd_en := io.rd_en
    ram.io.rd_addr := io.rd_addr
    ram.io.wr_en := io.wr_en
    ram.io.wr_addr := io.wr_addr
    ram.io.wr_data := io.wr_data
    io.rd_data := ram.io.rd_data
  } else {
    val mem = SyncReadMem(depth, UInt(dataWidth.W))
    io.rd_data := mem.read(io.rd_addr, io.rd_en)
    when(io.wr_en) {
      mem.write(io.wr_addr, io.wr_data)
    }
  }
}

class TriCacheStatsMonitor extends BlackBox with HasBlackBoxInline {
  override def desiredName: String = "TriCacheStatsMonitor"

  val io = IO(new Bundle {
    val clk = Input(Clock())
    val reset = Input(Reset())
    val valid = Input(Bool())
    val hit = Input(Bool())
    val bank = Input(UInt(32.W))
  })

  setInline(
    "TriCacheStatsMonitor.sv",
    """import "DPI-C" function void tri_cache_stats_record(input int bank, input int hit);
      |
      |module TriCacheStatsMonitor (
      |  input clk,
      |  input reset,
      |  input valid,
      |  input hit,
      |  input [31:0] bank
      |);
      |  always @(posedge clk) begin
      |    if (!reset && valid) begin
      |      tri_cache_stats_record(bank, {31'b0, hit});
      |    end
      |  end
      |endmodule
      |""".stripMargin
  )
}

class TriCacheRefillStatsMonitor extends BlackBox with HasBlackBoxInline {
  override def desiredName: String = "TriCacheRefillStatsMonitor"

  val io = IO(new Bundle {
    val clk = Input(Clock())
    val reset = Input(Reset())
    val busyCycle = Input(Bool())
    val stallCycle = Input(Bool())
    val refillFire = Input(Bool())
  })

  setInline(
    "TriCacheRefillStatsMonitor.sv",
    """import "DPI-C" function void tri_cache_refill_stats_record(
      |  input int busy_cycle,
      |  input int stall_cycle,
      |  input int refill_fire
      |);
      |
      |module TriCacheRefillStatsMonitor (
      |  input clk,
      |  input reset,
      |  input busyCycle,
      |  input stallCycle,
      |  input refillFire
      |);
      |  always @(posedge clk) begin
      |    if (!reset && (busyCycle || stallCycle || refillFire)) begin
      |      tri_cache_refill_stats_record({31'b0, busyCycle}, {31'b0, stallCycle}, {31'b0, refillFire});
      |    end
      |  end
      |endmodule
      |""".stripMargin
  )
}

class TriMemRefillReq(numBanks: Int) extends Bundle {
  val bank = UInt(math.max(1, log2Ceil(numBanks)).W)
  val addr = UInt(GlobalConfig.triMemAddrWidth.W)
}

class TriangleMemSharedDPI(
  val c: TriPeConfig,
  val latency: Int = 1,
  val numBanks: Int = GlobalConfig.triMemNumBanks,
  val maxEntries: Int = GlobalConfig.triMemBankDepth
) extends BlackBox with HasBlackBoxInline {
  private val totalBits = c.numPEs * 9 * c.cfg.totalWidth

  val io = IO(new Bundle {
    val clk = Input(Clock())
    val reset = Input(Reset())
    val bank = Input(UInt(32.W))
    val addr = Input(UInt(GlobalConfig.triMemAddrWidth.W))
    val req_valid = Input(Bool())
    val data = Output(UInt(totalBits.W))
    val valid = Output(Bool())
    val req_ready = Output(Bool())
  })

  setInline(
    "TriangleMemSharedDPI.sv",
    s"""import "DPI-C" function void tri_mem_read_bank(input int bank, input int addr, output byte data[]);
       |
       |module TriangleMemSharedDPI (
       |  input clk,
       |  input reset,
       |  input [31:0] bank,
       |  input [${GlobalConfig.triMemAddrWidth - 1}:0] addr,
       |  input req_valid,
       |  output [${totalBits - 1}:0] data,
       |  output valid,
       |  output req_ready
       |);
       |  byte raw_buffer[${c.numPEs * 3 * 3 * (c.cfg.totalWidth / 8)}];
       |  reg [${totalBits - 1}:0] data_pipe[0:${latency - 1}];
       |  reg [${latency - 1}:0] valid_pipe;
       |  integer i;
       |  integer j;
       |
       |  assign req_ready = 1'b1;
       |
       |  always @(posedge clk) begin
       |    if (reset) begin
       |      valid_pipe <= '0;
       |      for (j = 0; j < ${latency}; j = j + 1) begin
       |        data_pipe[j] <= '0;
       |      end
       |    end else begin
       |      valid_pipe[0] <= req_valid;
       |      if (req_valid) begin
       |        tri_mem_read_bank(bank, addr, raw_buffer);
       |        for (i = 0; i < ${c.numPEs * 3 * 3 * (c.cfg.totalWidth / 8)}; i = i + 1) begin
       |          data_pipe[0][i*8 +: 8] <= raw_buffer[i];
       |        end
       |      end
       |      for (j = 1; j < ${latency}; j = j + 1) begin
       |        valid_pipe[j] <= valid_pipe[j - 1];
       |        data_pipe[j] <= data_pipe[j - 1];
       |      end
       |    end
       |  end
       |
       |  assign data = data_pipe[${latency - 1}];
       |  assign valid = valid_pipe[${latency - 1}];
       |endmodule
       |""".stripMargin
  )
}

class TriangleMemWrapper(val c: TriPeConfig) extends Module {
  require(isPow2(c.numPEs), s"TriangleMemWrapper requires numPEs to be power-of-two, got ${c.numPEs}")

  val io = IO(new Bundle {
    val req      = Flipped(Decoupled(UInt(GlobalConfig.triMemAddrWidth.W)))
    val req_mask = Flipped(Decoupled(UInt(c.numPEs.W)))
    val resp     = Decoupled(new TriangleBlock(c))
  })
  val dpiMem = Module(new TriangleMemDPI(c, latency = GlobalConfig.triMemDpiLatency))

  dpiMem.io.clk := clock
  dpiMem.io.reset := reset
  io.req.ready := dpiMem.io.req_ready
  io.req_mask.ready := dpiMem.io.req_ready
  dpiMem.io.addr := io.req.bits
  dpiMem.io.req_valid := io.req.valid
  dpiMem.io.req_mask := io.req_mask.bits

  val blockData = Wire(new TriangleBlock(c))
  val bitsPerTri = 3 * 3 * c.cfg.totalWidth
  for (i <- 0 until c.numPEs) {
    val hi = bitsPerTri * (i + 1) - 1
    val lo = bitsPerTri * i
    val triBits = dpiMem.io.data(hi, lo)
    blockData.tris(i).v0.x := triBits(31, 0)
    blockData.tris(i).v0.y := triBits(63, 32)
    blockData.tris(i).v0.z := triBits(95, 64)
    blockData.tris(i).v1.x := triBits(127, 96)
    blockData.tris(i).v1.y := triBits(159, 128)
    blockData.tris(i).v1.z := triBits(191, 160)
    blockData.tris(i).v2.x := triBits(223, 192)
    blockData.tris(i).v2.y := triBits(255, 224)
    blockData.tris(i).v2.z := triBits(287, 256)
    blockData.tris(i).id := dpiMem.io.addr_q * c.numPEs.U(GlobalConfig.triMemAddrWidth.W) +
      i.U(GlobalConfig.triMemAddrWidth.W)
    blockData.mask(i) := dpiMem.io.valid_mask(i)
  }

  io.resp.valid := dpiMem.io.valid
  io.resp.bits := blockData
}

class TriMemBackendReq(val c: TriPeConfig, val numBanks: Int, val idWidth: Int) extends Bundle {
  val bank = UInt(math.max(1, log2Ceil(numBanks)).W)
  val addr = UInt(GlobalConfig.triMemAddrWidth.W)
  val id = UInt(idWidth.W)
}

class TriMemBackendResp(val c: TriPeConfig, val idWidth: Int) extends Bundle {
  val data = UInt((c.numPEs * 9 * c.cfg.totalWidth).W)
  val id = UInt(idWidth.W)
}

class TriMemAllocReq(
  val c: TriPeConfig,
  val numBanks: Int,
  val numSets: Int,
  val idWidth: Int
) extends Bundle {
  val bank = UInt(math.max(1, log2Ceil(numBanks)).W)
  val addr = UInt(GlobalConfig.triMemAddrWidth.W)
  val set = UInt(math.max(1, log2Ceil(numSets)).W)
  val tag = UInt((GlobalConfig.triMemAddrWidth - math.max(1, log2Ceil(numSets))).W)
  val victimWay = UInt(1.W)
}

class TriMemAllocResp(val idWidth: Int) extends Bundle {
  val id = UInt(idWidth.W)
  val merged = Bool()
}

class TriMemRefillDone(val c: TriPeConfig, val idWidth: Int) extends Bundle {
  val id = UInt(idWidth.W)
  val data = UInt((c.numPEs * 9 * c.cfg.totalWidth).W)
}

class TriMemRelease(val idWidth: Int) extends Bundle {
  val id = UInt(idWidth.W)
}

class TriangleMemRefillBackend(
  val c: TriPeConfig,
  val numBanks: Int,
  val idWidth: Int
) extends Module {
  private val totalBits = c.numPEs * 9 * c.cfg.totalWidth
  private val bankSelW = math.max(1, log2Ceil(numBanks))

  val io = IO(new Bundle {
    val req = Flipped(Decoupled(new TriMemBackendReq(c, numBanks, idWidth)))
    val resp = Decoupled(new TriMemBackendResp(c, idWidth))
  })

  GlobalConfig.memImplMode match {
    case 0 =>
      val mem = Module(new TriangleMemSharedDPI(c, latency = 1, numBanks = numBanks))
      val idPipe = RegInit(0.U(idWidth.W))
      val validPipe = RegInit(false.B)

      mem.io.clk := clock
      mem.io.reset := reset
      mem.io.bank := io.req.bits.bank
      mem.io.addr := io.req.bits.addr
      mem.io.req_valid := io.req.valid

      io.req.ready := mem.io.req_ready
      when(io.req.fire) {
        idPipe := io.req.bits.id
      }
      validPipe := io.req.fire

      io.resp.valid := validPipe && mem.io.valid
      io.resp.bits.id := idPipe
      io.resp.bits.data := mem.io.data

    case 2 =>
      val mems = Seq.tabulate(numBanks)(b => Module(new TriangleMemDPI(
        c,
        latency = 1,
        bankId = b,
        numBanks = numBanks,
        maxEntries = GlobalConfig.triMemBankDepth
      )))
      val respArb = Module(new RRArbiter(new TriMemBackendResp(c, idWidth), numBanks))
      val idPipes = Seq.fill(numBanks)(RegInit(0.U(idWidth.W)))
      val validPipes = Seq.fill(numBanks)(RegInit(false.B))

      io.req.ready := false.B
      for (b <- 0 until numBanks) {
        val hitBank = io.req.bits.bank === b.U(bankSelW.W)
        mems(b).io.clk := clock
        mems(b).io.reset := reset
        mems(b).io.addr := io.req.bits.addr
        mems(b).io.req_valid := io.req.valid && hitBank
        mems(b).io.req_mask := Fill(c.numPEs, 1.U(1.W))
        when(hitBank) {
          io.req.ready := mems(b).io.req_ready
        }
        when(io.req.fire && hitBank) {
          idPipes(b) := io.req.bits.id
        }
        validPipes(b) := io.req.fire && hitBank

        respArb.io.in(b).valid := validPipes(b) && mems(b).io.valid
        respArb.io.in(b).bits.id := idPipes(b)
        respArb.io.in(b).bits.data := mems(b).io.data
      }

      io.resp <> respArb.io.out
  }
}

class TriangleMemCachedBank(
  val c: TriPeConfig,
  val srcWidth: Int,
  val tagWidth: Int,
  val bankId: Int,
  val numBanks: Int = GlobalConfig.triMemNumBanks,
  val numSets: Int = GlobalConfig.triMemCacheSets,
  val ways: Int = GlobalConfig.triMemCacheWays,
  val reqQueueDepth: Int = GlobalConfig.triMemReqQueueDepth,
  val mergeQueueDepth: Int = GlobalConfig.triMemMergeQueueDepth,
  val mshrEntries: Int = GlobalConfig.triMemMshrEntries
) extends Module {
  require(ways == 2, s"TriangleMemCachedBank currently supports exactly 2 ways, got $ways")
  require(isPow2(numSets), s"TriangleMemCachedBank requires power-of-two numSets, got $numSets")
  require(reqQueueDepth > 0, "TriangleMemCachedBank reqQueueDepth must be > 0")
  require(mergeQueueDepth > 0, "TriangleMemCachedBank mergeQueueDepth must be > 0")
  require(mshrEntries > 0, "TriangleMemCachedBank mshrEntries must be > 0")

  private val bitsPerTri = 3 * 3 * c.cfg.totalWidth
  private val totalBits = c.numPEs * bitsPerTri
  private val setIdxW = math.max(1, log2Ceil(numSets))
  private val tagW = GlobalConfig.triMemAddrWidth - setIdxW
  private val mergeIdxW = math.max(1, log2Ceil(mergeQueueDepth))
  private val mergeCountW = log2Ceil(mergeQueueDepth + 1)
  private val mshrIdW = math.max(1, log2Ceil(mshrEntries))
  private val useXpmCache = GlobalConfig.memImplMode == 2

  class BankReq extends Bundle {
    val addr = UInt(GlobalConfig.triMemAddrWidth.W)
    val mask = UInt(c.numPEs.W)
    val src = UInt(srcWidth.W)
    val tag = UInt(tagWidth.W)
  }

  class BankResp extends Bundle {
    val src = UInt(srcWidth.W)
    val resp = new TriMemResp(c, tagWidth)
  }

  val io = IO(new Bundle {
    val req = Flipped(Decoupled(new BankReq))
    val resp = Decoupled(new BankResp)
    val stat = Valid(Bool())
    val allocReq = Decoupled(new TriMemAllocReq(c, numBanks, numSets, mshrIdW))
    val allocResp = Flipped(Decoupled(new TriMemAllocResp(mshrIdW)))
    val refillDone = Flipped(Decoupled(new TriMemRefillDone(c, mshrIdW)))
    val release = Valid(new TriMemRelease(mshrIdW))
  })

  private def setIdxOf(addr: UInt): UInt = {
    if (setIdxW == 0) 0.U else addr(setIdxW - 1, 0)
  }

  private def tagOf(addr: UInt): UInt = {
    if (tagW == 0) 0.U else addr(GlobalConfig.triMemAddrWidth - 1, setIdxW)
  }

  private def decodeBlock(data: UInt, addrQ: UInt, mask: UInt): TriangleBlock = {
    val block = Wire(new TriangleBlock(c))
    val numBanksU = numBanks.U(GlobalConfig.triMemAddrWidth.W)
    val globalBlock = addrQ * numBanksU + bankId.U(GlobalConfig.triMemAddrWidth.W)
    val bankBase = globalBlock * c.numPEs.U(GlobalConfig.triMemAddrWidth.W)
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
      block.tris(i).id := bankBase + i.U(GlobalConfig.triMemAddrWidth.W)
      block.mask(i) := mask(i)
    }
    block
  }

  val reqQ = Module(new Queue(new BankReq, reqQueueDepth))
  reqQ.io.enq <> io.req

  val cacheData = Seq.fill(ways)(Module(new TriCacheDataArray(totalBits, numSets, useXpmCache)))
  val cacheValid = RegInit(VecInit(Seq.fill(ways)(VecInit(Seq.fill(numSets)(false.B)))))
  val cacheTag = RegInit(VecInit(Seq.fill(ways)(VecInit(Seq.fill(numSets)(0.U(tagW.W))))))
  val cacheLru = RegInit(VecInit(Seq.fill(numSets)(false.B)))

  val sIdle :: sHitRead :: sHitResp :: sAllocWait :: sEmitResp :: Nil = Enum(5)
  val state = RegInit(sIdle)

  val hitWayReg = Reg(UInt(1.W))
  val hitMaskReg = Reg(UInt(c.numPEs.W))
  val hitAddrReg = Reg(UInt(GlobalConfig.triMemAddrWidth.W))
  val hitSrcReg = Reg(UInt(srcWidth.W))
  val hitTagReg = Reg(UInt(tagWidth.W))
  val hitDataReg = Reg(UInt(totalBits.W))
  val allocReqReg = Reg(new BankReq)
  val allocSetReg = Reg(UInt(setIdxW.W))
  val allocTagReg = Reg(UInt(tagW.W))
  val allocVictimWayReg = Reg(UInt(1.W))
  val resumeAllocAfterEmit = RegInit(false.B)

  val pendingReqs = Reg(Vec(mshrEntries, Vec(mergeQueueDepth, new BankReq)))
  val pendingCount = RegInit(VecInit(Seq.fill(mshrEntries)(0.U(mergeCountW.W))))
  val pendingAddr = Reg(Vec(mshrEntries, UInt(GlobalConfig.triMemAddrWidth.W)))
  val pendingSet = Reg(Vec(mshrEntries, UInt(setIdxW.W)))
  val pendingTag = Reg(Vec(mshrEntries, UInt(tagW.W)))
  val pendingVictimWay = Reg(Vec(mshrEntries, UInt(1.W)))

  val emitMshrId = Reg(UInt(mshrIdW.W))
  val emitIdx = RegInit(0.U(mergeCountW.W))
  val emitTotal = RegInit(0.U(mergeCountW.W))
  val emitDataReg = Reg(UInt(totalBits.W))

  val way0ReadEn = WireDefault(false.B)
  val way1ReadEn = WireDefault(false.B)
  val readSetIdx = WireDefault(0.U(setIdxW.W))
  val way0WriteEn = WireDefault(false.B)
  val way1WriteEn = WireDefault(false.B)
  val writeSetIdx = WireDefault(0.U(setIdxW.W))
  val writeData = WireDefault(0.U(totalBits.W))

  cacheData(0).io.rd_en := way0ReadEn
  cacheData(0).io.rd_addr := readSetIdx
  cacheData(0).io.wr_en := way0WriteEn
  cacheData(0).io.wr_addr := writeSetIdx
  cacheData(0).io.wr_data := writeData
  cacheData(1).io.rd_en := way1ReadEn
  cacheData(1).io.rd_addr := readSetIdx
  cacheData(1).io.wr_en := way1WriteEn
  cacheData(1).io.wr_addr := writeSetIdx
  cacheData(1).io.wr_data := writeData

  val way0ReadData = cacheData(0).io.rd_data
  val way1ReadData = cacheData(1).io.rd_data
  val statsValid = WireDefault(false.B)
  val statsHit = WireDefault(false.B)
  io.stat.valid := statsValid
  io.stat.bits := statsHit
  io.allocReq.valid := false.B
  io.allocReq.bits.bank := bankId.U
  io.allocReq.bits.addr := allocReqReg.addr
  io.allocReq.bits.set := allocSetReg
  io.allocReq.bits.tag := allocTagReg
  io.allocReq.bits.victimWay := allocVictimWayReg
  io.allocResp.ready := false.B
  io.refillDone.ready := false.B
  io.release.valid := false.B
  io.release.bits.id := emitMshrId

  val headReq = reqQ.io.deq.bits
  val headSet = setIdxOf(headReq.addr)
  val headTag = tagOf(headReq.addr)
  val headHitWay0 = cacheValid(0)(headSet) && cacheTag(0)(headSet) === headTag
  val headHitWay1 = cacheValid(1)(headSet) && cacheTag(1)(headSet) === headTag
  val headHit = headHitWay0 || headHitWay1

  reqQ.io.deq.ready := false.B

  def enqueuePending(mshrId: UInt, req: BankReq, set: UInt, tag: UInt, victimWay: UInt): Unit = {
    val count = pendingCount(mshrId)
    assert(count =/= mergeQueueDepth.U, s"TriangleMemCachedBank[$bankId] merge queue overflow")
    pendingReqs(mshrId)(count(mergeIdxW - 1, 0)) := req
    pendingCount(mshrId) := count + 1.U
    when(count === 0.U) {
      pendingAddr(mshrId) := req.addr
      pendingSet(mshrId) := set
      pendingTag(mshrId) := tag
      pendingVictimWay(mshrId) := victimWay
    }
  }

  when((state === sIdle || state === sAllocWait) && io.refillDone.valid) {
    io.refillDone.ready := true.B
    val doneId = io.refillDone.bits.id
    assert(pendingCount(doneId) =/= 0.U, s"TriangleMemCachedBank[$bankId] refill done with no pending waiters")
    resumeAllocAfterEmit := state === sAllocWait
    writeSetIdx := pendingSet(doneId)
    writeData := io.refillDone.bits.data
    emitMshrId := doneId
    emitIdx := 0.U
    emitTotal := pendingCount(doneId)
    emitDataReg := io.refillDone.bits.data
    when(pendingVictimWay(doneId) === 0.U) {
      way0WriteEn := true.B
    }.otherwise {
      way1WriteEn := true.B
    }
    cacheValid(pendingVictimWay(doneId))(pendingSet(doneId)) := true.B
    cacheTag(pendingVictimWay(doneId))(pendingSet(doneId)) := pendingTag(doneId)
    cacheLru(pendingSet(doneId)) := !pendingVictimWay(doneId)
    state := sEmitResp
  }.elsewhen(state === sIdle && reqQ.io.deq.valid) {
    reqQ.io.deq.ready := true.B
    statsValid := true.B
    statsHit := headHit
    when(headHit) {
      hitWayReg := Mux(headHitWay0, 0.U, 1.U)
      hitMaskReg := headReq.mask
      hitAddrReg := headReq.addr
      hitSrcReg := headReq.src
      hitTagReg := headReq.tag
      readSetIdx := headSet
      way0ReadEn := headHitWay0
      way1ReadEn := headHitWay1
      state := sHitRead
      when(headHitWay0) {
        cacheLru(headSet) := true.B
      }.otherwise {
        cacheLru(headSet) := false.B
      }
    }.otherwise {
      allocReqReg := headReq
      allocSetReg := headSet
      allocTagReg := headTag
      allocVictimWayReg := Mux(!cacheValid(0)(headSet), 0.U, Mux(!cacheValid(1)(headSet), 1.U, cacheLru(headSet)))
      state := sAllocWait
    }
  }.elsewhen(state === sHitRead) {
    hitDataReg := Mux(hitWayReg === 0.U, way0ReadData, way1ReadData)
    state := sHitResp
  }.elsewhen(state === sAllocWait) {
    io.allocReq.valid := true.B
    io.allocReq.bits.addr := allocReqReg.addr
    io.allocReq.bits.set := allocSetReg
    io.allocReq.bits.tag := allocTagReg
    io.allocReq.bits.victimWay := allocVictimWayReg
    io.allocResp.ready := true.B
    when(io.allocResp.fire) {
      enqueuePending(io.allocResp.bits.id, allocReqReg, allocSetReg, allocTagReg, allocVictimWayReg)
      resumeAllocAfterEmit := false.B
      state := sIdle
    }
  }

  val hitRespBlock = decodeBlock(hitDataReg, hitAddrReg, hitMaskReg)
  val emitReq = Wire(new BankReq)
  emitReq := pendingReqs(emitMshrId)(emitIdx(mergeIdxW - 1, 0))
  val missRespBlock = decodeBlock(emitDataReg, pendingAddr(emitMshrId), emitReq.mask)

  io.resp.valid := false.B
  io.resp.bits := 0.U.asTypeOf(new BankResp)

  when(state === sHitResp) {
    io.resp.valid := true.B
    io.resp.bits.src := hitSrcReg
    io.resp.bits.resp.block := hitRespBlock
    io.resp.bits.resp.tag := hitTagReg
    when(io.resp.ready) {
      state := sIdle
    }
  }.elsewhen(state === sEmitResp) {
    io.resp.valid := true.B
    io.resp.bits.src := emitReq.src
    io.resp.bits.resp.block := missRespBlock
    io.resp.bits.resp.tag := emitReq.tag
    when(io.resp.ready) {
      when(emitIdx + 1.U >= emitTotal) {
        pendingCount(emitMshrId) := 0.U
        io.release.valid := true.B
        when(resumeAllocAfterEmit) {
          state := sAllocWait
        }.otherwise {
          state := sIdle
        }
      }.otherwise {
        emitIdx := emitIdx + 1.U
      }
    }
  }
}

class TriangleMemMultiPort(
  val c: TriPeConfig,
  val numPorts: Int,
  val tagWidth: Int,
  val numBanks: Int = GlobalConfig.triMemNumBanks,
  val mshrEntries: Int = GlobalConfig.triMemMshrEntries
) extends Module {
  require(numPorts > 0, "TriangleMemMultiPort needs at least one port")
  require(numBanks > 0, "TriangleMemMultiPort needs at least one bank")
  require(isPow2(numBanks), s"TriangleMemMultiPort currently requires numBanks to be power-of-two, got $numBanks")
  require(isPow2(c.numPEs), s"TriangleMemMultiPort requires numPEs to be power-of-two, got ${c.numPEs}")

  private val srcW = math.max(1, log2Ceil(numPorts))
  private val bankSelW = math.max(1, log2Ceil(numBanks))
  private val mshrIdW = math.max(1, log2Ceil(mshrEntries))

  class BankReq extends Bundle {
    val addr = UInt(GlobalConfig.triMemAddrWidth.W)
    val mask = UInt(c.numPEs.W)
    val src = UInt(srcW.W)
    val tag = UInt(tagWidth.W)
  }

  val io = IO(new Bundle {
    val req = Vec(numPorts, Flipped(Decoupled(new TriMemReq(c, tagWidth))))
    val resp = Vec(numPorts, Decoupled(new TriMemResp(c, tagWidth)))
  })

  private def decodeBlock(data: UInt, addrQ: UInt, mask: UInt, bankId: Int): TriangleBlock = {
    val block = Wire(new TriangleBlock(c))
    val bitsPerTri = 3 * 3 * c.cfg.totalWidth
    val numBanksU = numBanks.U(GlobalConfig.triMemAddrWidth.W)
    val globalBlock = addrQ * numBanksU + bankId.U(GlobalConfig.triMemAddrWidth.W)
    val bankBase = globalBlock * c.numPEs.U(GlobalConfig.triMemAddrWidth.W)
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
      block.tris(i).id := bankBase + i.U(GlobalConfig.triMemAddrWidth.W)
      block.mask(i) := mask(i)
    }
    block
  }

  GlobalConfig.memImplMode match {
    case 0 | 2 =>
      val banks = Seq.tabulate(numBanks)(b => Module(new TriangleMemCachedBank(c, srcW, tagWidth, b, numBanks, mshrEntries = mshrEntries)))
      val statMons = Seq.fill(numBanks)(Module(new TriCacheStatsMonitor))
      val refillStatMon = Module(new TriCacheRefillStatsMonitor)
      val reqArbs = Seq.fill(numBanks)(Module(new RRArbiter(new BankReq, numPorts)))
      val allocArb = Module(new RRArbiter(new TriMemAllocReq(c, numBanks, GlobalConfig.triMemCacheSets, mshrIdW), numBanks))
      val backend = Module(new TriangleMemRefillBackend(c, numBanks, mshrIdW))
      val mshrValid = RegInit(VecInit(Seq.fill(mshrEntries)(false.B)))
      val mshrIssued = RegInit(VecInit(Seq.fill(mshrEntries)(false.B)))
      val mshrDone = RegInit(VecInit(Seq.fill(mshrEntries)(false.B)))
      val mshrDelivered = RegInit(VecInit(Seq.fill(mshrEntries)(false.B)))
      val mshrBank = Reg(Vec(mshrEntries, UInt(bankSelW.W)))
      val mshrAddr = Reg(Vec(mshrEntries, UInt(GlobalConfig.triMemAddrWidth.W)))
      val mshrData = Reg(Vec(mshrEntries, UInt((c.numPEs * 9 * c.cfg.totalWidth).W)))

      for (b <- 0 until numBanks) {
        statMons(b).io.clk := clock
        statMons(b).io.reset := reset
        statMons(b).io.valid := banks(b).io.stat.valid
        statMons(b).io.hit := banks(b).io.stat.bits
        statMons(b).io.bank := b.U
        allocArb.io.in(b) <> banks(b).io.allocReq

        val doneMatches = Wire(Vec(mshrEntries, Bool()))
        for (i <- 0 until mshrEntries) {
          doneMatches(i) := mshrValid(i) && mshrDone(i) && !mshrDelivered(i) && mshrBank(i) === b.U
        }
        val doneSelOH = PriorityEncoderOH(doneMatches)
        val doneSel = OHToUInt(doneSelOH)
        banks(b).io.refillDone.valid := doneMatches.asUInt.orR
        banks(b).io.refillDone.bits.id := doneSel
        banks(b).io.refillDone.bits.data := Mux1H(doneSelOH, mshrData)
        when(banks(b).io.refillDone.fire) {
          mshrDelivered(doneSel) := true.B
        }

        val allocMatches = Wire(Vec(mshrEntries, Bool()))
        for (i <- 0 until mshrEntries) {
          allocMatches(i) := mshrValid(i) &&
            mshrBank(i) === allocArb.io.out.bits.bank &&
            mshrAddr(i) === allocArb.io.out.bits.addr
        }
        val allocSelOH = PriorityEncoderOH(allocMatches)
        val allocSel = OHToUInt(allocSelOH)
        val freeVec = VecInit((0 until mshrEntries).map(i => !mshrValid(i)))
        val freeOH = PriorityEncoderOH(freeVec)
        val freeValid = freeVec.asUInt.orR
        val freeSel = OHToUInt(freeOH)
        val bankGranted = allocArb.io.out.valid && allocArb.io.out.bits.bank === b.U
        val bankMerge = allocMatches.asUInt.orR
        val bankAllocValid = bankGranted && (bankMerge || freeValid)
        banks(b).io.allocResp.valid := bankAllocValid
        banks(b).io.allocResp.bits.id := Mux(bankMerge, allocSel, freeSel)
        banks(b).io.allocResp.bits.merged := bankMerge

        when(bankGranted && banks(b).io.allocResp.fire && !bankMerge) {
          mshrValid(freeSel) := true.B
          mshrIssued(freeSel) := false.B
          mshrDone(freeSel) := false.B
          mshrDelivered(freeSel) := false.B
          mshrBank(freeSel) := b.U
          mshrAddr(freeSel) := allocArb.io.out.bits.addr
        }

        when(banks(b).io.release.valid) {
          val relId = banks(b).io.release.bits.id
          mshrValid(relId) := false.B
          mshrIssued(relId) := false.B
          mshrDone(relId) := false.B
          mshrDelivered(relId) := false.B
          mshrAddr(relId) := 0.U
          mshrData(relId) := 0.U
        }
      }

      val mergeHits = Wire(Vec(mshrEntries, Bool()))
      for (i <- 0 until mshrEntries) {
        mergeHits(i) := mshrValid(i) &&
          mshrBank(i) === allocArb.io.out.bits.bank &&
          mshrAddr(i) === allocArb.io.out.bits.addr
      }
      val freeVec = VecInit((0 until mshrEntries).map(i => !mshrValid(i)))
      val freeOH = PriorityEncoderOH(freeVec)
      val freeExists = freeVec.asUInt.orR
      val mergeExists = mergeHits.asUInt.orR
      val allocCanServe = mergeExists || freeExists
      allocArb.io.out.ready := allocCanServe

      val issueCandidates = Wire(Vec(mshrEntries, Bool()))
      for (i <- 0 until mshrEntries) {
        issueCandidates(i) := mshrValid(i) && !mshrIssued(i)
      }
      val issueOH = PriorityEncoderOH(issueCandidates)
      val issueValid = issueCandidates.asUInt.orR
      val issueSel = OHToUInt(issueOH)
      backend.io.req.valid := issueValid
      backend.io.req.bits.bank := Mux(issueValid, mshrBank(issueSel), 0.U)
      backend.io.req.bits.addr := Mux(issueValid, mshrAddr(issueSel), 0.U)
      backend.io.req.bits.id := issueSel
      when(backend.io.req.fire) {
        mshrIssued(issueSel) := true.B
      }
      backend.io.resp.ready := true.B
      when(backend.io.resp.fire) {
        val respId = backend.io.resp.bits.id
        assert(mshrValid(respId), "TriangleMemMultiPort refill response hit invalid MSHR entry")
        mshrDone(respId) := true.B
        mshrData(respId) := backend.io.resp.bits.data
      }

      val refillFire = backend.io.req.fire
      val refillArbStall = issueValid && !backend.io.req.ready
      refillStatMon.io.clk := clock
      refillStatMon.io.reset := reset
      refillStatMon.io.busyCycle := issueValid
      refillStatMon.io.stallCycle := refillArbStall
      refillStatMon.io.refillFire := refillFire

      val targetedBank = Wire(Vec(numPorts, UInt(bankSelW.W)))
      val bankLocalAddr = Wire(Vec(numPorts, UInt(GlobalConfig.triMemAddrWidth.W)))
      for (p <- 0 until numPorts) {
        targetedBank(p) := (if (numBanks == 1) 0.U else io.req(p).bits.addr(bankSelW - 1, 0))
        bankLocalAddr(p) := (if (numBanks == 1) io.req(p).bits.addr else io.req(p).bits.addr >> bankSelW)
      }

      val reqReady = Wire(Vec(numPorts, Bool()))
      reqReady := VecInit(Seq.fill(numPorts)(false.B))

      for (b <- 0 until numBanks) {
        for (p <- 0 until numPorts) {
          val targetsBank = targetedBank(p) === b.U
          reqArbs(b).io.in(p).valid := io.req(p).valid && targetsBank
          reqArbs(b).io.in(p).bits.addr := bankLocalAddr(p)
          reqArbs(b).io.in(p).bits.mask := io.req(p).bits.mask
          reqArbs(b).io.in(p).bits.src := p.U
          reqArbs(b).io.in(p).bits.tag := io.req(p).bits.tag
          when(targetsBank) {
            reqReady(p) := reqArbs(b).io.in(p).ready && banks(b).io.req.ready
          }
        }
        banks(b).io.req.valid := reqArbs(b).io.out.valid
        banks(b).io.req.bits := reqArbs(b).io.out.bits
        reqArbs(b).io.out.ready := banks(b).io.req.ready
      }

      for (p <- 0 until numPorts) {
        io.req(p).ready := reqReady(p)
      }

      val respArbs = Seq.fill(numPorts)(Module(new RRArbiter(new TriMemResp(c, tagWidth), numBanks)))
      for (p <- 0 until numPorts) {
        for (b <- 0 until numBanks) {
          respArbs(p).io.in(b).valid := banks(b).io.resp.valid && banks(b).io.resp.bits.src === p.U
          respArbs(p).io.in(b).bits := banks(b).io.resp.bits.resp
        }
        io.resp(p) <> respArbs(p).io.out
      }

      for (b <- 0 until numBanks) {
        val respReadyVec = Wire(Vec(numPorts, Bool()))
        for (p <- 0 until numPorts) {
          respReadyVec(p) := respArbs(p).io.in(b).ready && banks(b).io.resp.bits.src === p.U
        }
        banks(b).io.resp.ready := respReadyVec.asUInt.orR
      }

    case _ =>
      val banks = Seq.tabulate(numBanks)(b => Module(new TriangleMemDPI(
        c,
        latency = GlobalConfig.triMemDpiLatency,
        bankId = b,
        numBanks = numBanks,
        maxEntries = GlobalConfig.triMemBankDepth
      )))
      val arbs = Seq.fill(numBanks)(Module(new RRArbiter(new BankReq, numPorts)))

      val targetedBank = Wire(Vec(numPorts, UInt(bankSelW.W)))
      val bankLocalAddr = Wire(Vec(numPorts, UInt(GlobalConfig.triMemAddrWidth.W)))
      for (p <- 0 until numPorts) {
        targetedBank(p) := (if (numBanks == 1) 0.U else io.req(p).bits.addr(bankSelW - 1, 0))
        bankLocalAddr(p) := (if (numBanks == 1) io.req(p).bits.addr else io.req(p).bits.addr >> bankSelW)
      }

      val reqReady = Wire(Vec(numPorts, Bool()))
      reqReady := VecInit(Seq.fill(numPorts)(false.B))
      val tagPipe = Seq.fill(numBanks)(RegInit(VecInit(Seq.fill(GlobalConfig.triMemDpiLatency)(0.U(tagWidth.W)))))

      for (b <- 0 until numBanks) {
        for (p <- 0 until numPorts) {
          val targetsBank = targetedBank(p) === b.U
          arbs(b).io.in(p).valid := io.req(p).valid && targetsBank
          arbs(b).io.in(p).bits.addr := bankLocalAddr(p)
          arbs(b).io.in(p).bits.mask := io.req(p).bits.mask
          arbs(b).io.in(p).bits.src := p.U
          arbs(b).io.in(p).bits.tag := io.req(p).bits.tag

          when(targetsBank) {
            reqReady(p) := arbs(b).io.in(p).ready && banks(b).io.req_ready
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
      }

      val respValid = Wire(Vec(numPorts, Bool()))
      val respBits = Wire(Vec(numPorts, new TriMemResp(c, tagWidth)))
      for (p <- 0 until numPorts) {
        respValid(p) := false.B
        respBits(p) := 0.U.asTypeOf(new TriMemResp(c, tagWidth))
      }

      for (b <- 0 until numBanks) {
        val srcPipe = Reg(Vec(GlobalConfig.triMemDpiLatency, UInt(srcW.W)))
        srcPipe(0) := Mux(arbs(b).io.out.fire, arbs(b).io.out.bits.src, 0.U)
        tagPipe(b)(0) := Mux(arbs(b).io.out.fire, arbs(b).io.out.bits.tag, 0.U)
        for (i <- 1 until GlobalConfig.triMemDpiLatency) {
          srcPipe(i) := srcPipe(i - 1)
          tagPipe(b)(i) := tagPipe(b)(i - 1)
        }

        val bankBlock = decodeBlock(banks(b).io.data, banks(b).io.addr_q, banks(b).io.valid_mask, b)
        when(banks(b).io.valid) {
          respValid(srcPipe.last) := true.B
          respBits(srcPipe.last).block := bankBlock
          respBits(srcPipe.last).tag := tagPipe(b).last
        }
      }

      for (p <- 0 until numPorts) {
        io.resp(p).valid := respValid(p)
        io.resp(p).bits := respBits(p)
      }
  }
}
