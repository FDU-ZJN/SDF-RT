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

class TriangleMemCachedBank(
  val c: TriPeConfig,
  val srcWidth: Int,
  val tagWidth: Int,
  val bankId: Int,
  val numBanks: Int = GlobalConfig.triMemNumBanks,
  val numSets: Int = GlobalConfig.triMemCacheSets,
  val ways: Int = GlobalConfig.triMemCacheWays,
  val reqQueueDepth: Int = GlobalConfig.triMemReqQueueDepth,
  val mergeQueueDepth: Int = GlobalConfig.triMemMergeQueueDepth
) extends Module {
  require(ways == 2, s"TriangleMemCachedBank currently supports exactly 2 ways, got $ways")
  require(isPow2(numSets), s"TriangleMemCachedBank requires power-of-two numSets, got $numSets")
  require(reqQueueDepth > 0, "TriangleMemCachedBank reqQueueDepth must be > 0")
  require(mergeQueueDepth > 0, "TriangleMemCachedBank mergeQueueDepth must be > 0")

  private val bitsPerTri = 3 * 3 * c.cfg.totalWidth
  private val totalBits = c.numPEs * bitsPerTri
  private val bankSelW = math.max(1, log2Ceil(numBanks))
  private val setIdxW = math.max(1, log2Ceil(numSets))
  private val tagW = GlobalConfig.triMemAddrWidth - setIdxW
  private val mergeIdxW = math.max(1, log2Ceil(mergeQueueDepth))
  private val mergeCountW = log2Ceil(mergeQueueDepth + 1)

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
    val missReq = Decoupled(new TriMemRefillReq(numBanks))
    val refillResp = Flipped(Decoupled(UInt(totalBits.W)))
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

  val cacheData = Seq.fill(ways)(SyncReadMem(numSets, UInt(totalBits.W)))
  val cacheValid = RegInit(VecInit(Seq.fill(ways)(VecInit(Seq.fill(numSets)(false.B)))))
  val cacheTag = RegInit(VecInit(Seq.fill(ways)(VecInit(Seq.fill(numSets)(0.U(tagW.W))))))
  val cacheLru = RegInit(VecInit(Seq.fill(numSets)(false.B)))

  val sIdle :: sHitRead :: sHitResp :: sMissReq :: sMissWait :: sEmitMissResp :: Nil = Enum(6)
  val state = RegInit(sIdle)

  val activeReq = Reg(new BankReq)
  val activeAddr = Reg(UInt(GlobalConfig.triMemAddrWidth.W))
  val activeMask = Reg(UInt(c.numPEs.W))
  val activeSrc = Reg(UInt(srcWidth.W))
  val activeTag = Reg(UInt(tagWidth.W))

  val hitWayReg = Reg(UInt(1.W))
  val hitSetReg = Reg(UInt(setIdxW.W))
  val hitMaskReg = Reg(UInt(c.numPEs.W))
  val hitAddrReg = Reg(UInt(GlobalConfig.triMemAddrWidth.W))
  val hitSrcReg = Reg(UInt(srcWidth.W))
  val hitTagReg = Reg(UInt(tagWidth.W))
  val hitDataReg = Reg(UInt(totalBits.W))

  val way0ReadEn = WireDefault(false.B)
  val way1ReadEn = WireDefault(false.B)
  val readSetIdx = WireDefault(0.U(setIdxW.W))
  val way0ReadData = cacheData(0).read(readSetIdx, way0ReadEn)
  val way1ReadData = cacheData(1).read(readSetIdx, way1ReadEn)

  val refillSetReg = Reg(UInt(setIdxW.W))
  val refillTagReg = Reg(UInt(tagW.W))
  val refillVictimWay = Reg(UInt(1.W))
  val refillDataReg = Reg(UInt(totalBits.W))

  val mergedReqs = Reg(Vec(mergeQueueDepth, new BankReq))
  val mergedCount = RegInit(0.U(mergeCountW.W))
  val emitIdx = RegInit(0.U(mergeCountW.W))
  val emitTotal = RegInit(0.U(mergeCountW.W))
  val missReqWaitCycles = RegInit(0.U(16.W))
  val missRespWaitCycles = RegInit(0.U(16.W))
  val statsValid = WireDefault(false.B)
  val statsHit = WireDefault(false.B)
  io.stat.valid := statsValid
  io.stat.bits := statsHit
  io.missReq.valid := false.B
  io.missReq.bits.bank := bankId.U
  io.missReq.bits.addr := activeAddr
  io.refillResp.ready := false.B

  when(state === sMissReq) {
    missReqWaitCycles := missReqWaitCycles + 1.U
  }.otherwise {
    missReqWaitCycles := 0.U
  }
  when(state === sMissWait) {
    missRespWaitCycles := missRespWaitCycles + 1.U
  }.otherwise {
    missRespWaitCycles := 0.U
  }

  assert(
    missReqWaitCycles < 4096.U,
    s"TriangleMemCachedBank[$bankId] miss request arbitration timeout"
  )
  assert(
    missRespWaitCycles < 4096.U,
    s"TriangleMemCachedBank[$bankId] refill response timeout"
  )

  val headReq = reqQ.io.deq.bits
  val headSet = setIdxOf(headReq.addr)
  val headTag = tagOf(headReq.addr)
  val headHitWay0 = cacheValid(0)(headSet) && cacheTag(0)(headSet) === headTag
  val headHitWay1 = cacheValid(1)(headSet) && cacheTag(1)(headSet) === headTag
  val headHit = headHitWay0 || headHitWay1
  val headMissSameLine = (state === sMissReq || state === sMissWait) && headReq.addr === activeAddr
  val canMergeHead = headMissSameLine && mergedCount =/= mergeQueueDepth.U
  val mergeHeadNow = reqQ.io.deq.valid && canMergeHead

  reqQ.io.deq.ready := false.B

  when(state === sIdle && reqQ.io.deq.valid) {
    reqQ.io.deq.ready := true.B
    statsValid := true.B
    statsHit := headHit
    activeReq := headReq
    activeAddr := headReq.addr
    activeMask := headReq.mask
    activeSrc := headReq.src
    activeTag := headReq.tag
    when(headHit) {
      hitWayReg := Mux(headHitWay0, 0.U, 1.U)
      hitSetReg := headSet
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
      refillSetReg := headSet
      refillTagReg := headTag
      refillVictimWay := Mux(!cacheValid(0)(headSet), 0.U, Mux(!cacheValid(1)(headSet), 1.U, cacheLru(headSet)))
      mergedCount := 0.U
      emitIdx := 0.U
      emitTotal := 0.U
      io.missReq.valid := true.B
      io.missReq.bits.addr := headReq.addr
      when(io.missReq.ready) {
        state := sMissWait
      }.otherwise {
        state := sMissReq
      }
    }
  }.elsewhen(state === sHitRead) {
    hitDataReg := Mux(hitWayReg === 0.U, way0ReadData, way1ReadData)
    state := sHitResp
  }.elsewhen(state === sMissReq) {
    io.missReq.valid := true.B
    when(mergeHeadNow) {
      reqQ.io.deq.ready := true.B
      statsValid := true.B
      statsHit := false.B
      mergedReqs(mergedCount(mergeIdxW - 1, 0)) := headReq
      mergedCount := mergedCount + 1.U
    }
    when(io.missReq.fire) {
      state := sMissWait
    }
  }.elsewhen(state === sMissWait) {
    io.refillResp.ready := true.B
    when(mergeHeadNow) {
      reqQ.io.deq.ready := true.B
      statsValid := true.B
      statsHit := false.B
      mergedReqs(mergedCount(mergeIdxW - 1, 0)) := headReq
      mergedCount := mergedCount + 1.U
    }

    when(io.refillResp.fire) {
      refillDataReg := io.refillResp.bits
      when(refillVictimWay === 0.U) {
        cacheData(0).write(refillSetReg, io.refillResp.bits)
      }.otherwise {
        cacheData(1).write(refillSetReg, io.refillResp.bits)
      }
      cacheValid(refillVictimWay)(refillSetReg) := true.B
      cacheTag(refillVictimWay)(refillSetReg) := refillTagReg
      cacheLru(refillSetReg) := !refillVictimWay
      emitIdx := 0.U
      emitTotal := mergedCount + 1.U + Mux(mergeHeadNow, 1.U, 0.U)
      state := sEmitMissResp
    }
  }

  val hitRespBlock = decodeBlock(hitDataReg, hitAddrReg, hitMaskReg)

  val emitReq = Wire(new BankReq)
  emitReq := activeReq
  when(emitIdx =/= 0.U) {
    emitReq := mergedReqs((emitIdx - 1.U)(mergeIdxW - 1, 0))
  }
  val missRespBlock = decodeBlock(refillDataReg, activeAddr, emitReq.mask)

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
  }.elsewhen(state === sEmitMissResp) {
    io.resp.valid := true.B
    io.resp.bits.src := emitReq.src
    io.resp.bits.resp.block := missRespBlock
    io.resp.bits.resp.tag := emitReq.tag
    when(io.resp.ready) {
      when(emitIdx + 1.U >= emitTotal) {
        state := sIdle
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
  val numBanks: Int = GlobalConfig.triMemNumBanks
) extends Module {
  require(numPorts > 0, "TriangleMemMultiPort needs at least one port")
  require(numBanks > 0, "TriangleMemMultiPort needs at least one bank")
  require(isPow2(numBanks), s"TriangleMemMultiPort currently requires numBanks to be power-of-two, got $numBanks")
  require(isPow2(c.numPEs), s"TriangleMemMultiPort requires numPEs to be power-of-two, got ${c.numPEs}")

  private val srcW = math.max(1, log2Ceil(numPorts))
  private val bankSelW = math.max(1, log2Ceil(numBanks))

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
    case 0 =>
      val banks = Seq.tabulate(numBanks)(b => Module(new TriangleMemCachedBank(c, srcW, tagWidth, b, numBanks)))
      val statMons = Seq.fill(numBanks)(Module(new TriCacheStatsMonitor))
      val refillStatMon = Module(new TriCacheRefillStatsMonitor)
      val reqArbs = Seq.fill(numBanks)(Module(new RRArbiter(new BankReq, numPorts)))
      val missArb = Module(new RRArbiter(new TriMemRefillReq(numBanks), numBanks))
      val sharedRefill = Module(new TriangleMemSharedDPI(c, latency = 1, numBanks = numBanks))
      val refillBusy = RegInit(false.B)
      val refillBankReg = Reg(UInt(bankSelW.W))
      val refillPendingValid = RegInit(VecInit(Seq.fill(numBanks)(false.B)))
      val refillPendingData = Reg(Vec(numBanks, UInt((c.numPEs * 9 * c.cfg.totalWidth).W)))

      for (b <- 0 until numBanks) {
        statMons(b).io.clk := clock
        statMons(b).io.reset := reset
        statMons(b).io.valid := banks(b).io.stat.valid
        statMons(b).io.hit := banks(b).io.stat.bits
        statMons(b).io.bank := b.U
        missArb.io.in(b) <> banks(b).io.missReq
        banks(b).io.refillResp.valid := refillPendingValid(b)
        banks(b).io.refillResp.bits := refillPendingData(b)
        when(banks(b).io.refillResp.fire) {
          refillPendingValid(b) := false.B
        }
      }

      val refillFire = missArb.io.out.valid && !refillBusy && sharedRefill.io.req_ready
      val refillArbStall = missArb.io.out.valid && refillBusy
      refillStatMon.io.clk := clock
      refillStatMon.io.reset := reset
      refillStatMon.io.busyCycle := refillBusy
      refillStatMon.io.stallCycle := refillArbStall
      refillStatMon.io.refillFire := refillFire

      sharedRefill.io.clk := clock
      sharedRefill.io.reset := reset
      sharedRefill.io.bank := missArb.io.out.bits.bank
      sharedRefill.io.addr := missArb.io.out.bits.addr
      sharedRefill.io.req_valid := missArb.io.out.valid && !refillBusy
      missArb.io.out.ready := !refillBusy && sharedRefill.io.req_ready

      when(refillFire) {
        refillBusy := true.B
        refillBankReg := missArb.io.out.bits.bank
      }
      when(sharedRefill.io.valid) {
        refillBusy := false.B
        refillPendingValid(refillBankReg) := true.B
        refillPendingData(refillBankReg) := sharedRefill.io.data
      }

      assert(
        !(sharedRefill.io.valid && refillPendingValid(refillBankReg)),
        "TriangleMemMultiPort refill response overwritten before bank consumed it"
      )

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
