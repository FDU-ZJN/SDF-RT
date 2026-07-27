package Trace

import chisel3._
import chisel3.util._
import raytrace_utils._

class TriMemReq(val c: TriPeConfig, val tagWidth: Int) extends Bundle {
  val addr = UInt(GlobalConfig.triMemAddrWidth.W)
  val mask = UInt(c.cacheLineTriangles.W)
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

class TriCacheTagArrayBB(
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
    val rd_en = Input(Bool())
    val rd_addr = Input(UInt(addrWidth.W))
    val rd_data = Output(UInt(dataWidth.W))
    val wr_en = Input(Bool())
    val wr_addr = Input(UInt(addrWidth.W))
    val wr_data = Input(UInt(dataWidth.W))
  })
  addResource("/TriCacheTagArray.sv")
}

class TriCacheTagArray(
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
    val ram = Module(new TriCacheTagArrayBB(dataWidth, depth))
    ram.io.clk := clock
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
  val addr = UInt(GlobalConfig.triMemBankAddrWidthFor(numBanks).W)
}

class TriangleMemSharedDPI(
  val c: TriPeConfig,
  val latency: Int = 1,
  val numBanks: Int = GlobalConfig.triMemNumBanks,
  val maxEntries: Int = GlobalConfig.triMemBankDepth
) extends BlackBox with HasBlackBoxInline {
  private val totalBits = c.cacheLineTriangles * 9 * c.cfg.totalWidth
  private val bankAddrW = GlobalConfig.triMemBankAddrWidthFor(numBanks)

  val io = IO(new Bundle {
    val clk = Input(Clock())
    val reset = Input(Reset())
    val bank = Input(UInt(32.W))
    val addr = Input(UInt(bankAddrW.W))
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
       |  input [${bankAddrW - 1}:0] addr,
       |  input req_valid,
       |  output [${totalBits - 1}:0] data,
       |  output valid,
       |  output req_ready
       |);
       |  byte raw_buffer[${c.cacheLineTriangles * 3 * 3 * (c.cfg.totalWidth / 8)}];
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
       |        for (i = 0; i < ${c.cacheLineTriangles * 3 * 3 * (c.cfg.totalWidth / 8)}; i = i + 1) begin
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
  require(isPow2(c.cacheLineTriangles), s"TriangleMemWrapper requires a power-of-two cache line, got ${c.cacheLineTriangles}")

  val io = IO(new Bundle {
    val req      = Flipped(Decoupled(UInt(GlobalConfig.triMemAddrWidth.W)))
    val req_mask = Flipped(Decoupled(UInt(c.cacheLineTriangles.W)))
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
  for (i <- 0 until c.cacheLineTriangles) {
    val hi = bitsPerTri * (i + 1) - 1
    val lo = bitsPerTri * i
    val triBits = dpiMem.io.data(hi, lo)
    blockData.tris(i).v0.x := triBits(31, 0)
    blockData.tris(i).v0.y := triBits(63, 32)
    blockData.tris(i).v0.z := triBits(95, 64)
    blockData.tris(i).e1.x := triBits(127, 96)
    blockData.tris(i).e1.y := triBits(159, 128)
    blockData.tris(i).e1.z := triBits(191, 160)
    blockData.tris(i).e2.x := triBits(223, 192)
    blockData.tris(i).e2.y := triBits(255, 224)
    blockData.tris(i).e2.z := triBits(287, 256)
    blockData.tris(i).id := dpiMem.io.addr_q * c.cacheLineTriangles.U(GlobalConfig.triangleAddrWidth.W) +
      i.U(GlobalConfig.triangleAddrWidth.W)
    blockData.mask(i) := dpiMem.io.valid_mask(i)
  }

  io.resp.valid := dpiMem.io.valid
  io.resp.bits := blockData
}

class TriMemBackendReq(val c: TriPeConfig, val numBanks: Int, val idWidth: Int) extends Bundle {
  val bank = UInt(math.max(1, log2Ceil(numBanks)).W)
  val addr = UInt(GlobalConfig.triMemBankAddrWidthFor(numBanks).W)
  val id = UInt(idWidth.W)
}

class TriMemBackendResp(val c: TriPeConfig, val idWidth: Int) extends Bundle {
  val id = UInt(idWidth.W)
  val error = Bool()
}

class TriMemRefillWrite(val c: TriPeConfig, val idWidth: Int) extends Bundle {
  val id = UInt(idWidth.W)
  val data = UInt((c.cacheLineTriangles * 9 * c.cfg.totalWidth).W)
}

class TriMemDmaReadData(val dataWidth: Int) extends Bundle {
  val data = UInt(dataWidth.W)
  val last = Bool()
}

class TriMemAllocReq(
  val c: TriPeConfig,
  val numBanks: Int,
  val numSets: Int,
  val idWidth: Int
) extends Bundle {
  private val bankAddrW = GlobalConfig.triMemBankAddrWidthFor(numBanks)
  val bank = UInt(math.max(1, log2Ceil(numBanks)).W)
  val addr = UInt(bankAddrW.W)
  val set = UInt(math.max(1, log2Ceil(numSets)).W)
  val tag = UInt((bankAddrW - math.max(1, log2Ceil(numSets))).W)
  val victimWay = UInt(1.W)
}

class TriMemAllocResp(val idWidth: Int) extends Bundle {
  val id = UInt(idWidth.W)
  val merged = Bool()
}

class TriMemRefillDone(val c: TriPeConfig, val idWidth: Int) extends Bundle {
  val id = UInt(idWidth.W)
  val data = UInt((c.cacheLineTriangles * 9 * c.cfg.totalWidth).W)
}

class TriMemRefillLineArrayBB(val dataWidth: Int, val depth: Int) extends BlackBox(
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
    val wr_en = Input(Bool())
    val wr_addr = Input(UInt(addrWidth.W))
    val wr_data = Input(UInt(dataWidth.W))
    val rd_en = Input(Bool())
    val rd_addr = Input(UInt(addrWidth.W))
    val rd_data = Output(UInt(dataWidth.W))
  })
  addResource("/TriMemRefillLineArray.sv")
}

class TriMemRefillLineArray(val dataWidth: Int, val depth: Int, val useXpm: Boolean) extends Module {
  val addrWidth = math.max(1, log2Ceil(depth))
  val io = IO(new Bundle {
    val wr_en = Input(Bool())
    val wr_addr = Input(UInt(addrWidth.W))
    val wr_data = Input(UInt(dataWidth.W))
    val rd_en = Input(Bool())
    val rd_addr = Input(UInt(addrWidth.W))
    val rd_data = Output(UInt(dataWidth.W))
  })

  if (useXpm) {
    val ram = Module(new TriMemRefillLineArrayBB(dataWidth, depth))
    ram.io.clk := clock
    ram.io.wr_en := io.wr_en
    ram.io.wr_addr := io.wr_addr
    ram.io.wr_data := io.wr_data
    ram.io.rd_en := io.rd_en
    ram.io.rd_addr := io.rd_addr
    io.rd_data := ram.io.rd_data
  } else {
    val mem = SyncReadMem(depth, UInt(dataWidth.W))
    when(io.wr_en) { mem.write(io.wr_addr, io.wr_data) }
    io.rd_data := mem.read(io.rd_addr, io.rd_en)
  }
}

class TriMemRelease(val idWidth: Int) extends Bundle {
  val id = UInt(idWidth.W)
}

class TriangleMemRefillBackend(
  val c: TriPeConfig,
  val numBanks: Int,
  val idWidth: Int
) extends Module {
  private val totalBits = c.cacheLineTriangles * 9 * c.cfg.totalWidth
  private val bankSelW = math.max(1, log2Ceil(numBanks))
  private val bankBits = log2Ceil(numBanks)
  private val dmaAddrWidth = 40
  private val dmaDataWidth = 128
  private val dmaLineBits = {
    val raw = c.cacheLineTriangles * 9 * c.cfg.totalWidth
    ((raw + dmaDataWidth - 1) / dmaDataWidth) * dmaDataWidth
  }
  private val dmaTagWidth = 4

  val io = IO(new Bundle {
    val req = Flipped(Decoupled(new TriMemBackendReq(c, numBanks, idWidth)))
    val resp = Decoupled(new TriMemBackendResp(c, idWidth))
    val refillWrite = Valid(new TriMemRefillWrite(c, idWidth))
    val triangleBaseAddress = Input(UInt(64.W))
    val dmaCmd = Decoupled(UInt(80.W))
    val dmaData = Flipped(Decoupled(new TriMemDmaReadData(dmaDataWidth)))
    val dmaStatus = Flipped(Decoupled(UInt(8.W)))
    val dmaReadError = Output(Bool())
    val dmaMalformedLineCount = Output(UInt(32.W))
    val dmaStatusErrorCount = Output(UInt(32.W))
    val dmaTagMismatchCount = Output(UInt(32.W))
    val issuedCount = Output(UInt(32.W))
    val completedCount = Output(UInt(32.W))
    val outstandingCount = Output(UInt(8.W))
  })

  io.dmaCmd.valid := false.B
  io.dmaCmd.bits := 0.U
  io.dmaData.ready := false.B
  io.dmaStatus.ready := false.B
  io.refillWrite.valid := false.B
  io.refillWrite.bits := 0.U.asTypeOf(new TriMemRefillWrite(c, idWidth))
  io.dmaReadError := false.B
  io.dmaMalformedLineCount := 0.U
  io.dmaStatusErrorCount := 0.U
  io.dmaTagMismatchCount := 0.U
  io.issuedCount := 0.U
  io.completedCount := 0.U
  io.outstandingCount := 0.U

  GlobalConfig.memImplMode match {
    case 0 if !GlobalConfig.useTriangleDmaRefill =>
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
      io.resp.bits.error := false.B
      io.refillWrite.valid := validPipe && mem.io.valid
      io.refillWrite.bits.id := idPipe
      io.refillWrite.bits.data := mem.io.data

    case 0 | 2 =>
      val dmaBeats = dmaLineBits / dmaDataWidth
      require(totalBits <= dmaLineBits,
        s"DataMover refill line is $dmaLineBits bits, block is $totalBits bits")
      require(idWidth <= dmaTagWidth, s"DataMover tag is $dmaTagWidth bits, MSHR ID is $idWidth bits")

      val idCount = 1 << idWidth
      val beatCount = RegInit(0.U(log2Ceil(dmaBeats + 1).W))
      val lineBuffer = Reg(Vec(dmaBeats - 1, UInt(dmaDataWidth.W)))
      val idBusy = RegInit(VecInit(Seq.fill(idCount)(false.B)))
      val dataDone = RegInit(VecInit(Seq.fill(idCount)(false.B)))
      val statusDone = RegInit(VecInit(Seq.fill(idCount)(false.B)))
      val responseError = RegInit(VecInit(Seq.fill(idCount)(false.B)))
      val issuedCount = RegInit(0.U(32.W))
      val completedCount = RegInit(0.U(32.W))
      val errorSticky = RegInit(false.B)
      val malformedLineCount = RegInit(0.U(32.W))
      val statusErrorCount = RegInit(0.U(32.W))
      val tagMismatchCount = RegInit(0.U(32.W))
      val issuedIds = Module(new Queue(UInt(idWidth.W), entries = idCount, useSyncReadMem = true))

      val globalTriangle = if (numBanks == 1) io.req.bits.addr else Cat(io.req.bits.addr, io.req.bits.bank(bankBits - 1, 0))
      val blockBytes = dmaLineBits / 8
      val byteAddress = io.triangleBaseAddress + (globalTriangle * blockBytes.U)
      // PG022 Full command, 40-bit address: RSVD, TAG, SADDR, EOF/DSA/TYPE/BTT.
      val commandLow = Cat(0.U(1.W), 1.U(1.W), 0.U(6.W), 1.U(1.W), blockBytes.U(23.W))
      val commandTag = io.req.bits.id.pad(dmaTagWidth)

      io.dmaCmd.valid := io.req.valid && !idBusy(io.req.bits.id) && issuedIds.io.enq.ready
      io.dmaCmd.bits := Cat(0.U(4.W), commandTag, byteAddress(dmaAddrWidth - 1, 0), commandLow)
      io.req.ready := io.dmaCmd.ready && !idBusy(io.req.bits.id) && issuedIds.io.enq.ready
      issuedIds.io.enq.valid := io.req.fire
      issuedIds.io.enq.bits := io.req.bits.id

      when(io.req.fire) {
        val reqId = io.req.bits.id
        idBusy(reqId) := true.B
        dataDone(reqId) := false.B
        statusDone(reqId) := false.B
        responseError(reqId) := false.B
        issuedCount := issuedCount + 1.U
      }

      val activeId = issuedIds.io.deq.bits
      io.dmaData.ready := issuedIds.io.deq.valid && !dataDone(activeId)
      issuedIds.io.deq.ready := io.dmaData.fire && io.dmaData.bits.last
      when(io.dmaData.fire) {
        for (i <- 0 until dmaBeats - 1) {
          when(beatCount === i.U) {
            lineBuffer(i) := io.dmaData.bits.data
          }
        }
        val malformed = io.dmaData.bits.last =/= (beatCount === (dmaBeats - 1).U)
        when(malformed) {
          malformedLineCount := malformedLineCount + 1.U
          responseError(activeId) := true.B
          errorSticky := true.B
        }
        assert(!malformed, s"DataMover TLAST does not match $dmaBeats-beat triangle block")
        when(io.dmaData.bits.last) {
          val prevBeats = (0 until dmaBeats - 1).foldRight(io.dmaData.bits.data) { (i, acc) =>
            Cat(acc, lineBuffer(i))
          }
          val completedLine = prevBeats
          io.refillWrite.valid := true.B
          io.refillWrite.bits.id := activeId
          io.refillWrite.bits.data := completedLine(totalBits - 1, 0)
          dataDone(activeId) := true.B
          beatCount := 0.U
        }.otherwise {
          beatCount := beatCount + 1.U
        }
      }

      val statusTag = io.dmaStatus.bits(3, 0)
      val statusId = statusTag(idWidth - 1, 0)
      val statusTagInRange = if (idWidth == dmaTagWidth) true.B else statusTag(dmaTagWidth - 1, idWidth) === 0.U
      io.dmaStatus.ready := true.B
      when(io.dmaStatus.fire) {
        val tagMatches = statusTagInRange && idBusy(statusId) && !statusDone(statusId)
        val statusError = !io.dmaStatus.bits(7) || io.dmaStatus.bits(6, 4).orR
        when(tagMatches) {
          statusDone(statusId) := true.B
          when(statusError) {
            responseError(statusId) := true.B
            statusErrorCount := statusErrorCount + 1.U
            errorSticky := true.B
          }
        }.otherwise {
          tagMismatchCount := tagMismatchCount + 1.U
          errorSticky := true.B
        }
        assert(tagMatches, "DataMover status tag does not identify an outstanding MSHR")
      }

      val responseValid = VecInit((0 until idCount).map(i => idBusy(i) && dataDone(i) && statusDone(i)))
      val responseOH = PriorityEncoderOH(responseValid)
      val responseId = OHToUInt(responseOH)
      io.resp.valid := responseValid.asUInt.orR
      io.resp.bits.id := responseId
      io.resp.bits.error := responseError(responseId)
      when(io.resp.fire) {
        idBusy(responseId) := false.B
        dataDone(responseId) := false.B
        statusDone(responseId) := false.B
        completedCount := completedCount + 1.U
      }

      io.dmaReadError := errorSticky
      io.dmaMalformedLineCount := malformedLineCount
      io.dmaStatusErrorCount := statusErrorCount
      io.dmaTagMismatchCount := tagMismatchCount
      io.issuedCount := issuedCount
      io.completedCount := completedCount
      io.outstandingCount := PopCount(idBusy)
  }
}

class TriangleMemCachedBank(
  val c: TriPeConfig,
  val srcWidth: Int,
  val tagWidth: Int,
  val numBanks: Int = GlobalConfig.triMemNumBanks,
  val numSets: Int = GlobalConfig.triMemCacheSets,
  val ways: Int = GlobalConfig.triMemCacheWays,
  val reqQueueDepth: Int = GlobalConfig.triMemReqQueueDepth,
  val mergeQueueDepth: Int = GlobalConfig.triMemMergeQueueDepth,
  val mshrEntries: Int = GlobalConfig.triMemMshrEntries,
  val useXpmCacheOverride: Option[Boolean] = None
) extends Module {
  require(ways >= 1, s"TriangleMemCachedBank requires ways >= 1, got $ways")
  require(isPow2(numSets), s"TriangleMemCachedBank requires power-of-two numSets, got $numSets")
  require(reqQueueDepth > 0, "TriangleMemCachedBank reqQueueDepth must be > 0")
  require(mergeQueueDepth > 0, "TriangleMemCachedBank mergeQueueDepth must be > 0")
  require(mshrEntries > 0, "TriangleMemCachedBank mshrEntries must be > 0")

  private val bitsPerTri = 3 * 3 * c.cfg.totalWidth
  private val totalBits = c.cacheLineTriangles * bitsPerTri
  private val setIdxW = math.max(1, log2Ceil(numSets))
  private val bankBits = log2Ceil(numBanks)
  private val bankAddrW = GlobalConfig.triMemBankAddrWidthFor(numBanks)
  private val tagW = bankAddrW - setIdxW
  require(tagW > 0, s"TriangleMemCachedBank needs address bits above $numSets sets")
  private val mergeIdxW = math.max(1, log2Ceil(mergeQueueDepth))
  private val mergeCountW = log2Ceil(mergeQueueDepth + 1)
  private val mshrIdW = math.max(1, log2Ceil(mshrEntries))
  private val bankIdW = math.max(1, log2Ceil(numBanks))
  private val useXpmCache = useXpmCacheOverride.getOrElse(GlobalConfig.memImplMode == 2)

  class BankReq extends Bundle {
    val addr = UInt(bankAddrW.W)
    val mask = UInt(c.cacheLineTriangles.W)
    val src = UInt(srcWidth.W)
    val tag = UInt(tagWidth.W)
  }

  class BankResp extends Bundle {
    val src = UInt(srcWidth.W)
    val resp = new TriMemResp(c, tagWidth)
  }

  val io = IO(new Bundle {
    val bankId = Input(UInt(bankIdW.W))
    val req = Flipped(Decoupled(new BankReq))
    val resp = Decoupled(new BankResp)
    val stat = Valid(Bool())
    val allocReq = Decoupled(new TriMemAllocReq(c, numBanks, numSets, mshrIdW))
    val allocProbeId = Input(UInt(mshrIdW.W))
    val allocProbeReady = Output(Bool())
    val allocResp = Flipped(Decoupled(new TriMemAllocResp(mshrIdW)))
    val refillDone = Flipped(Decoupled(new TriMemRefillDone(c, mshrIdW)))
    val release = Decoupled(new TriMemRelease(mshrIdW))
  })

  private def setIdxOf(addr: UInt): UInt = {
    if (setIdxW == 0) 0.U else addr(setIdxW - 1, 0)
  }

  private def tagOf(addr: UInt): UInt = {
    addr(bankAddrW - 1, setIdxW)
  }

  private def decodeBlock(data: UInt, addrQ: UInt, mask: UInt): TriangleBlock = {
    val block = Wire(new TriangleBlock(c))
    val globalBlock = if (numBanks == 1) addrQ else Cat(addrQ, io.bankId(bankBits - 1, 0))
    val bankBase = globalBlock << log2Ceil(c.cacheLineTriangles)
    for (i <- 0 until c.cacheLineTriangles) {
      val hi = bitsPerTri * (i + 1) - 1
      val lo = bitsPerTri * i
      val triBits = data(hi, lo)
      block.tris(i).v0.x := triBits(31, 0)
      block.tris(i).v0.y := triBits(63, 32)
      block.tris(i).v0.z := triBits(95, 64)
      block.tris(i).e1.x := triBits(127, 96)
      block.tris(i).e1.y := triBits(159, 128)
      block.tris(i).e1.z := triBits(191, 160)
      block.tris(i).e2.x := triBits(223, 192)
      block.tris(i).e2.y := triBits(255, 224)
      block.tris(i).e2.z := triBits(287, 256)
      block.tris(i).id := bankBase + i.U(GlobalConfig.triangleAddrWidth.W)
      block.mask(i) := mask(i)
    }
    block
  }

  val reqQ = Module(new Queue(new BankReq, reqQueueDepth, useSyncReadMem = true))
  reqQ.io.enq <> io.req

  val cacheData = Seq.fill(ways)(Module(new TriCacheDataArray(totalBits, numSets, useXpmCache)))
  val cacheTag = Seq.fill(ways)(Module(new TriCacheTagArray(tagW, numSets, useXpmCache)))
  val cacheValid = RegInit(VecInit(Seq.fill(ways)(VecInit(Seq.fill(numSets)(false.B)))))
  val cacheLru = if (ways > 1) Some(RegInit(VecInit(Seq.fill(numSets)(false.B)))) else None

  val pendingReqs = Reg(Vec(mshrEntries, Vec(mergeQueueDepth, new BankReq)))
  val pendingCount = RegInit(VecInit(Seq.fill(mshrEntries)(0.U(mergeCountW.W))))
  val pendingAddr = Reg(Vec(mshrEntries, UInt(bankAddrW.W)))
  val pendingSet = Reg(Vec(mshrEntries, UInt(setIdxW.W)))
  val pendingTag = Reg(Vec(mshrEntries, UInt(tagW.W)))
  val pendingVictimWay = Reg(Vec(mshrEntries, UInt(math.max(1, log2Ceil(ways)).W)))

  val emitMshrId = Reg(UInt(mshrIdW.W))
  val emitIdx = RegInit(0.U(mergeCountW.W))
  val emitTotal = RegInit(0.U(mergeCountW.W))
  val emitDataReg = Reg(UInt(totalBits.W))
  val refillEmitValid = RegInit(false.B)
  // Refill payload and metadata are captured before they can drive either
  // cache RAM.  Commit waits until every lookup for the target set has left
  // the tag/data pipeline, so no generation table or read/write bypass is
  // needed to keep an old tag snapshot paired with old data.
  val refillCommitValid = RegInit(false.B)
  val refillCommitId = Reg(UInt(mshrIdW.W))
  val refillCommitSet = Reg(UInt(setIdxW.W))
  val refillCommitTag = Reg(UInt(tagW.W))
  val refillCommitWay = Reg(UInt(math.max(1, log2Ceil(ways)).W))
  val refillCommitData = Reg(UInt(totalBits.W))
  val releasePendingValid = RegInit(false.B)
  val releasePendingId = Reg(UInt(mshrIdW.W))

  private val dataReadLatency = GlobalConfig.triMemCacheHitLatency
  // The global response arbiter accepts one bank per cycle.  A pair of local
  // registers absorbs concurrent bank responses without implementing the wide
  // BankResp queue as LUTRAM, whose address and write-enable nets fan out over
  // every bit of the triangle block.
  private val respBufferDepth = 2
  val tagReqValid = RegInit(false.B)
  val tagReq = Reg(new BankReq)
  val tagReqWayValid = Reg(Vec(ways, Bool()))
  val tagPipeValid = RegInit(false.B)
  val tagPipeReq = Reg(new BankReq)
  val tagPipeData = Reg(Vec(ways, UInt(tagW.W)))
  val tagPipeWayValid = Reg(Vec(ways, Bool()))
  val dataPipeValid = RegInit(VecInit(Seq.fill(dataReadLatency)(false.B)))
  val dataPipeReq = Reg(Vec(dataReadLatency, new BankReq))
  val dataPipeWay = Reg(Vec(dataReadLatency, UInt(math.max(1, log2Ceil(ways)).W)))
  val missQ = Module(new Queue(new BankReq, reqQueueDepth, useSyncReadMem = true))
  val allocPendingValid = RegInit(false.B)
  val allocPendingReq = Reg(new BankReq)
  val allocPendingVictimWay = Reg(UInt(math.max(1, log2Ceil(ways)).W))
  val respHead = Reg(new BankResp)
  val respTail = Reg(new BankResp)
  val respCount = RegInit(0.U(log2Ceil(respBufferDepth + 1).W))
  val wayReadEn = Wire(Vec(ways, Bool()))
  val wayWriteEn = Wire(Vec(ways, Bool()))
  val tagReadEn = Wire(Vec(ways, Bool()))
  val tagWriteEn = Wire(Vec(ways, Bool()))
  wayReadEn := VecInit(Seq.fill(ways)(false.B))
  wayWriteEn := VecInit(Seq.fill(ways)(false.B))
  tagReadEn := VecInit(Seq.fill(ways)(false.B))
  tagWriteEn := VecInit(Seq.fill(ways)(false.B))
  val dataReadSetIdx = WireDefault(0.U(setIdxW.W))
  val tagReadSetIdx = WireDefault(0.U(setIdxW.W))
  val writeSetIdx = WireDefault(0.U(setIdxW.W))
  val writeData = WireDefault(0.U(totalBits.W))
  val tagWriteData = WireDefault(0.U(tagW.W))

  for (w <- 0 until ways) {
    cacheData(w).io.rd_en := wayReadEn(w)
    cacheData(w).io.rd_addr := dataReadSetIdx
    cacheData(w).io.wr_en := wayWriteEn(w)
    cacheData(w).io.wr_addr := writeSetIdx
    cacheData(w).io.wr_data := writeData
    cacheTag(w).io.rd_en := tagReadEn(w)
    cacheTag(w).io.rd_addr := tagReadSetIdx
    cacheTag(w).io.wr_en := tagWriteEn(w)
    cacheTag(w).io.wr_addr := writeSetIdx
    cacheTag(w).io.wr_data := tagWriteData
  }

  val wayReadData = VecInit(cacheData.map(_.io.rd_data))
  val wayTagData = VecInit(cacheTag.map(_.io.rd_data))
  val statsValid = WireDefault(false.B)
  val statsHit = WireDefault(false.B)
  io.stat.valid := statsValid
  io.stat.bits := statsHit
  io.allocReq.valid := false.B
  io.allocReq.bits.bank := io.bankId
  io.allocReq.bits.addr := missQ.io.deq.bits.addr
  io.allocReq.bits.set := setIdxOf(missQ.io.deq.bits.addr)
  io.allocReq.bits.tag := tagOf(missQ.io.deq.bits.addr)
  io.allocReq.bits.victimWay := 0.U
  io.allocProbeReady := allocPendingValid &&
    pendingCount(io.allocProbeId) =/= mergeQueueDepth.U
  io.allocResp.ready := false.B
  val allocRespFire = WireDefault(false.B)
  io.refillDone.ready := false.B
  io.release.valid := releasePendingValid
  io.release.bits.id := releasePendingId

  val headReq = reqQ.io.deq.bits
  val headSet = setIdxOf(headReq.addr)
  val lookupSet = setIdxOf(tagPipeReq.addr)
  val lookupTag = tagOf(tagPipeReq.addr)
  val lookupHitWayOH = VecInit((0 until ways).map(w =>
    tagPipeWayValid(w) &&
      tagPipeData(w) === lookupTag
  ))
  val lookupHit = lookupHitWayOH.asUInt.orR
  val lookupHitWayIdx = OHToUInt(lookupHitWayOH)
  private val victimWayForMiss =
    if (ways == 1) 0.U(math.max(1, log2Ceil(ways)).W)
    else Mux(!tagPipeWayValid(0), 0.U, Mux(!tagPipeWayValid(1), 1.U, cacheLru.get(lookupSet)))

  reqQ.io.deq.ready := false.B

  def enqueuePending(mshrId: UInt, req: BankReq, set: UInt, tag: UInt, victimWay: UInt): Unit = {
    val count = pendingCount(mshrId)
    assert(count =/= mergeQueueDepth.U, "TriangleMemCachedBank merge queue overflow")
    pendingReqs(mshrId)(count(mergeIdxW - 1, 0)) := req
    pendingCount(mshrId) := count + 1.U
    when(count === 0.U) {
      pendingAddr(mshrId) := req.addr
      pendingSet(mshrId) := set
      pendingTag(mshrId) := tag
      pendingVictimWay(mshrId) := victimWay
    }
  }

  missQ.io.enq.valid := false.B
  missQ.io.enq.bits := tagPipeReq
  missQ.io.deq.ready := false.B

  io.refillDone.ready := !refillCommitValid && !refillEmitValid && !releasePendingValid
  val refillCapture = io.refillDone.fire
  val incomingRefillSet = pendingSet(io.refillDone.bits.id)
  val blockedRefillSetValid = refillCommitValid || refillCapture
  val blockedRefillSet = Mux(refillCommitValid, refillCommitSet, incomingRefillSet)
  val headConflictsWithRefill = blockedRefillSetValid && headSet === blockedRefillSet
  val refillPipelineConflict =
    (tagReqValid && setIdxOf(tagReq.addr) === refillCommitSet) ||
      (tagPipeValid && lookupSet === refillCommitSet) ||
      VecInit((0 until dataReadLatency).map(i =>
        dataPipeValid(i) && setIdxOf(dataPipeReq(i).addr) === refillCommitSet
      )).asUInt.orR
  val refillCommit = refillCommitValid && !refillPipelineConflict
  val localMergeHits = Wire(Vec(mshrEntries, Bool()))
  for (i <- 0 until mshrEntries) {
    localMergeHits(i) := pendingCount(i) =/= 0.U &&
      pendingCount(i) =/= mergeQueueDepth.U &&
      pendingAddr(i) === tagPipeReq.addr
  }
  val localMergeExists = localMergeHits.asUInt.orR
  val localMergeId = OHToUInt(PriorityEncoderOH(localMergeHits))
  // Serialize the two bank-local pending-list writers. A simultaneous global
  // allocation response is already registered and can take this cycle; the
  // lookup miss remains losslessly buffered by missQ.
  val localMergeReady = localMergeExists && !allocRespFire
  val dataPipeBusy = PopCount(dataPipeValid)
  val dataCanIssue = !refillEmitValid &&
    (respCount + dataPipeBusy < respBufferDepth.U)
  val tagPipeCanAdvance = !tagPipeValid ||
    Mux(lookupHit, dataCanIssue, localMergeReady || missQ.io.enq.ready)
  // Reserve enough missQ slots for every lookup already beyond reqQ.  This
  // guarantees tagReq can drain before a conflicting refill commit even when
  // all global MSHRs are occupied by completed refills.
  val lookupMissSlotsUsed = missQ.io.count + tagReqValid.asUInt + tagPipeValid.asUInt
  val lookupCanLaunch = lookupMissSlotsUsed < reqQueueDepth.U
  val reqFire = reqQ.io.deq.valid && tagPipeCanAdvance && lookupCanLaunch &&
    !headConflictsWithRefill
  val dataReadFire = tagPipeValid && lookupHit && dataCanIssue
  val localMergeFire = tagPipeValid && !lookupHit && localMergeReady &&
    tagPipeCanAdvance

  reqQ.io.deq.ready := tagPipeCanAdvance && lookupCanLaunch && !headConflictsWithRefill
  missQ.io.enq.valid := tagPipeValid && !lookupHit && !localMergeReady

  when(tagPipeCanAdvance) {
    tagPipeValid := tagReqValid
    when(tagReqValid) {
      tagPipeReq := tagReq
      for (w <- 0 until ways) {
        tagPipeData(w) := wayTagData(w)
        tagPipeWayValid(w) := tagReqWayValid(w)
      }
    }
    tagReqValid := reqFire
    when(reqFire) {
      tagReq := headReq
      for (w <- 0 until ways) {
        tagReqWayValid(w) := cacheValid(w)(headSet)
      }
    }
  }

  for (i <- 1 until dataReadLatency) {
    dataPipeValid(i) := dataPipeValid(i - 1)
    dataPipeReq(i) := dataPipeReq(i - 1)
    dataPipeWay(i) := dataPipeWay(i - 1)
  }
  dataPipeValid(0) := dataReadFire
  when(dataReadFire) {
    dataPipeReq(0) := tagPipeReq
    dataPipeWay(0) := lookupHitWayIdx
  }

  val dataReturnValid = dataPipeValid.last
  val dataReturnWay = dataPipeWay.last
  val dataReturnBits = if (ways == 1) wayReadData(0) else Mux1H(UIntToOH(dataReturnWay, ways), wayReadData)
  val dataResp = Wire(new BankResp)
  dataResp := 0.U.asTypeOf(new BankResp)
  dataResp.src := dataPipeReq.last.src
  dataResp.resp.block := decodeBlock(dataReturnBits, dataPipeReq.last.addr, dataPipeReq.last.mask)
  dataResp.resp.tag := dataPipeReq.last.tag

  val emitReq = Wire(new BankReq)
  emitReq := pendingReqs(emitMshrId)(emitIdx(mergeIdxW - 1, 0))
  val refillResp = Wire(new BankResp)
  refillResp := 0.U.asTypeOf(new BankResp)
  refillResp.src := emitReq.src
  refillResp.resp.block := decodeBlock(emitDataReg, pendingAddr(emitMshrId), emitReq.mask)
  refillResp.resp.tag := emitReq.tag

  val respEnqValid = dataReturnValid || refillEmitValid
  val respEnqBits = Mux(dataReturnValid, dataResp, refillResp)
  val respDeqFire = respCount =/= 0.U && io.resp.ready
  val respEnqReady = respCount =/= respBufferDepth.U || respDeqFire
  io.resp.valid := respCount =/= 0.U
  io.resp.bits := respHead

  // Two-entry FIFO with simultaneous dequeue/enqueue support.  The response
  // payload stays in registers; only the bank-level arbiter sees valid/src.
  when(respEnqValid && respEnqReady) {
    when(respCount === 0.U) {
      respHead := respEnqBits
      respCount := 1.U
    }.elsewhen(respCount === 1.U) {
      when(respDeqFire) {
        respHead := respEnqBits
      }.otherwise {
        respTail := respEnqBits
        respCount := 2.U
      }
    }.otherwise {
      respHead := respTail
      respTail := respEnqBits
    }
  }.elsewhen(respDeqFire) {
    when(respCount === 2.U) {
      respHead := respTail
      respCount := 1.U
    }.otherwise {
      respCount := 0.U
    }
  }

  val refillEmitFire = refillEmitValid && !dataReturnValid && respEnqReady
  when(dataReturnValid) {
    assert(respEnqReady, "TriangleMemCachedBank response buffer overflow")
  }

  when(!reset.asBool) {
    when(io.release.fire) {
      releasePendingValid := false.B
    }
    for (w <- 0 until ways) {
      tagReadEn(w) := reqFire
    }
    tagReadSetIdx := headSet
    when(dataReadFire) {
      dataReadSetIdx := lookupSet
      for (w <- 0 until ways) {
        wayReadEn(w) := lookupHitWayOH(w)
      }
    }

    statsValid := tagPipeValid && tagPipeCanAdvance
    statsHit := lookupHit

    when(localMergeFire) {
      enqueuePending(
        localMergeId,
        tagPipeReq,
        lookupSet,
        lookupTag,
        pendingVictimWay(localMergeId)
      )
    }

    when(refillCapture) {
      val doneId = io.refillDone.bits.id
      assert(pendingCount(doneId) =/= 0.U, "TriangleMemCachedBank refill done with no pending waiters")
      refillCommitValid := true.B
      refillCommitId := doneId
      refillCommitSet := pendingSet(doneId)
      refillCommitTag := pendingTag(doneId)
      refillCommitWay := pendingVictimWay(doneId)
      refillCommitData := io.refillDone.bits.data
    }

    when(refillCommit) {
    val mergeOnRefillCommit = allocRespFire && io.allocResp.bits.merged &&
      io.allocResp.bits.id === refillCommitId
    val localMergeOnRefillCommit = localMergeFire && localMergeId === refillCommitId
    writeSetIdx := refillCommitSet
    writeData := refillCommitData
    emitMshrId := refillCommitId
    emitIdx := 0.U
    emitTotal := pendingCount(refillCommitId) + mergeOnRefillCommit.asUInt +
      localMergeOnRefillCommit.asUInt
    emitDataReg := refillCommitData
    for (w <- 0 until ways) {
      when(refillCommitWay === w.U) {
        wayWriteEn(w) := true.B
        tagWriteEn(w) := true.B
        cacheValid(w)(refillCommitSet) := true.B
      }
    }
    tagWriteData := refillCommitTag
    if (ways > 1) {
      cacheLru.get(refillCommitSet) := !refillCommitWay
    }
    refillCommitValid := false.B
    refillEmitValid := true.B
    }

    io.allocReq.valid := missQ.io.deq.valid && !allocPendingValid &&
      !refillEmitValid
    io.allocReq.bits.victimWay := victimWayForMiss
    missQ.io.deq.ready := io.allocReq.fire
    when(io.allocReq.fire) {
      allocPendingValid := true.B
      allocPendingReq := missQ.io.deq.bits
      allocPendingVictimWay := victimWayForMiss
    }

    io.allocResp.ready := allocPendingValid &&
      pendingCount(io.allocResp.bits.id) =/= mergeQueueDepth.U
    allocRespFire := io.allocResp.fire
    when(io.allocResp.fire) {
      allocPendingValid := false.B
      enqueuePending(
        io.allocResp.bits.id,
        allocPendingReq,
        setIdxOf(allocPendingReq.addr),
        tagOf(allocPendingReq.addr),
        allocPendingVictimWay
      )
    }

    val extendsActiveRefill = refillEmitValid &&
      ((io.allocResp.fire && io.allocResp.bits.merged &&
        io.allocResp.bits.id === emitMshrId) ||
        (localMergeFire && localMergeId === emitMshrId))
    when(extendsActiveRefill) {
      // emitTotal snapshots pendingCount when refill data arrives.  Registered
      // merge decisions may append another waiter anywhere in the subsequent
      // emit window, so extend that snapshot as soon as the waiter is accepted.
      emitTotal := emitTotal + 1.U
    }

    when(refillEmitFire) {
      when(emitIdx + 1.U >= emitTotal) {
        when(extendsActiveRefill) {
          // The global MSHR decision is registered.  A merge can therefore
          // arrive on the cycle that the previous last waiter is emitted.
          // Keep ownership of the completed line and emit the newly appended
          // waiter instead of releasing the MSHR and clearing pendingCount.
          emitIdx := emitIdx + 1.U
        }.otherwise {
          pendingCount(emitMshrId) := 0.U
          assert(!releasePendingValid, "TriangleMemCachedBank release queue overflow")
          releasePendingValid := true.B
          releasePendingId := emitMshrId
          refillEmitValid := false.B
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
  require(isPow2(c.cacheLineTriangles), s"TriangleMemMultiPort requires a power-of-two cache line, got ${c.cacheLineTriangles}")

  private val srcW = math.max(1, log2Ceil(numPorts))
  private val bankSelW = math.max(1, log2Ceil(numBanks))
  private val bankBits = log2Ceil(numBanks)
  private val bankAddrW = GlobalConfig.triMemBankAddrWidthFor(numBanks)
  private val mshrIdW = math.max(1, log2Ceil(mshrEntries))

  class BankReq extends Bundle {
    val addr = UInt(bankAddrW.W)
    val mask = UInt(c.cacheLineTriangles.W)
    val src = UInt(srcW.W)
    val tag = UInt(tagWidth.W)
  }

  val io = IO(new Bundle {
    val req = Vec(numPorts, Flipped(Decoupled(new TriMemReq(c, tagWidth))))
    val resp = Vec(numPorts, Decoupled(new TriMemResp(c, tagWidth)))
    val triangleBaseAddress = Input(UInt(64.W))
    val dmaCmd = Decoupled(UInt(80.W))
    val dmaData = Flipped(Decoupled(new TriMemDmaReadData(128)))
    val dmaStatus = Flipped(Decoupled(UInt(8.W)))
    val dmaReadError = Output(Bool())
    val dmaMalformedLineCount = Output(UInt(32.W))
    val dmaStatusErrorCount = Output(UInt(32.W))
    val dmaTagMismatchCount = Output(UInt(32.W))
    val axiIssuedCount = Output(UInt(32.W))
    val axiCompletedCount = Output(UInt(32.W))
    val axiOutstandingCount = Output(UInt(8.W))
  })

  io.dmaCmd.valid := false.B
  io.dmaCmd.bits := 0.U
  io.dmaData.ready := false.B
  io.dmaStatus.ready := false.B
  io.dmaReadError := false.B
  io.dmaMalformedLineCount := 0.U
  io.dmaStatusErrorCount := 0.U
  io.dmaTagMismatchCount := 0.U
  io.axiIssuedCount := 0.U
  io.axiCompletedCount := 0.U
  io.axiOutstandingCount := 0.U

  private def decodeBlock(data: UInt, addrQ: UInt, mask: UInt, bankId: Int): TriangleBlock = {
    val block = Wire(new TriangleBlock(c))
    val bitsPerTri = 3 * 3 * c.cfg.totalWidth
    val globalBlock = if (numBanks == 1) addrQ else Cat(addrQ, bankId.U(bankBits.W))
    val bankBase = globalBlock << log2Ceil(c.cacheLineTriangles)
    for (i <- 0 until c.cacheLineTriangles) {
      val hi = bitsPerTri * (i + 1) - 1
      val lo = bitsPerTri * i
      val triBits = data(hi, lo)
      block.tris(i).v0.x := triBits(31, 0)
      block.tris(i).v0.y := triBits(63, 32)
      block.tris(i).v0.z := triBits(95, 64)
      block.tris(i).e1.x := triBits(127, 96)
      block.tris(i).e1.y := triBits(159, 128)
      block.tris(i).e1.z := triBits(191, 160)
      block.tris(i).e2.x := triBits(223, 192)
      block.tris(i).e2.y := triBits(255, 224)
      block.tris(i).e2.z := triBits(287, 256)
      block.tris(i).id := bankBase + i.U(GlobalConfig.triangleAddrWidth.W)
      block.mask(i) := mask(i)
    }
    block
  }

  GlobalConfig.memImplMode match {
    case 0 | 2 =>
      val banks = Seq.fill(numBanks)(Module(
        new TriangleMemCachedBank(c, srcW, tagWidth, numBanks, mshrEntries = mshrEntries)
      ))
      val statMons = Seq.fill(numBanks)(Module(new TriCacheStatsMonitor))
      val refillStatMon = Module(new TriCacheRefillStatsMonitor)
      val reqArbs = Seq.fill(numBanks)(Module(new RRArbiter(new BankReq, numPorts)))
      val allocBankArb = Module(new RRArbiter(UInt(bankSelW.W), numBanks))
      val releaseArb = Module(new RRArbiter(UInt(mshrIdW.W), numBanks))
      val backend = Module(new TriangleMemRefillBackend(c, numBanks, mshrIdW))
      val mshrValid = RegInit(VecInit(Seq.fill(mshrEntries)(false.B)))
      val mshrIssued = RegInit(VecInit(Seq.fill(mshrEntries)(false.B)))
      val mshrDone = RegInit(VecInit(Seq.fill(mshrEntries)(false.B)))
      val mshrDelivered = RegInit(VecInit(Seq.fill(mshrEntries)(false.B)))
      val mshrError = RegInit(VecInit(Seq.fill(mshrEntries)(false.B)))
      val mshrBank = Reg(Vec(mshrEntries, UInt(bankSelW.W)))
      val mshrAddr = Reg(Vec(mshrEntries, UInt(bankAddrW.W)))
      val allocReqValid = RegInit(false.B)
      val allocReqBank = Reg(UInt(bankSelW.W))
      val allocReqAddr = Reg(UInt(bankAddrW.W))
      val allocDecisionValid = RegInit(false.B)
      val allocDecisionBank = Reg(UInt(bankSelW.W))
      val allocDecisionAddr = Reg(UInt(bankAddrW.W))
      val allocDecisionId = Reg(UInt(mshrIdW.W))
      val allocDecisionMerged = Reg(Bool())
      // Keep the small number of completed DMA lines in registers.  A
      // synchronous refill RAM adds a second, implicit read-latency contract
      // between mshrDone and refillReadValid; under reordered status/data
      // completion that can expose the RAM's stale output to a bank refill.
      // Four 1152-bit lines are small enough to make the ownership explicit.
      val refillLines = Reg(Vec(mshrEntries,
        UInt((c.cacheLineTriangles * 9 * c.cfg.totalWidth).W)))
      // Register the selected completion before it fans into the banks. This
      // keeps mshrValid/mshrBank scans out of the bank tag-RAM enable path;
      // refill delivery is off the hit path and can absorb one cycle.
      val refillDispatchValid = RegInit(false.B)
      val refillDispatchId = Reg(UInt(mshrIdW.W))
      val refillDispatchBank = Reg(UInt(bankSelW.W))
      val refillDispatchData = Reg(UInt((c.cacheLineTriangles * 9 * c.cfg.totalWidth).W))

      backend.io.triangleBaseAddress := io.triangleBaseAddress
      io.dmaCmd <> backend.io.dmaCmd
      backend.io.dmaData <> io.dmaData
      backend.io.dmaStatus <> io.dmaStatus
      io.dmaReadError := backend.io.dmaReadError
      io.dmaMalformedLineCount := backend.io.dmaMalformedLineCount
      io.dmaStatusErrorCount := backend.io.dmaStatusErrorCount
      io.dmaTagMismatchCount := backend.io.dmaTagMismatchCount
      io.axiIssuedCount := backend.io.issuedCount
      io.axiCompletedCount := backend.io.completedCount
      io.axiOutstandingCount := backend.io.outstandingCount

      for (b <- 0 until numBanks) {
        banks(b).io.bankId := b.U
        statMons(b).io.clk := clock
        statMons(b).io.reset := reset
        statMons(b).io.valid := banks(b).io.stat.valid
        statMons(b).io.hit := banks(b).io.stat.bits
        statMons(b).io.bank := b.U
        allocBankArb.io.in(b).valid := banks(b).io.allocReq.valid
        allocBankArb.io.in(b).bits := b.U
        banks(b).io.allocReq.ready := allocBankArb.io.in(b).ready
        releaseArb.io.in(b).valid := banks(b).io.release.valid
        releaseArb.io.in(b).bits := banks(b).io.release.bits.id
        banks(b).io.release.ready := releaseArb.io.in(b).ready

        banks(b).io.refillDone.valid := refillDispatchValid &&
          refillDispatchBank === b.U
        banks(b).io.refillDone.bits.id := refillDispatchId
        banks(b).io.refillDone.bits.data := refillDispatchData
        when(banks(b).io.refillDone.fire) {
          mshrDelivered(refillDispatchId) := true.B
        }

      }

      releaseArb.io.out.ready := true.B
      when(releaseArb.io.out.fire) {
        val relId = releaseArb.io.out.bits
        mshrValid(relId) := false.B
        mshrIssued(relId) := false.B
        mshrDone(relId) := false.B
        mshrDelivered(relId) := false.B
        mshrError(relId) := false.B
        mshrAddr(relId) := 0.U
      }

      // Arbitrate only the bank selector.  The selected address is muxed once
      // and registered before it reaches any MSHR comparison logic.
      allocBankArb.io.out.ready := !allocReqValid
      val selectedAllocBankOH = UIntToOH(allocBankArb.io.out.bits, numBanks)
      val selectedAllocAddr = Mux1H(
        selectedAllocBankOH.asBools,
        banks.map(_.io.allocReq.bits.addr)
      )
      when(allocBankArb.io.out.fire) {
        allocReqValid := true.B
        allocReqBank := allocBankArb.io.out.bits
        allocReqAddr := selectedAllocAddr
      }

      val mergeHits = Wire(Vec(mshrEntries, Bool()))
      for (i <- 0 until mshrEntries) {
        mergeHits(i) := mshrValid(i) &&
          mshrBank(i) === allocReqBank &&
          mshrAddr(i) === allocReqAddr
      }
      val freeVec = VecInit((0 until mshrEntries).map(i => !mshrValid(i)))
      val freeOH = PriorityEncoderOH(freeVec)
      val freeExists = freeVec.asUInt.orR
      val freeSel = OHToUInt(freeOH)
      val mergeExists = mergeHits.asUInt.orR
      val mergeSel = OHToUInt(PriorityEncoderOH(mergeHits))
      val allocCandidateId = Mux(mergeExists, mergeSel, freeSel)
      val allocCandidateBankOH = UIntToOH(allocReqBank, numBanks)
      for (b <- 0 until numBanks) {
        val responseTargetsBank = allocDecisionValid && allocDecisionBank === b.U
        banks(b).io.allocResp.valid := responseTargetsBank
        banks(b).io.allocResp.bits.id := allocDecisionId
        banks(b).io.allocResp.bits.merged := allocDecisionMerged
        banks(b).io.allocProbeId := allocCandidateId
      }
      val allocCandidateReady = Mux1H(
        allocCandidateBankOH.asBools,
        banks.map(_.io.allocProbeReady)
      )
      val allocCanServe = mergeExists || freeExists
      val allocDecisionCapture = allocReqValid && !allocDecisionValid &&
        allocCanServe && allocCandidateReady && !releaseArb.io.out.valid
      when(allocDecisionCapture) {
        allocReqValid := false.B
        allocDecisionValid := true.B
        allocDecisionBank := allocReqBank
        allocDecisionAddr := allocReqAddr
        allocDecisionId := allocCandidateId
        allocDecisionMerged := mergeExists
      }

      val allocDecisionFire = VecInit(banks.map(_.io.allocResp.fire)).asUInt.orR
      when(allocDecisionFire) {
        allocDecisionValid := false.B
        when(!allocDecisionMerged) {
          mshrValid(allocDecisionId) := true.B
          mshrIssued(allocDecisionId) := false.B
          mshrDone(allocDecisionId) := false.B
          mshrDelivered(allocDecisionId) := false.B
          mshrError(allocDecisionId) := false.B
          mshrBank(allocDecisionId) := allocDecisionBank
          mshrAddr(allocDecisionId) := allocDecisionAddr
        }
      }

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
        mshrError(respId) := backend.io.resp.bits.error
      }

      when(backend.io.refillWrite.valid) {
        refillLines(backend.io.refillWrite.bits.id) := backend.io.refillWrite.bits.data
      }
      val refillReadyVec = VecInit((0 until mshrEntries).map(i =>
        mshrValid(i) && mshrDone(i) && !mshrDelivered(i)
      ))
      val refillDispatchFire = VecInit(banks.map(_.io.refillDone.fire)).asUInt.orR
      when(refillDispatchFire) {
        refillDispatchValid := false.B
      }
      when(!refillDispatchValid && refillReadyVec.asUInt.orR) {
        val dispatchOH = PriorityEncoderOH(refillReadyVec)
        val dispatchId = OHToUInt(dispatchOH)
        refillDispatchValid := true.B
        refillDispatchId := dispatchId
        refillDispatchBank := mshrBank(dispatchId)
        refillDispatchData := Mux(mshrError(dispatchId), 0.U, refillLines(dispatchId))
      }

      val refillFire = backend.io.req.fire
      val refillArbStall = issueValid && !backend.io.req.ready
      refillStatMon.io.clk := clock
      refillStatMon.io.reset := reset
      refillStatMon.io.busyCycle := issueValid
      refillStatMon.io.stallCycle := refillArbStall
      refillStatMon.io.refillFire := refillFire

      val targetedBank = Wire(Vec(numPorts, UInt(bankSelW.W)))
      val bankLocalAddr = Wire(Vec(numPorts, UInt(bankAddrW.W)))
      for (p <- 0 until numPorts) {
        targetedBank(p) := (if (numBanks == 1) 0.U else io.req(p).bits.addr(bankSelW - 1, 0))
        bankLocalAddr(p) := (if (numBanks == 1) io.req(p).bits.addr else io.req(p).bits.addr(GlobalConfig.triMemAddrWidth - 1, bankBits))
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

      // Select one bank with a narrow arbiter, then select its wide
      // TriangleBlock once.  Per-port arbiters replicated a four-way wide
      // response mux for every destination port, consuming several thousand
      // LUTs and making all cache response data high-fanout.
      val respArb = Module(new RRArbiter(UInt(bankSelW.W), numBanks))
      for (b <- 0 until numBanks) {
        respArb.io.in(b).valid := banks(b).io.resp.valid
        respArb.io.in(b).bits := b.U
        banks(b).io.resp.ready := respArb.io.in(b).ready
      }

      val selectedBankOH = UIntToOH(respArb.io.out.bits, numBanks)
      val selectedResp = Mux1H(selectedBankOH.asBools, banks.map(_.io.resp.bits.resp))
      val selectedSrc = Mux1H(selectedBankOH.asBools, banks.map(_.io.resp.bits.src))
      val selectedPortReady = Mux1H(UIntToOH(selectedSrc, numPorts).asBools, io.resp.map(_.ready))
      respArb.io.out.ready := selectedPortReady

      for (p <- 0 until numPorts) {
        io.resp(p).valid := respArb.io.out.valid && selectedSrc === p.U
        io.resp(p).bits := selectedResp
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
      val bankLocalAddr = Wire(Vec(numPorts, UInt(bankAddrW.W)))
      for (p <- 0 until numPorts) {
        targetedBank(p) := (if (numBanks == 1) 0.U else io.req(p).bits.addr(bankSelW - 1, 0))
        bankLocalAddr(p) := (if (numBanks == 1) io.req(p).bits.addr else io.req(p).bits.addr(GlobalConfig.triMemAddrWidth - 1, bankBits))
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
