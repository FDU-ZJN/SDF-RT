package Trace

import chisel3._
import chisel3.util._
import raytrace_utils._

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


class TriangleMemRefillBackend(
  val c: TriPeConfig,
  val numBanks: Int,
  val idWidth: Int
) extends Module {
  private val totalBits = c.cacheLineTriangles * 9 * c.cfg.totalWidth
  private val bankSelW = math.max(1, log2Ceil(numBanks))
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
      val issuedIds = Module(new Queue(UInt(idWidth.W), entries = idCount))

      val globalTriangle = (io.req.bits.addr << bankSelW) | io.req.bits.bank
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

