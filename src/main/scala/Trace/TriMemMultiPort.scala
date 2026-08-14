package Trace

import chisel3._
import chisel3.util._
import raytrace_utils._

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
  private val mshrIdW = math.max(1, log2Ceil(mshrEntries))

  class BankReq extends Bundle {
    val addr = UInt(GlobalConfig.triMemAddrWidth.W)
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
    val numBanksU = numBanks.U(GlobalConfig.triMemAddrWidth.W)
    val globalBlock = addrQ * numBanksU + bankId.U(GlobalConfig.triMemAddrWidth.W)
    val bankBase = globalBlock * c.cacheLineTriangles.U(GlobalConfig.triMemAddrWidth.W)
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
      block.tris(i).id := bankBase + i.U(GlobalConfig.triMemAddrWidth.W)
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
      val allocArb = Module(new RRArbiter(new TriMemAllocReq(c, numBanks, GlobalConfig.triMemCacheSets, mshrIdW), numBanks))
      val allocQ = Module(new Queue(new TriMemAllocReq(c, numBanks, GlobalConfig.triMemCacheSets, mshrIdW), 2))
      val allocInFlight = RegInit(VecInit(Seq.fill(numBanks)(false.B)))
      val backend = Module(new TriangleMemRefillBackend(c, numBanks, mshrIdW))
      val mshrValid = RegInit(VecInit(Seq.fill(mshrEntries)(false.B)))
      val mshrIssued = RegInit(VecInit(Seq.fill(mshrEntries)(false.B)))
      val mshrDone = RegInit(VecInit(Seq.fill(mshrEntries)(false.B)))
      val mshrDelivered = RegInit(VecInit(Seq.fill(mshrEntries)(false.B)))
      val mshrError = RegInit(VecInit(Seq.fill(mshrEntries)(false.B)))
      val mshrBank = Reg(Vec(mshrEntries, UInt(bankSelW.W)))
      val mshrAddr = Reg(Vec(mshrEntries, UInt(GlobalConfig.triMemAddrWidth.W)))
      // Keep the small number of completed DMA lines in registers.  A
      // synchronous refill RAM adds a second, implicit read-latency contract
      // between mshrDone and refillReadValid; under reordered status/data
      // completion that can expose the RAM's stale output to a bank refill.
      // Four 1152-bit lines are small enough to make the ownership explicit.
      val refillLines = Reg(Vec(mshrEntries,
        UInt((c.cacheLineTriangles * 9 * c.cfg.totalWidth).W)))
      val refillDispatchValid = RegInit(false.B)
      val refillDispatchId = Reg(UInt(mshrIdW.W))
      val refillDispatchBank = Reg(UInt(bankSelW.W))
      val refillDispatchBankOH = Reg(UInt(numBanks.W))
      val refillDispatchAddr = Reg(UInt(GlobalConfig.triMemAddrWidth.W))
      val refillDispatchSet = Reg(UInt(log2Ceil(GlobalConfig.triMemCacheSets).W))
      val refillDispatchTag = Reg(UInt((GlobalConfig.triMemAddrWidth - log2Ceil(GlobalConfig.triMemCacheSets)).W))
      val refillDispatchError = RegInit(false.B)
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
        allocArb.io.in(b).valid := banks(b).io.allocReq.valid && !allocInFlight(b)
        allocArb.io.in(b).bits := banks(b).io.allocReq.bits
        banks(b).io.allocReq.ready := allocArb.io.in(b).ready && !allocInFlight(b)

        banks(b).io.refillDone.valid := refillDispatchValid && refillDispatchBankOH(b)
        banks(b).io.refillDone.bits.id := refillDispatchId
        banks(b).io.refillDone.bits.addr := refillDispatchAddr
        banks(b).io.refillDone.bits.set := refillDispatchSet
        banks(b).io.refillDone.bits.tag := refillDispatchTag
        banks(b).io.refillDone.bits.data := Mux(refillDispatchError, 0.U, refillDispatchData)
      }

      allocQ.io.enq <> allocArb.io.out
      when(allocQ.io.enq.fire) {
        allocInFlight(allocArb.io.out.bits.bank) := true.B
      }

      val mergeHits = Wire(Vec(mshrEntries, Bool()))
      for (i <- 0 until mshrEntries) {
        mergeHits(i) := mshrValid(i) &&
          mshrBank(i) === allocQ.io.deq.bits.bank &&
          mshrAddr(i) === allocQ.io.deq.bits.addr
      }
      val allocSelOH = PriorityEncoderOH(mergeHits)
      val allocSel = OHToUInt(allocSelOH)
      val freeVec = VecInit((0 until mshrEntries).map(i => !mshrValid(i)))
      val freeOH = PriorityEncoderOH(freeVec)
      val freeSel = OHToUInt(freeOH)
      val freeExists = freeVec.asUInt.orR
      val mergeExists = mergeHits.asUInt.orR
      val allocCanServe = mergeExists || freeExists
      allocQ.io.deq.ready := false.B
      for (b <- 0 until numBanks) {
        val selected = allocQ.io.deq.valid && allocQ.io.deq.bits.bank === b.U
        val bankMerge = selected && mergeExists
        banks(b).io.allocResp.valid := bankMerge || (selected && freeExists)
        banks(b).io.allocResp.bits.id := Mux(mergeExists, allocSel, freeSel)
        banks(b).io.allocResp.bits.merged := mergeExists
        when(banks(b).io.allocResp.fire) {
          allocInFlight(b) := false.B
        }
        when(selected && banks(b).io.allocResp.ready && allocCanServe) {
          allocQ.io.deq.ready := true.B
        }
      }

      val allocFire = allocQ.io.deq.fire
      val allocId = Mux(mergeExists, allocSel, freeSel)
      val allocNew = allocFire && !mergeExists
      val allocBank = allocQ.io.deq.bits.bank
      val allocAddr = allocQ.io.deq.bits.addr
      val releaseForId = VecInit((0 until mshrEntries).map(i =>
        VecInit(banks.map(bank => bank.io.release.valid && bank.io.release.bits.id === i.U)).asUInt.orR
      ))
      for (b <- 0 until numBanks) {
        when(banks(b).io.release.valid) {
          assert(mshrValid(banks(b).io.release.bits.id),
            "TriangleMemMultiPort release hit invalid MSHR entry")
        }
        for (other <- b + 1 until numBanks) {
          assert(!(banks(b).io.release.valid && banks(other).io.release.valid &&
            banks(b).io.release.bits.id === banks(other).io.release.bits.id),
            "TriangleMemMultiPort received conflicting releases for one MSHR")
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
      backend.io.resp.ready := true.B
      when(backend.io.resp.fire) {
        val respId = backend.io.resp.bits.id
        assert(mshrValid(respId), "TriangleMemMultiPort refill response hit invalid MSHR entry")
      }

      when(backend.io.refillWrite.valid) {
        refillLines(backend.io.refillWrite.bits.id) := backend.io.refillWrite.bits.data
      }
      val refillReadyVec = VecInit((0 until mshrEntries).map(i =>
        mshrValid(i) && mshrDone(i) && !mshrDelivered(i)
      ))
      when(!refillDispatchValid && refillReadyVec.asUInt.orR) {
        val dispatchOH = PriorityEncoderOH(refillReadyVec)
        val dispatchId = OHToUInt(dispatchOH)
        val dispatchAddr = Mux1H(dispatchOH, mshrAddr)
        refillDispatchValid := true.B
        refillDispatchId := dispatchId
        refillDispatchBank := Mux1H(dispatchOH, mshrBank)
        refillDispatchBankOH := Mux1H(dispatchOH, mshrBank.map(UIntToOH(_, numBanks)))
        refillDispatchAddr := dispatchAddr
        refillDispatchSet := dispatchAddr(log2Ceil(GlobalConfig.triMemCacheSets) - 1, 0)
        refillDispatchTag := dispatchAddr(
          GlobalConfig.triMemAddrWidth - 1,
          log2Ceil(GlobalConfig.triMemCacheSets)
        )
        refillDispatchError := Mux1H(dispatchOH, mshrError)
        // The wide line is selected only after the small one-hot arbitration.
        refillDispatchData := Mux1H(dispatchOH, refillLines)
      }
      when(refillDispatchValid && VecInit(banks.map(_.io.refillDone.fire)).asUInt.orR) {
        refillDispatchValid := false.B
      }

      // All MSHR mutations are resolved per ID here.  A release has priority
      // over any impossible concurrent completion/delivery and assertions
      // below make such ownership violations visible in simulation.
      val dispatchFire = refillDispatchValid && VecInit(banks.map(_.io.refillDone.fire)).asUInt.orR
      for (i <- 0 until mshrEntries) {
        val isAlloc = allocNew && allocId === i.U
        val isIssue = backend.io.req.fire && issueSel === i.U
        val isResp = backend.io.resp.fire && backend.io.resp.bits.id === i.U
        val isDispatch = dispatchFire && refillDispatchId === i.U
        assert(!(releaseForId(i) && (isAlloc || isIssue || isResp || isDispatch)),
          "TriangleMemMultiPort received conflicting updates for one MSHR")
        when(releaseForId(i)) {
          mshrValid(i) := false.B
          mshrIssued(i) := false.B
          mshrDone(i) := false.B
          mshrDelivered(i) := false.B
          mshrError(i) := false.B
          mshrAddr(i) := 0.U
        }.elsewhen(isAlloc) {
          mshrValid(i) := true.B
          mshrIssued(i) := false.B
          mshrDone(i) := false.B
          mshrDelivered(i) := false.B
          mshrError(i) := false.B
          mshrBank(i) := allocBank
          mshrAddr(i) := allocAddr
        }.otherwise {
          when(isIssue) { mshrIssued(i) := true.B }
          when(isResp) {
            mshrDone(i) := true.B
            mshrError(i) := backend.io.resp.bits.error
          }
          when(isDispatch) { mshrDelivered(i) := true.B }
        }
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

      // Arbitrate only the small bank selector.  Carrying the complete
      // TriangleBlock through one RRArbiter per port replicated a very wide
      // mux tree (four ports x four banks).  The selected bank is now used to
      // perform one data mux after arbitration, keeping block data out of the
      // control decision path.
      val respArbs = Seq.fill(numPorts)(Module(new RRArbiter(UInt(bankSelW.W), numBanks)))
      val respSelected = Seq.fill(numPorts)(Wire(new TriMemResp(c, tagWidth)))
      for (p <- 0 until numPorts) {
        for (b <- 0 until numBanks) {
          respArbs(p).io.in(b).valid := banks(b).io.resp.valid && banks(b).io.resp.bits.src === p.U
          respArbs(p).io.in(b).bits := b.U
        }

        val selectedOH = UIntToOH(respArbs(p).io.out.bits, numBanks)
        respSelected(p) := Mux1H(
          selectedOH.asBools,
          banks.map(_.io.resp.bits.resp)
        )

        io.resp(p).valid := respArbs(p).io.out.valid
        io.resp(p).bits := respSelected(p)
        respArbs(p).io.out.ready := io.resp(p).ready
      }

      for (b <- 0 until numBanks) {
        val respReadyVec = Wire(Vec(numPorts, Bool()))
        for (p <- 0 until numPorts) {
          respReadyVec(p) := respArbs(p).io.out.valid &&
            respArbs(p).io.out.bits === b.U && io.resp(p).ready
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
