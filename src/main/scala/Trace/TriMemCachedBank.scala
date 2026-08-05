package Trace

import chisel3._
import chisel3.util._
import raytrace_utils._

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
  private val tagW = GlobalConfig.triMemAddrWidth - setIdxW
  private val mergeIdxW = math.max(1, log2Ceil(mergeQueueDepth))
  private val mergeCountW = log2Ceil(mergeQueueDepth + 1)
  private val mshrIdW = math.max(1, log2Ceil(mshrEntries))
  private val bankIdW = math.max(1, log2Ceil(numBanks))
  private val useXpmCache = useXpmCacheOverride.getOrElse(GlobalConfig.memImplMode == 2)

  class BankReq extends Bundle {
    val addr = UInt(GlobalConfig.triMemAddrWidth.W)
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
    val globalBlock = addrQ * numBanksU + io.bankId
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

  val reqQ = Module(new Queue(new BankReq, reqQueueDepth, useSyncReadMem = true))
  reqQ.io.enq <> io.req

  val cacheData = Seq.fill(ways)(Module(new TriCacheDataArray(totalBits, numSets, useXpmCache)))
  val cacheTag = Seq.fill(ways)(Module(new TriCacheTagArray(tagW, numSets, useXpmCache)))
  // Version each cache line so a tag snapshot can never be paired with data
  // installed by a later refill of the same direct-mapped set.  This is the
  // metadata hazard check; no second full tag shadow array is required.
  val cacheMeta = RegInit(VecInit(Seq.fill(ways)(VecInit(Seq.fill(numSets)(
    0.U.asTypeOf(new TriCacheLineMeta)
  )))))
  val cacheLru = if (ways > 1) Some(RegInit(VecInit(Seq.fill(numSets)(false.B)))) else None

  val pendingReqs = Reg(Vec(mshrEntries, Vec(mergeQueueDepth, new BankReq)))
  val pendingCount = RegInit(VecInit(Seq.fill(mshrEntries)(0.U(mergeCountW.W))))
  val pendingAddr = Reg(Vec(mshrEntries, UInt(GlobalConfig.triMemAddrWidth.W)))
  val pendingSet = Reg(Vec(mshrEntries, UInt(setIdxW.W)))
  val pendingTag = Reg(Vec(mshrEntries, UInt(tagW.W)))
  val pendingVictimWay = Reg(Vec(mshrEntries, UInt(math.max(1, log2Ceil(ways)).W)))

  val emitMshrId = Reg(UInt(mshrIdW.W))
  val emitIdx = RegInit(0.U(mergeCountW.W))
  val emitTotal = RegInit(0.U(mergeCountW.W))
  val emitDataReg = Reg(UInt(totalBits.W))
  val refillEmitValid = RegInit(false.B)
  val refillTagBypassValid = RegInit(false.B)
  val refillTagBypassSet = Reg(UInt(setIdxW.W))
  val refillTagBypassWay = Reg(UInt(math.max(1, log2Ceil(ways)).W))
  val refillTagBypassData = Reg(UInt(tagW.W))

  private val dataReadLatency = GlobalConfig.triMemCacheHitLatency
  private val respQueueDepth = 16
  val tagReqValid = RegInit(false.B)
  val tagReq = Reg(new BankReq)
  val tagReqMeta = Reg(Vec(ways, new TriCacheLineMeta))
  val tagPipeValid = RegInit(false.B)
  val tagPipeReq = Reg(new BankReq)
  val tagPipeData = Reg(Vec(ways, UInt(tagW.W)))
  val tagPipeMeta = Reg(Vec(ways, new TriCacheLineMeta))
  val dataPipeValid = RegInit(VecInit(Seq.fill(dataReadLatency)(false.B)))
  val dataPipeReq = Reg(Vec(dataReadLatency, new BankReq))
  val dataPipeWay = Reg(Vec(dataReadLatency, UInt(math.max(1, log2Ceil(ways)).W)))
  val missQ = Module(new Queue(new BankReq, reqQueueDepth, useSyncReadMem = true))
  val respQ = Module(new Queue(new BankResp, respQueueDepth, useSyncReadMem = true))
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
  io.allocResp.ready := false.B
  io.refillDone.ready := false.B
  io.release.valid := false.B
  io.release.bits.id := emitMshrId

  val headReq = reqQ.io.deq.bits
  val headSet = setIdxOf(headReq.addr)
  val lookupSet = setIdxOf(tagPipeReq.addr)
  val lookupTag = tagOf(tagPipeReq.addr)
  val lookupHitWayOH = VecInit((0 until ways).map(w =>
    tagPipeMeta(w).valid &&
      tagPipeData(w) === lookupTag &&
      tagPipeMeta(w).generation === cacheMeta(w)(lookupSet).generation
  ))
  val lookupHit = lookupHitWayOH.asUInt.orR
  val lookupHitWayIdx = OHToUInt(lookupHitWayOH)
  private val victimWayForMiss =
    if (ways == 1) 0.U(math.max(1, log2Ceil(ways)).W)
    else Mux(!tagPipeMeta(0).valid, 0.U, Mux(!tagPipeMeta(1).valid, 1.U, cacheLru.get(lookupSet)))

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

  val refillAccept = io.refillDone.valid && !refillEmitValid
  val dataPipeBusy = PopCount(dataPipeValid)
  val dataCanIssue = !refillAccept && !refillEmitValid &&
    (respQ.io.count + dataPipeBusy < respQueueDepth.U)
  val tagPipeCanAdvance = !tagPipeValid || Mux(lookupHit, dataCanIssue, missQ.io.enq.ready)
  val reqFire = reqQ.io.deq.valid && tagPipeCanAdvance && !refillAccept
  val dataReadFire = tagPipeValid && lookupHit && dataCanIssue
  val missEnqFire = tagPipeValid && !lookupHit && missQ.io.enq.ready

  reqQ.io.deq.ready := tagPipeCanAdvance && !refillAccept
  missQ.io.enq.valid := tagPipeValid && !lookupHit

  when(tagPipeCanAdvance) {
    tagPipeValid := tagReqValid
    when(tagReqValid) {
      tagPipeReq := tagReq
      for (w <- 0 until ways) {
        val priorRefillBypass = refillTagBypassValid &&
          refillTagBypassSet === setIdxOf(tagReq.addr) &&
          refillTagBypassWay === w.U
        // A tag SRAM read launched before this refill still returns the old
        // tag on the refill cycle.  Bypass the tag being written now as well
        // as the registered one-cycle refill bypass.
        val currentRefillBypass = refillAccept &&
          pendingSet(io.refillDone.bits.id) === setIdxOf(tagReq.addr) &&
          pendingVictimWay(io.refillDone.bits.id) === w.U
        tagPipeData(w) := Mux(currentRefillBypass, pendingTag(io.refillDone.bits.id),
          Mux(priorRefillBypass, refillTagBypassData, wayTagData(w)))
        tagPipeMeta(w).valid := Mux(currentRefillBypass || priorRefillBypass,
          true.B, tagReqMeta(w).valid)
        // The default version is captured when the tag SRAM read is launched,
        // not when its result is consumed.  Otherwise a refill between those
        // events can pair the old SRAM tag with the new cache generation.
        tagPipeMeta(w).generation := Mux(currentRefillBypass,
          !cacheMeta(w)(setIdxOf(tagReq.addr)).generation,
          Mux(priorRefillBypass,
            cacheMeta(w)(setIdxOf(tagReq.addr)).generation,
            tagReqMeta(w).generation))
      }
    }
    tagReqValid := reqFire
    when(reqFire) {
      tagReq := headReq
      for (w <- 0 until ways) {
        tagReqMeta(w) := cacheMeta(w)(headSet)
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

  respQ.io.enq.valid := dataReturnValid || refillEmitValid
  respQ.io.enq.bits := Mux(dataReturnValid, dataResp, refillResp)
  io.resp.valid := respQ.io.deq.valid
  io.resp.bits := respQ.io.deq.bits
  respQ.io.deq.ready := io.resp.ready

  val refillEmitFire = refillEmitValid && !dataReturnValid && respQ.io.enq.ready
  when(dataReturnValid) {
    assert(respQ.io.enq.ready, "TriangleMemCachedBank response queue overflow")
  }

  when(!reset.asBool) {
    refillTagBypassValid := false.B
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

    io.refillDone.ready := !refillEmitValid
    when(io.refillDone.fire) {
    val doneId = io.refillDone.bits.id
    assert(pendingCount(doneId) =/= 0.U, "TriangleMemCachedBank refill done with no pending waiters")
    writeSetIdx := pendingSet(doneId)
    writeData := io.refillDone.bits.data
    emitMshrId := doneId
    emitIdx := 0.U
    emitTotal := pendingCount(doneId)
    emitDataReg := io.refillDone.bits.data
    for (w <- 0 until ways) {
      when(pendingVictimWay(doneId) === w.U) {
        wayWriteEn(w) := true.B
        tagWriteEn(w) := true.B
        cacheMeta(w)(pendingSet(doneId)).valid := true.B
        cacheMeta(w)(pendingSet(doneId)).generation := !cacheMeta(w)(pendingSet(doneId)).generation
      }
    }
    tagWriteData := pendingTag(doneId)
    refillTagBypassValid := true.B
    refillTagBypassSet := pendingSet(doneId)
    refillTagBypassWay := pendingVictimWay(doneId)
    refillTagBypassData := pendingTag(doneId)
    // A tag lookup can already be parked in tagPipe while a refill overwrites
    // the same direct-mapped set.  Update the parked lookup's tag snapshot as
    // part of the refill; otherwise it may retain a stale hit decision and,
    // after the refill stall clears, read the newly written data under the old
    // request address.
    // Only rewrite the snapshot when the current lookup remains parked.  If
    // it advances this cycle, the block above has already loaded tagPipe for
    // tagReq (including the current-refill bypass).  Applying the old
    // lookupSet here would corrupt that next request's tag snapshot.
    when(tagPipeValid && !tagPipeCanAdvance && lookupSet === pendingSet(doneId)) {
      for (w <- 0 until ways) {
        when(pendingVictimWay(doneId) === w.U) {
          tagPipeData(w) := pendingTag(doneId)
          tagPipeMeta(w).valid := true.B
          tagPipeMeta(w).generation := !cacheMeta(w)(pendingSet(doneId)).generation
        }
      }
    }
    if (ways > 1) {
      cacheLru.get(pendingSet(doneId)) := !pendingVictimWay(doneId)
    }
    refillEmitValid := true.B
    }

    io.allocReq.valid := missQ.io.deq.valid && !refillEmitValid && !refillAccept
    io.allocReq.bits.victimWay := victimWayForMiss
    io.allocResp.ready := io.allocReq.valid &&
      pendingCount(io.allocResp.bits.id) =/= mergeQueueDepth.U
    missQ.io.deq.ready := io.allocResp.fire
    when(io.allocResp.fire) {
      enqueuePending(
        io.allocResp.bits.id,
        missQ.io.deq.bits,
        setIdxOf(missQ.io.deq.bits.addr),
        tagOf(missQ.io.deq.bits.addr),
        victimWayForMiss
      )
    }

    when(refillEmitFire) {
      when(emitIdx + 1.U >= emitTotal) {
        pendingCount(emitMshrId) := 0.U
        io.release.valid := true.B
        refillEmitValid := false.B
      }.otherwise {
        emitIdx := emitIdx + 1.U
      }
    }
  }
}
