package DDA

import chisel3._
import chisel3.util._
import raytrace_utils._

class DdaTraceCmdBuffer(
  cfg: FloatConfig = FloatConfig.FP32,
  addrWidth: Int = 32,
  maxCmds: Int = 1024,
  slotCount: Int = GlobalConfig.ddaRetryQueueDepth
) extends Module {
  private val cmdCountW = log2Ceil(maxCmds + 1)

  val io = IO(new Bundle {
    val clear = Flipped(Valid(UInt(GlobalConfig.ddaTraceSlotBits.W)))
    val write = Flipped(Valid(new DdaTraceCmdWrite(addrWidth, maxCmds)))
    val readSlot = Input(UInt(GlobalConfig.ddaTraceSlotBits.W))
    val readCmdCount = Output(UInt(cmdCountW.W))
    val readCmds = Output(Vec(maxCmds, new TriBatch(addrWidth)))
  })

  val cmdCounts = RegInit(VecInit(Seq.fill(slotCount)(0.U(cmdCountW.W))))
  val cmdStore = Reg(Vec(slotCount, Vec(maxCmds, new TriBatch(addrWidth))))

  when(io.clear.valid) {
    cmdCounts(io.clear.bits) := 0.U
  }

  when(io.write.valid) {
    assert(io.write.bits.cmdIdx < maxCmds.U, "DDA trace command buffer overflow")
    cmdStore(io.write.bits.slotIdx)(io.write.bits.cmdIdx) := io.write.bits.tri
    cmdCounts(io.write.bits.slotIdx) := io.write.bits.cmdIdx + 1.U
  }

  io.readCmdCount := cmdCounts(io.readSlot)
  io.readCmds := cmdStore(io.readSlot)
}
