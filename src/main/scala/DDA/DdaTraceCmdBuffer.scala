package DDA

import chisel3._
import chisel3.util._
import raytrace_utils._

class DdaTraceCmdBuffer(
  cfg: FloatConfig = FloatConfig.FP32,
  addrWidth: Int = 32,
  maxCmds: Int = 1024,
  slotCount: Int = GlobalConfig.ddaTraceSlotCount
) extends Module {
  private val cmdCountW = log2Ceil(maxCmds + 1)

  val io = IO(new Bundle {
    val clear = Flipped(Valid(UInt(GlobalConfig.ddaTraceSlotBits.W)))
    val write = Vec(2, Flipped(Valid(new DdaTraceCmdWrite(addrWidth, maxCmds))))
    val readSlot = Input(UInt(GlobalConfig.ddaTraceSlotBits.W))
    val readCmdCount = Output(UInt(cmdCountW.W))
    val readCmds = Output(Vec(maxCmds, new TriBatch(addrWidth)))
  })

  val cmdCounts = RegInit(VecInit(Seq.fill(slotCount)(0.U(cmdCountW.W))))
  val cmdStore = Reg(Vec(slotCount, Vec(maxCmds, new TriBatch(addrWidth))))

  when(io.clear.valid) {
    cmdCounts(io.clear.bits) := 0.U
  }

  when(io.write(0).valid && io.write(1).valid) {
    assert(io.write(0).bits.slotIdx =/= io.write(1).bits.slotIdx, "DDA trace command buffer does not allow same-slot dual writes")
  }

  for (lane <- 0 until 2) {
    when(io.write(lane).valid) {
      assert(io.write(lane).bits.cmdIdx < maxCmds.U, s"DDA trace command buffer lane $lane overflow")
      cmdStore(io.write(lane).bits.slotIdx)(io.write(lane).bits.cmdIdx) := io.write(lane).bits.tri
      cmdCounts(io.write(lane).bits.slotIdx) := io.write(lane).bits.cmdIdx + 1.U
    }
  }

  io.readCmdCount := cmdCounts(io.readSlot)
  io.readCmds := cmdStore(io.readSlot)
}
