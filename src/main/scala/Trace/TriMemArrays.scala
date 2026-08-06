package Trace

import chisel3._
import chisel3.util._
import raytrace_utils._

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
    val mem = RegInit(VecInit(Seq.fill(depth)(0.U(dataWidth.W))))
    val rdData = Reg(UInt(dataWidth.W))
    when(io.rd_en) {
      rdData := mem(io.rd_addr)
    }
    io.rd_data := rdData
    when(io.wr_en) {
      mem(io.wr_addr) := io.wr_data
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
