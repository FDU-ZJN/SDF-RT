package Render

import chisel3._
import chisel3.util._
import raytrace_utils.GlobalConfig

class NormalMemWriteIO extends Bundle {
  val wr_en   = Output(Bool())
  val wr_addr = Output(UInt(32.W))  // 32-bit word address within normal memory window
  val wr_data = Output(UInt(32.W))  // single FP32 lane write
}

private class NormalMemDPICore(
  val addrWidth: Int = 16,
  val latency: Int = GlobalConfig.normalMemDpiLatency
) extends BlackBox with HasBlackBoxInline {
  require(latency >= 1, s"NormalMemDPI latency must be >= 1, got $latency")

  val bytesPerNormal = 3 * 4 // 3 floats * 4 bytes
  val totalBits = bytesPerNormal * 8

  val io = IO(new Bundle {
    val clk = Input(Clock())
    val reset = Input(Reset())
    val addr = Input(UInt(addrWidth.W))
    val en = Input(Bool())
    val data = Output(UInt(totalBits.W))
    val valid = Output(Bool())
    val addr_q = Output(UInt(addrWidth.W))
    val wr_en = Input(Bool())
    val wr_addr = Input(UInt(32.W))
    val wr_data = Input(UInt(32.W))
  })

  val svCode =
    s"""
       |import "DPI-C" function void normal_mem_read(input int addr, output byte data[]);
       |
       |module NormalMemDPICore (
       |    input clk,
       |    input reset,
       |    input [${addrWidth - 1}:0] addr,
       |    input en,
       |    output [${totalBits - 1}:0] data,
       |    output valid,
       |    output [${addrWidth - 1}:0] addr_q,
       |    input wr_en,
       |    input [31:0] wr_addr,
       |    input [31:0] wr_data
       |);
       |    byte raw_buffer[${bytesPerNormal}];
       |    reg [${totalBits - 1}:0] data_pipe[0:${latency - 1}];
       |    reg [${addrWidth - 1}:0] addr_pipe[0:${latency - 1}];
       |    reg [${latency - 1}:0] valid_pipe;
       |    integer i;
       |    integer j;
       |
       |    always @(posedge clk) begin
       |        if (reset) begin
       |            valid_pipe <= '0;
       |            for (j = 0; j < ${latency}; j = j + 1) begin
       |                data_pipe[j] <= '0;
       |                addr_pipe[j] <= '0;
       |            end
       |        end else begin
       |            valid_pipe[0] <= en;
       |            if (en) begin
       |                normal_mem_read(addr, raw_buffer);
       |                addr_pipe[0] <= addr;
       |                for (i = 0; i < ${bytesPerNormal}; i = i + 1) begin
       |                    data_pipe[0][i*8 +: 8] <= raw_buffer[i];
       |                end
       |            end
       |
       |            for (j = 1; j < ${latency}; j = j + 1) begin
       |                valid_pipe[j] <= valid_pipe[j - 1];
       |                data_pipe[j] <= data_pipe[j - 1];
       |                addr_pipe[j] <= addr_pipe[j - 1];
       |            end
       |        end
       |    end
       |
       |    assign data = data_pipe[${latency - 1}];
       |    assign valid = valid_pipe[${latency - 1}];
       |    assign addr_q = addr_pipe[${latency - 1}];
       |    // Write port is unused in DPI mode.
       |    wire _unused_ok = &{1'b0, wr_en, wr_addr, wr_data};
       |
       |endmodule
  """.stripMargin

  setInline("NormalMemDPI.sv", svCode)
}

private class NormalMemResourceBB(
  val addrWidth: Int = GlobalConfig.normalMemAddrWidth,
  val latency: Int = GlobalConfig.normalMemDpiLatency
) extends BlackBox(
      Map(
        "ADDR_WIDTH" -> GlobalConfig.normalMemAddrWidth,
        "DATA_WIDTH" -> GlobalConfig.normalMemDataWidth,
        "LATENCY" -> latency,
        "MAX_ENTRIES" -> GlobalConfig.normalMemDepth
      )
    )
    with HasBlackBoxResource {
  val io = IO(new Bundle {
    val clk = Input(Clock())
    val reset = Input(Reset())
    val addr = Input(UInt(GlobalConfig.normalMemAddrWidth.W))
    val en = Input(Bool())
    val data = Output(UInt(GlobalConfig.normalMemDataWidth.W))
    val valid = Output(Bool())
    val addr_q = Output(UInt(GlobalConfig.normalMemAddrWidth.W))
    val wr_en = Input(Bool())
    val wr_addr = Input(UInt(32.W))
    val wr_data = Input(UInt(32.W))
  })
  addResource("/NormalMemBlackBox.sv")
}

private class NormalMemIpBB(
  val addrWidth: Int = GlobalConfig.normalMemAddrWidth,
  val latency: Int = GlobalConfig.normalMemDpiLatency
) extends BlackBox(
      Map(
        "ADDR_WIDTH" -> GlobalConfig.normalMemAddrWidth,
        "DATA_WIDTH" -> GlobalConfig.normalMemDataWidth,
        "LATENCY" -> latency,
        "MAX_ENTRIES" -> GlobalConfig.normalMemDepth
      )
    )
    with HasBlackBoxResource {
  override def desiredName: String = "NormalMem"
  val io = IO(new Bundle {
    val clk = Input(Clock())
    val reset = Input(Reset())
    val addr = Input(UInt(GlobalConfig.normalMemAddrWidth.W))
    val en = Input(Bool())
    val data = Output(UInt(GlobalConfig.normalMemDataWidth.W))
    val valid = Output(Bool())
    val addr_q = Output(UInt(GlobalConfig.normalMemAddrWidth.W))
    val wr_en = Input(Bool())
    val wr_addr = Input(UInt(32.W))
    val wr_data = Input(UInt(32.W))
  })
  addResource("/NormalMem.sv")
}

class NormalMemDPI(
  val addrWidth: Int = GlobalConfig.normalMemAddrWidth,
  val latency: Int = GlobalConfig.normalMemDpiLatency
) extends Module {
  private val totalBits = GlobalConfig.normalMemDataWidth
  val io = IO(new Bundle {
    val clk = Input(Clock())
    val reset = Input(Reset())
    val addr = Input(UInt(GlobalConfig.normalMemAddrWidth.W))
    val en = Input(Bool())
    val data = Output(UInt(totalBits.W))
    val valid = Output(Bool())
    val addr_q = Output(UInt(GlobalConfig.normalMemAddrWidth.W))
    val wr = Flipped(new NormalMemWriteIO)
  })

  GlobalConfig.memImplMode match {
    case 0 =>
      val impl = Module(new NormalMemDPICore(addrWidth, latency))
      impl.io.clk := io.clk
      impl.io.reset := io.reset
      impl.io.addr := io.addr
      impl.io.en := io.en
      io.data := impl.io.data
      io.valid := impl.io.valid
      io.addr_q := impl.io.addr_q
      impl.io.wr_en := io.wr.wr_en
      impl.io.wr_addr := io.wr.wr_addr
      impl.io.wr_data := io.wr.wr_data
    case 1 =>
      val impl = Module(new NormalMemResourceBB(addrWidth, latency))
      impl.io.clk := io.clk
      impl.io.reset := io.reset
      impl.io.addr := io.addr
      impl.io.en := io.en
      io.data := impl.io.data
      io.valid := impl.io.valid
      io.addr_q := impl.io.addr_q
      impl.io.wr_en := io.wr.wr_en
      impl.io.wr_addr := io.wr.wr_addr
      impl.io.wr_data := io.wr.wr_data
    case 2 =>
      val impl = Module(new NormalMemIpBB(addrWidth, latency))
      impl.io.clk := io.clk
      impl.io.reset := io.reset
      impl.io.addr := io.addr
      impl.io.en := io.en
      io.data := impl.io.data
      io.valid := impl.io.valid
      io.addr_q := impl.io.addr_q
      impl.io.wr_en := io.wr.wr_en
      impl.io.wr_addr := io.wr.wr_addr
      impl.io.wr_data := io.wr.wr_data
  }
}
