package SDF

import chisel3._
import chisel3.experimental.StringParam
import chisel3.util.{HasBlackBoxInline, HasBlackBoxResource}
import raytrace_utils.GlobalConfig


class SdfMemWriteIO extends Bundle {
  val wr_en   = Output(Bool())
  val wr_addr = Output(UInt(32.W))    // Full 32-bit address (auto-decoded)
  val wr_data = Output(UInt(32.W))    // 32-bit wide: single entry write
}

private class SdfMemDPICore(
  addrWidth: Int = 32,
  dataWidth: Int = 32,
  latency: Int = GlobalConfig.sdfMemDpiLatency
)
    extends BlackBox
    with HasBlackBoxInline {
  require(latency >= 1, s"SdfMemDPI latency must be >= 1, got $latency")

  val io = IO(new Bundle {
    val clk = Input(Clock())
    val reset = Input(Reset())
    val globalIdx = Input(UInt(addrWidth.W))
    val localIdx = Input(UInt(addrWidth.W))
    val en = Input(Bool())
    val data = Output(UInt(dataWidth.W))
    val valid = Output(Bool())
    // Simplified write port (unused in DPI mode)
    val wr_en   = Input(Bool())
    val wr_addr = Input(UInt(32.W))
    val wr_data = Input(UInt(2048.W))
  })

  private val svCode =
    s"""
       |import \"DPI-C\" function int sdf_mem_read(input int unsigned global_idx, input int unsigned local_idx);
       |
       |module SdfMemDPICore (
       |  input clk,
       |  input reset,
       |  input  [${addrWidth - 1}:0] globalIdx,
       |  input  [${addrWidth - 1}:0] localIdx,
       |  input en,
       |  output [${dataWidth - 1}:0] data,
       |  output valid,
       |  input wr_en,
       |  input  [31:0] wr_addr,
       |  input  [2047:0] wr_data
       |);
       |  int dpi_data;
       |  reg [${dataWidth - 1}:0] data_pipe[0:${latency - 1}];
       |  reg [${latency - 1}:0] valid_pipe;
       |  integer i;
       |
       |  always @(posedge clk) begin
       |    if (reset) begin
       |      valid_pipe <= '0;
       |      for (i = 0; i < ${latency}; i = i + 1) begin
       |        data_pipe[i] <= '0;
       |      end
       |    end else begin
       |      valid_pipe[0] <= en;
       |      if (en) begin
       |        dpi_data = sdf_mem_read(globalIdx, localIdx);
       |        data_pipe[0] <= dpi_data[${dataWidth - 1}:0];
       |      end
       |      for (i = 1; i < ${latency}; i = i + 1) begin
       |        valid_pipe[i] <= valid_pipe[i - 1];
       |        data_pipe[i] <= data_pipe[i - 1];
       |      end
       |    end
       |  end
       |
       |  // Write ports are unused in DPI mode (no-op)
       |  assign data = data_pipe[${latency - 1}];
       |  assign valid = valid_pipe[${latency - 1}];
       |endmodule
       |""".stripMargin

  setInline("SdfMemDPI.sv", svCode)
}

private class SdfMemResourceBB(
  addrWidth: Int = 32,
  dataWidth: Int = 32
) extends BlackBox(
      Map(
        "ADDR_WIDTH"       -> addrWidth,
        "DATA_WIDTH"       -> dataWidth,
        "GLOBAL_ADDR_BITS" -> GlobalConfig.sdfMemGlobalAddrWidth,
        "LATENCY"          -> GlobalConfig.sdfMemDpiLatency
      )
    )
    with HasBlackBoxResource {
  val io = IO(new Bundle {
    val clk = Input(Clock())
    val reset = Input(Reset())
    val globalIdx = Input(UInt(addrWidth.W))
    val localIdx = Input(UInt(addrWidth.W))
    val en = Input(Bool())
    val data = Output(UInt(dataWidth.W))
    val valid = Output(Bool())
    val wr_en   = Input(Bool())
    val wr_addr = Input(UInt(32.W))
    val wr_data = Input(UInt(2048.W))
  })
  addResource("/SdfMemBlackBox.sv")
}

private class SdfMemIpBB(
  addrWidth: Int = 32,
  dataWidth: Int = 32
) extends BlackBox(
      Map(
        "ADDR_WIDTH"       -> addrWidth,
        "DATA_WIDTH"       -> dataWidth,
        "GLOBAL_ADDR_BITS" -> GlobalConfig.sdfMemGlobalAddrWidth,
        "LATENCY"          -> GlobalConfig.sdfMemDpiLatency,
        "LOCAL_IDX_INIT_FILE" -> StringParam("sdf_local_mapping.mem")
      )
    )
    with HasBlackBoxResource {
  override def desiredName: String = "SdfMem"
  val io = IO(new Bundle {
    val clk = Input(Clock())
    val reset = Input(Reset())
    val globalIdx = Input(UInt(addrWidth.W))
    val localIdx = Input(UInt(addrWidth.W))
    val en = Input(Bool())
    val data = Output(UInt(dataWidth.W))
    val valid = Output(Bool())
    val wr_en   = Input(Bool())
    val wr_addr = Input(UInt(32.W))
    val wr_data = Input(UInt(32.W))
  })
  addResource("/SdfMem.sv")
}

class SdfMemDPI(
  addrWidth: Int = 32,
  dataWidth: Int = GlobalConfig.sdfMemDataWidth,
  latency: Int = GlobalConfig.sdfMemDpiLatency
) extends Module {
  val io = IO(new Bundle {
    val clk = Input(Clock())
    val reset = Input(Reset())
    val globalIdx = Input(UInt(addrWidth.W))
    val localIdx = Input(UInt(addrWidth.W))
    val en = Input(Bool())
    val data = Output(UInt(dataWidth.W))
    val valid = Output(Bool())
    // Simplified write port for PS initialization
    val wr = Flipped(new SdfMemWriteIO)
  })

  GlobalConfig.memImplMode match {
    case 0 =>
      val impl = Module(new SdfMemDPICore(addrWidth, dataWidth, latency))
      impl.io.clk := io.clk
      impl.io.reset := io.reset
      impl.io.globalIdx := io.globalIdx
      impl.io.localIdx := io.localIdx
      impl.io.en := io.en
      io.data := impl.io.data
      io.valid := impl.io.valid
      // Connect write ports (unused in DPI mode)
      impl.io.wr_en := io.wr.wr_en
      impl.io.wr_addr := io.wr.wr_addr
      impl.io.wr_data := io.wr.wr_data
    case 1 =>
      val impl = Module(new SdfMemResourceBB(addrWidth, dataWidth))
      impl.io.clk := io.clk
      impl.io.reset := io.reset
      impl.io.globalIdx := io.globalIdx
      impl.io.localIdx := io.localIdx
      impl.io.en := io.en
      io.data := impl.io.data
      io.valid := impl.io.valid
      // Connect write ports
      impl.io.wr_en := io.wr.wr_en
      impl.io.wr_addr := io.wr.wr_addr
      impl.io.wr_data := io.wr.wr_data
    case 2 =>
      val impl = Module(new SdfMemIpBB(addrWidth, dataWidth))
      impl.io.clk := io.clk
      impl.io.reset := io.reset
      impl.io.globalIdx := io.globalIdx
      impl.io.localIdx := io.localIdx
      impl.io.en := io.en
      io.data := impl.io.data
      io.valid := impl.io.valid
      // Connect write ports
      impl.io.wr_en := io.wr.wr_en
      impl.io.wr_addr := io.wr.wr_addr
      impl.io.wr_data := io.wr.wr_data
  }
}
