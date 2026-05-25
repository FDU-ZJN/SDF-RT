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
    val wr_data = Input(UInt(32.W))
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
       |  input  [31:0] wr_data
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
        "LATENCY"          -> GlobalConfig.sdfMemDpiLatency,
        "LOCAL_CELL_COUNT" -> GlobalConfig.LocalCell,
        "LOCAL_PER_CELL"   -> GlobalConfig.sdfMemLocalPerCell
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
    val wr_data = Input(UInt(32.W))
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
        "LOCAL_CELL_COUNT" -> GlobalConfig.LocalCell,
        "LOCAL_PER_CELL"   -> GlobalConfig.sdfMemLocalPerCell,
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

private class SdfMem2RIpBB(
  addrWidth: Int = 32,
  dataWidth: Int = 32
) extends BlackBox(
      Map(
        "ADDR_WIDTH"       -> addrWidth,
        "DATA_WIDTH"       -> dataWidth,
        "GLOBAL_ADDR_BITS" -> GlobalConfig.sdfMemGlobalAddrWidth,
        "LATENCY"          -> GlobalConfig.sdfMemDpiLatency,
        "LOCAL_CELL_COUNT" -> GlobalConfig.LocalCell,
        "LOCAL_PER_CELL"   -> GlobalConfig.sdfMemLocalPerCell,
        "LOCAL_IDX_INIT_FILE" -> StringParam("sdf_local_mapping.mem")
      )
    )
    with HasBlackBoxResource {
  override def desiredName: String = "SdfMem2R"
  val io = IO(new Bundle {
    val clk = Input(Clock())
    val reset = Input(Reset())
    val globalIdx0 = Input(UInt(addrWidth.W))
    val localIdx0 = Input(UInt(addrWidth.W))
    val en0 = Input(Bool())
    val data0 = Output(UInt(dataWidth.W))
    val valid0 = Output(Bool())
    val globalIdx1 = Input(UInt(addrWidth.W))
    val localIdx1 = Input(UInt(addrWidth.W))
    val en1 = Input(Bool())
    val data1 = Output(UInt(dataWidth.W))
    val valid1 = Output(Bool())
    val wr_en = Input(Bool())
    val wr_addr = Input(UInt(32.W))
    val wr_data = Input(UInt(32.W))
  })
  addResource("/SdfMem.sv")
}

private class SdfMemDPICore2R(
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
    val globalIdx0 = Input(UInt(addrWidth.W))
    val localIdx0 = Input(UInt(addrWidth.W))
    val en0 = Input(Bool())
    val data0 = Output(UInt(dataWidth.W))
    val valid0 = Output(Bool())
    val globalIdx1 = Input(UInt(addrWidth.W))
    val localIdx1 = Input(UInt(addrWidth.W))
    val en1 = Input(Bool())
    val data1 = Output(UInt(dataWidth.W))
    val valid1 = Output(Bool())
    val wr_en = Input(Bool())
    val wr_addr = Input(UInt(32.W))
    val wr_data = Input(UInt(32.W))
  })

  private val svCode =
    s"""
       |import \"DPI-C\" function int sdf_mem_read(input int unsigned global_idx, input int unsigned local_idx);
       |
       |module SdfMemDPICore2R (
       |  input clk,
       |  input reset,
       |  input  [${addrWidth - 1}:0] globalIdx0,
       |  input  [${addrWidth - 1}:0] localIdx0,
       |  input en0,
       |  output [${dataWidth - 1}:0] data0,
       |  output valid0,
       |  input  [${addrWidth - 1}:0] globalIdx1,
       |  input  [${addrWidth - 1}:0] localIdx1,
       |  input en1,
       |  output [${dataWidth - 1}:0] data1,
       |  output valid1,
       |  input wr_en,
       |  input  [31:0] wr_addr,
       |  input  [31:0] wr_data
       |);
       |  int dpi_data0;
       |  int dpi_data1;
       |  reg [${dataWidth - 1}:0] data_pipe0[0:${latency - 1}];
       |  reg [${dataWidth - 1}:0] data_pipe1[0:${latency - 1}];
       |  reg [${latency - 1}:0] valid_pipe0;
       |  reg [${latency - 1}:0] valid_pipe1;
       |  integer i;
       |
       |  always @(posedge clk) begin
       |    if (reset) begin
       |      valid_pipe0 <= '0;
       |      valid_pipe1 <= '0;
       |      for (i = 0; i < ${latency}; i = i + 1) begin
       |        data_pipe0[i] <= '0;
       |        data_pipe1[i] <= '0;
       |      end
       |    end else begin
       |      valid_pipe0[0] <= en0;
       |      valid_pipe1[0] <= en1;
       |      if (en0) begin
       |        dpi_data0 = sdf_mem_read(globalIdx0, localIdx0);
       |        data_pipe0[0] <= dpi_data0[${dataWidth - 1}:0];
       |      end
       |      if (en1) begin
       |        dpi_data1 = sdf_mem_read(globalIdx1, localIdx1);
       |        data_pipe1[0] <= dpi_data1[${dataWidth - 1}:0];
       |      end
       |      for (i = 1; i < ${latency}; i = i + 1) begin
       |        valid_pipe0[i] <= valid_pipe0[i - 1];
       |        valid_pipe1[i] <= valid_pipe1[i - 1];
       |        data_pipe0[i] <= data_pipe0[i - 1];
       |        data_pipe1[i] <= data_pipe1[i - 1];
       |      end
       |    end
       |  end
       |
       |  assign data0 = data_pipe0[${latency - 1}];
       |  assign valid0 = valid_pipe0[${latency - 1}];
       |  assign data1 = data_pipe1[${latency - 1}];
       |  assign valid1 = valid_pipe1[${latency - 1}];
       |endmodule
       |""".stripMargin

  setInline("SdfMemDPI2R.sv", svCode)
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

class SdfMem2R(
  addrWidth: Int = 32,
  dataWidth: Int = GlobalConfig.sdfMemDataWidth,
  latency: Int = GlobalConfig.sdfMemDpiLatency
) extends Module {
  val io = IO(new Bundle {
    val clk = Input(Clock())
    val reset = Input(Reset())
    val globalIdx = Input(Vec(2, UInt(addrWidth.W)))
    val localIdx = Input(Vec(2, UInt(addrWidth.W)))
    val en = Input(Vec(2, Bool()))
    val data = Output(Vec(2, UInt(dataWidth.W)))
    val valid = Output(Vec(2, Bool()))
    val wr = Flipped(new SdfMemWriteIO)
  })

  GlobalConfig.memImplMode match {
    case 0 =>
      val impl = Module(new SdfMemDPICore2R(addrWidth, dataWidth, latency))
      impl.io.clk := io.clk
      impl.io.reset := io.reset
      impl.io.globalIdx0 := io.globalIdx(0)
      impl.io.localIdx0 := io.localIdx(0)
      impl.io.en0 := io.en(0)
      io.data(0) := impl.io.data0
      io.valid(0) := impl.io.valid0
      impl.io.globalIdx1 := io.globalIdx(1)
      impl.io.localIdx1 := io.localIdx(1)
      impl.io.en1 := io.en(1)
      io.data(1) := impl.io.data1
      io.valid(1) := impl.io.valid1
      impl.io.wr_en := io.wr.wr_en
      impl.io.wr_addr := io.wr.wr_addr
      impl.io.wr_data := io.wr.wr_data
    case 1 | 2 =>
      val impl = Module(new SdfMem2RIpBB(addrWidth, dataWidth))
      impl.io.clk := io.clk
      impl.io.reset := io.reset
      impl.io.globalIdx0 := io.globalIdx(0)
      impl.io.localIdx0 := io.localIdx(0)
      impl.io.en0 := io.en(0)
      io.data(0) := impl.io.data0
      io.valid(0) := impl.io.valid0
      impl.io.globalIdx1 := io.globalIdx(1)
      impl.io.localIdx1 := io.localIdx(1)
      impl.io.en1 := io.en(1)
      io.data(1) := impl.io.data1
      io.valid(1) := impl.io.valid1
      impl.io.wr_en := io.wr.wr_en
      impl.io.wr_addr := io.wr.wr_addr
      impl.io.wr_data := io.wr.wr_data
  }
}
