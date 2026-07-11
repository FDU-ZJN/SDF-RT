package Trace

import chisel3._
import chisel3.experimental.StringParam
import chisel3.util._
import raytrace_utils._

private class TriangleMemDPICore(
  val c: TriPeConfig,
  val latency: Int = GlobalConfig.triMemDpiLatency,
  val bankId: Int = 0,
  val numBanks: Int = 1
) extends BlackBox with HasBlackBoxInline {
  require(latency >= 1, s"TriangleMemDPI latency must be >= 1, got $latency")

  // 计算单个三角形的字节数：3顶点 * 3坐标 * (位宽/8)
  val bytesPerTri = 3 * 3 * (c.cfg.totalWidth / 8)
  val totalBytes = c.cacheLineTriangles * bytesPerTri
  val totalBits = totalBytes * 8

  val io = IO(new Bundle {
    val clk = Input(Clock())
    val reset = Input(Reset())
    val bank_id = Input(UInt(32.W))
    val addr = Input(UInt(c.addrWidth.W))
    val req_valid = Input(Bool())
    val req_mask = Input(UInt(c.cacheLineTriangles.W))
    val data = Output(UInt(totalBits.W))
    val valid = Output(Bool())
    val valid_mask = Output(UInt(c.cacheLineTriangles.W))
    val addr_q = Output(UInt(c.addrWidth.W))
    val req_ready = Output(Bool())  // always ready
  })

  val svCode =
    s"""
       |import "DPI-C" function void tri_mem_read_bank(input int bank, input int addr, output byte data[]);
       |
       |module TriangleMemDPICore (
       |    input clk,
       |    input reset,
       |    input [31:0] bank_id,
       |    input [${c.addrWidth - 1}:0] addr,
       |    input req_valid,
       |    input [${c.cacheLineTriangles - 1}:0] req_mask,
       |    output [${totalBits - 1}:0] data,
       |    output valid,
       |    output [${c.cacheLineTriangles - 1}:0] valid_mask,
       |    output [${c.addrWidth - 1}:0] addr_q,
       |    output req_ready
       |);
       |    byte raw_buffer[${totalBytes}];
       |    reg [${totalBits - 1}:0] data_pipe[0:${latency - 1}];
       |    reg [${c.addrWidth - 1}:0] addr_pipe[0:${latency - 1}];
       |    reg [${latency - 1}:0] valid_pipe;
       |    reg [${c.cacheLineTriangles - 1}:0] mask_pipe [0:${latency - 1}];
       |    integer j;
       |
       |    assign req_ready = 1'b1;
       |
       |    always @(posedge clk) begin
       |        if (reset) begin
       |            valid_pipe <= '0;
       |            for (j = 0; j < ${latency}; j = j + 1) begin
       |                data_pipe[j] <= '0;
       |                addr_pipe[j] <= '0;
       |                mask_pipe[j] <= '0;
       |            end
       |        end else begin
       |            valid_pipe[0] <= req_valid;
       |            if (req_valid) begin
       |                tri_mem_read_bank(bank_id, addr, raw_buffer);
       |                addr_pipe[0] <= addr;
       |                mask_pipe[0] <= req_mask[${c.cacheLineTriangles - 1}:0];
       |                for (int i = 0; i < ${totalBytes}; i = i + 1) begin
       |                    data_pipe[0][i*8 +: 8] <= raw_buffer[i];
       |                end
       |            end
       |
       |            for (j = 1; j < ${latency}; j = j + 1) begin
       |                valid_pipe[j] <= valid_pipe[j - 1];
       |                data_pipe[j] <= data_pipe[j - 1];
       |                addr_pipe[j] <= addr_pipe[j - 1];
       |                mask_pipe[j] <= mask_pipe[j - 1];
       |            end
       |        end
       |    end
       |
       |    assign data = data_pipe[${latency - 1}];
       |    assign valid = valid_pipe[${latency - 1}];
       |    assign valid_mask = mask_pipe[${latency - 1}];
       |    assign addr_q = addr_pipe[${latency - 1}];
       |
       |endmodule
  """.stripMargin

  setInline("TriangleMemDPI.sv", svCode)
}

private class TriangleMemResourceBB(
  val c: TriPeConfig,
  val latency: Int = GlobalConfig.triMemDpiLatency,
  val bankId: Int = 0,
  val numBanks: Int = 1,
  val maxEntries: Int = -1
) extends BlackBox(
      Map(
        "ADDR_WIDTH" -> GlobalConfig.triMemAddrWidth,
        "DATA_WIDTH" -> (c.cacheLineTriangles * 9 * c.cfg.totalWidth),
        "LATENCY" -> latency,
        "NUM_PES" -> c.cacheLineTriangles,
        "BANK_ID" -> bankId,
        "NUM_BANKS" -> numBanks,
        "MAX_ENTRIES" -> (if (maxEntries > 0) maxEntries else GlobalConfig.triMemDepthFor(numBanks, c.cacheLineTriangles)),
        "INIT_FILE" -> StringParam(s"triangle_mem_bank${bankId}.mem")
      )
    )
    with HasBlackBoxResource {
  private val totalBits = c.cacheLineTriangles * 9 * c.cfg.totalWidth
  val io = IO(new Bundle {
    val clk = Input(Clock())
    val reset = Input(Reset())
    val addr = Input(UInt(GlobalConfig.triMemAddrWidth.W))
    val req_valid = Input(Bool())
    val req_mask = Input(UInt(c.cacheLineTriangles.W))
    val data = Output(UInt(totalBits.W))
    val valid = Output(Bool())
    val valid_mask = Output(UInt(c.cacheLineTriangles.W))
    val addr_q = Output(UInt(GlobalConfig.triMemAddrWidth.W))
    val req_ready = Output(Bool())
  })
  addResource("/TriangleMemBlackBox.sv")
}

private class TriangleMemIpBB(
  val c: TriPeConfig,
  val latency: Int = GlobalConfig.triMemDpiLatency,
  val bankId: Int = 0,
  val numBanks: Int = 1,
  val maxEntries: Int = -1
) extends BlackBox(
      Map(
        "ADDR_WIDTH" -> GlobalConfig.triMemAddrWidth,
        "DATA_WIDTH" -> (c.cacheLineTriangles * 9 * c.cfg.totalWidth),
        "LATENCY" -> latency,
        "NUM_PES" -> c.cacheLineTriangles,
        "BANK_ID" -> bankId,
        "NUM_BANKS" -> numBanks,
        "MAX_ENTRIES" -> (if (maxEntries > 0) maxEntries else GlobalConfig.triMemDepthFor(numBanks, c.cacheLineTriangles)),
        "INIT_FILE" -> StringParam(s"triangle_mem_bank${bankId}.mem")
      )
    )
    with HasBlackBoxResource {
  override def desiredName: String = "TriangleMem"
  private val totalBits = c.cacheLineTriangles * 9 * c.cfg.totalWidth
  val io = IO(new Bundle {
    val clk = Input(Clock())
    val reset = Input(Reset())
    val addr = Input(UInt(GlobalConfig.triMemAddrWidth.W))
    val req_valid = Input(Bool())
    val req_mask = Input(UInt(c.cacheLineTriangles.W))
    val data = Output(UInt(totalBits.W))
    val valid = Output(Bool())
    val valid_mask = Output(UInt(c.cacheLineTriangles.W))
    val addr_q = Output(UInt(GlobalConfig.triMemAddrWidth.W))
    val req_ready = Output(Bool())
  })
  addResource("/TriangleMem.sv")
}

class TriangleMemDPI(
  val c: TriPeConfig,
  val latency: Int = GlobalConfig.triMemDpiLatency,
  val bankId: Int = 0,
  val numBanks: Int = 1,
  val maxEntries: Int = -1
) extends Module {
  private val totalBits = c.cacheLineTriangles * 9 * c.cfg.totalWidth
  private val resolvedMaxEntries = if (maxEntries > 0) maxEntries else GlobalConfig.triMemDepthFor(numBanks, c.cacheLineTriangles)
  val io = IO(new Bundle {
    val clk = Input(Clock())
    val reset = Input(Reset())
    val addr = Input(UInt(GlobalConfig.triMemAddrWidth.W))
    val req_valid = Input(Bool())
    val req_mask = Input(UInt(c.cacheLineTriangles.W))
    val data = Output(UInt(totalBits.W))
    val valid = Output(Bool())
    val valid_mask = Output(UInt(c.cacheLineTriangles.W))
    val addr_q = Output(UInt(GlobalConfig.triMemAddrWidth.W))
    val req_ready = Output(Bool())
  })

  GlobalConfig.memImplMode match {
    case 0 =>
      val impl = Module(new TriangleMemDPICore(c, latency, bankId, numBanks))
      impl.io.clk := io.clk
      impl.io.reset := io.reset
      impl.io.bank_id := bankId.U(32.W)
      impl.io.addr := io.addr
      impl.io.req_valid := io.req_valid
      impl.io.req_mask := io.req_mask
      io.data := impl.io.data
      io.valid := impl.io.valid
      io.valid_mask := impl.io.valid_mask
      io.addr_q := impl.io.addr_q
      io.req_ready := impl.io.req_ready
    case 1 =>
      val impl = Module(new TriangleMemResourceBB(c, latency, bankId, numBanks, resolvedMaxEntries))
      impl.io.clk := io.clk
      impl.io.reset := io.reset
      impl.io.addr := io.addr
      impl.io.req_valid := io.req_valid
      impl.io.req_mask := io.req_mask
      io.data := impl.io.data
      io.valid := impl.io.valid
      io.valid_mask := impl.io.valid_mask
      io.addr_q := impl.io.addr_q
      io.req_ready := impl.io.req_ready
    case 2 =>
      val impl = Module(new TriangleMemIpBB(c, latency, bankId, numBanks, resolvedMaxEntries))
      impl.io.clk := io.clk
      impl.io.reset := io.reset
      impl.io.addr := io.addr
      impl.io.req_valid := io.req_valid
      impl.io.req_mask := io.req_mask
      io.data := impl.io.data
      io.valid := impl.io.valid
      io.valid_mask := impl.io.valid_mask
      io.addr_q := impl.io.addr_q
      io.req_ready := impl.io.req_ready
  }
}
