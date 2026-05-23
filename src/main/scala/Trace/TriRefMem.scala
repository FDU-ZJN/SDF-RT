package Trace

import chisel3._
import chisel3.experimental.StringParam
import chisel3.util._
import raytrace_utils.GlobalConfig

private class TriRefMemDPICore(
  val addrWidth: Int = GlobalConfig.triMemAddrWidth,
  val latency: Int = GlobalConfig.triRefMemDpiLatency
) extends BlackBox with HasBlackBoxInline {
  require(latency >= 1, s"TriRefMemDPI latency must be >= 1, got $latency")

  private val totalBytes = GlobalConfig.triRefMemDataWidth / 8
  val io = IO(new Bundle {
    val clk = Input(Clock())
    val reset = Input(Reset())
    val addr_a = Input(UInt(addrWidth.W))
    val en_a = Input(Bool())
    val data_a = Output(UInt(GlobalConfig.triRefMemDataWidth.W))
    val valid_a = Output(Bool())
    val addr_q_a = Output(UInt(addrWidth.W))
    val addr_b = Input(UInt(addrWidth.W))
    val en_b = Input(Bool())
    val data_b = Output(UInt(GlobalConfig.triRefMemDataWidth.W))
    val valid_b = Output(Bool())
    val addr_q_b = Output(UInt(addrWidth.W))
  })

  private val svCode =
    s"""
       |import "DPI-C" function void tri_ref_mem_read(input int addr, output byte data[]);
       |
       |module TriRefMemDPICore (
       |  input clk,
       |  input reset,
       |  input [${addrWidth - 1}:0] addr_a,
       |  input en_a,
       |  output [${GlobalConfig.triRefMemDataWidth - 1}:0] data_a,
       |  output valid_a,
       |  output [${addrWidth - 1}:0] addr_q_a,
       |  input [${addrWidth - 1}:0] addr_b,
       |  input en_b,
       |  output [${GlobalConfig.triRefMemDataWidth - 1}:0] data_b,
       |  output valid_b,
       |  output [${addrWidth - 1}:0] addr_q_b
       |);
       |  byte raw_buffer_a[${totalBytes}];
       |  byte raw_buffer_b[${totalBytes}];
       |  reg [${GlobalConfig.triRefMemDataWidth - 1}:0] data_pipe_a[0:${latency - 1}];
       |  reg [${GlobalConfig.triRefMemDataWidth - 1}:0] data_pipe_b[0:${latency - 1}];
       |  reg [${addrWidth - 1}:0] addr_pipe_a[0:${latency - 1}];
       |  reg [${addrWidth - 1}:0] addr_pipe_b[0:${latency - 1}];
       |  reg [${latency - 1}:0] valid_pipe_a;
       |  reg [${latency - 1}:0] valid_pipe_b;
       |  integer i;
       |  integer j;
       |
       |  always @(posedge clk) begin
       |    if (reset) begin
       |      valid_pipe_a <= '0;
       |      valid_pipe_b <= '0;
       |      for (j = 0; j < ${latency}; j = j + 1) begin
       |        data_pipe_a[j] <= '0;
       |        data_pipe_b[j] <= '0;
       |        addr_pipe_a[j] <= '0;
       |        addr_pipe_b[j] <= '0;
       |      end
       |    end else begin
       |      valid_pipe_a[0] <= en_a;
       |      if (en_a) begin
       |        tri_ref_mem_read(addr_a, raw_buffer_a);
       |        addr_pipe_a[0] <= addr_a;
       |        for (i = 0; i < ${totalBytes}; i = i + 1) begin
       |          data_pipe_a[0][i*8 +: 8] <= raw_buffer_a[i];
       |        end
       |      end
       |      valid_pipe_b[0] <= en_b;
       |      if (en_b) begin
       |        tri_ref_mem_read(addr_b, raw_buffer_b);
       |        addr_pipe_b[0] <= addr_b;
       |        for (i = 0; i < ${totalBytes}; i = i + 1) begin
       |          data_pipe_b[0][i*8 +: 8] <= raw_buffer_b[i];
       |        end
       |      end
       |      for (j = 1; j < ${latency}; j = j + 1) begin
       |        valid_pipe_a[j] <= valid_pipe_a[j - 1];
       |        data_pipe_a[j] <= data_pipe_a[j - 1];
       |        addr_pipe_a[j] <= addr_pipe_a[j - 1];
       |        valid_pipe_b[j] <= valid_pipe_b[j - 1];
       |        data_pipe_b[j] <= data_pipe_b[j - 1];
       |        addr_pipe_b[j] <= addr_pipe_b[j - 1];
       |      end
       |    end
       |  end
       |
       |  assign data_a = data_pipe_a[${latency - 1}];
       |  assign valid_a = valid_pipe_a[${latency - 1}];
       |  assign addr_q_a = addr_pipe_a[${latency - 1}];
       |  assign data_b = data_pipe_b[${latency - 1}];
       |  assign valid_b = valid_pipe_b[${latency - 1}];
       |  assign addr_q_b = addr_pipe_b[${latency - 1}];
       |endmodule
       |""".stripMargin

  setInline("TriRefMemDPI.sv", svCode)
}

private class TriRefMemResourceBB(
  val addrWidth: Int = GlobalConfig.triMemAddrWidth,
  val latency: Int = GlobalConfig.triRefMemDpiLatency
) extends BlackBox(
      Map(
        "ADDR_WIDTH" -> addrWidth,
        "DATA_WIDTH" -> GlobalConfig.triRefMemDataWidth,
        "LATENCY" -> latency,
        "MAX_ENTRIES" -> GlobalConfig.triRefMemDepth
      )
    )
    with HasBlackBoxResource {
  val io = IO(new Bundle {
    val clk = Input(Clock())
    val reset = Input(Reset())
    val addr_a = Input(UInt(addrWidth.W))
    val en_a = Input(Bool())
    val data_a = Output(UInt(GlobalConfig.triRefMemDataWidth.W))
    val valid_a = Output(Bool())
    val addr_q_a = Output(UInt(addrWidth.W))
    val addr_b = Input(UInt(addrWidth.W))
    val en_b = Input(Bool())
    val data_b = Output(UInt(GlobalConfig.triRefMemDataWidth.W))
    val valid_b = Output(Bool())
    val addr_q_b = Output(UInt(addrWidth.W))
  })
  addResource("/TriRefMemBlackBox.sv")
}

private class TriRefMemIpBB(
  val addrWidth: Int = GlobalConfig.triMemAddrWidth,
  val latency: Int = GlobalConfig.triRefMemDpiLatency
) extends BlackBox(
      Map(
        "ADDR_WIDTH" -> addrWidth,
        "DATA_WIDTH" -> GlobalConfig.triRefMemDataWidth,
        "LATENCY" -> latency,
        "MAX_ENTRIES" -> GlobalConfig.triRefMemDepth,
        "INIT_FILE" -> StringParam("triangle_ref_mem.mem")
      )
    )
    with HasBlackBoxResource {
  override def desiredName: String = "TriRefMem"
  val io = IO(new Bundle {
    val clk = Input(Clock())
    val reset = Input(Reset())
    val addr_a = Input(UInt(addrWidth.W))
    val en_a = Input(Bool())
    val data_a = Output(UInt(GlobalConfig.triRefMemDataWidth.W))
    val valid_a = Output(Bool())
    val addr_q_a = Output(UInt(addrWidth.W))
    val addr_b = Input(UInt(addrWidth.W))
    val en_b = Input(Bool())
    val data_b = Output(UInt(GlobalConfig.triRefMemDataWidth.W))
    val valid_b = Output(Bool())
    val addr_q_b = Output(UInt(addrWidth.W))
  })
  addResource("/TriRefMem.sv")
}

class TriRefMemDPI(
  val addrWidth: Int = GlobalConfig.triMemAddrWidth,
  val latency: Int = GlobalConfig.triRefMemDpiLatency
) extends Module {
  val io = IO(new Bundle {
    val clk = Input(Clock())
    val reset = Input(Reset())
    val addr_a = Input(UInt(addrWidth.W))
    val en_a = Input(Bool())
    val data_a = Output(UInt(GlobalConfig.triRefMemDataWidth.W))
    val valid_a = Output(Bool())
    val addr_q_a = Output(UInt(addrWidth.W))
    val addr_b = Input(UInt(addrWidth.W))
    val en_b = Input(Bool())
    val data_b = Output(UInt(GlobalConfig.triRefMemDataWidth.W))
    val valid_b = Output(Bool())
    val addr_q_b = Output(UInt(addrWidth.W))
  })

  GlobalConfig.memImplMode match {
    case 0 =>
      val impl = Module(new TriRefMemDPICore(addrWidth, latency))
      impl.io.clk := io.clk
      impl.io.reset := io.reset
      impl.io.addr_a := io.addr_a
      impl.io.en_a := io.en_a
      impl.io.addr_b := io.addr_b
      impl.io.en_b := io.en_b
      io.data_a := impl.io.data_a
      io.valid_a := impl.io.valid_a
      io.addr_q_a := impl.io.addr_q_a
      io.data_b := impl.io.data_b
      io.valid_b := impl.io.valid_b
      io.addr_q_b := impl.io.addr_q_b
    case 1 =>
      val impl = Module(new TriRefMemResourceBB(addrWidth, latency))
      impl.io.clk := io.clk
      impl.io.reset := io.reset
      impl.io.addr_a := io.addr_a
      impl.io.en_a := io.en_a
      impl.io.addr_b := io.addr_b
      impl.io.en_b := io.en_b
      io.data_a := impl.io.data_a
      io.valid_a := impl.io.valid_a
      io.addr_q_a := impl.io.addr_q_a
      io.data_b := impl.io.data_b
      io.valid_b := impl.io.valid_b
      io.addr_q_b := impl.io.addr_q_b
    case 2 =>
      val impl = Module(new TriRefMemIpBB(addrWidth, latency))
      impl.io.clk := io.clk
      impl.io.reset := io.reset
      impl.io.addr_a := io.addr_a
      impl.io.en_a := io.en_a
      impl.io.addr_b := io.addr_b
      impl.io.en_b := io.en_b
      io.data_a := impl.io.data_a
      io.valid_a := impl.io.valid_a
      io.addr_q_a := impl.io.addr_q_a
      io.data_b := impl.io.data_b
      io.valid_b := impl.io.valid_b
      io.addr_q_b := impl.io.addr_q_b
  }
}

class TriRefMemMultiPort(
  val numPorts: Int,
  val addrWidth: Int = GlobalConfig.triMemAddrWidth
) extends Module {
  require(numPorts > 0, "TriRefMemMultiPort requires at least one port")
  private val srcW = math.max(1, log2Ceil(numPorts))
  class RefReq extends Bundle {
    val addr = UInt(addrWidth.W)
    val src = UInt(srcW.W)
  }

  val io = IO(new Bundle {
    val req = Vec(numPorts, Flipped(Decoupled(UInt(addrWidth.W))))
    val resp = Vec(numPorts, Decoupled(UInt(GlobalConfig.triRefMemDataWidth.W)))
    val resp_addr = Output(Vec(numPorts, UInt(addrWidth.W)))
  })

  private val singlePortMode = numPorts == 1
  require(singlePortMode || (numPorts & 1) == 0, s"TriRefMemMultiPort requires numPorts == 1 or an even number of ports, got $numPorts")
  private val refMemLatency = GlobalConfig.triRefMemDpiLatency
  require(refMemLatency >= 1, s"TriRefMemMultiPort requires refMemLatency >= 1, got $refMemLatency")

  private val mem = Module(new TriRefMemDPI(addrWidth, refMemLatency))
  mem.io.clk := clock
  mem.io.reset := reset

  private def pipeSrc(fire: Bool, src: UInt): UInt = {
    val srcPipe = Reg(Vec(GlobalConfig.triRefMemDpiLatency, UInt(srcW.W)))
    srcPipe(0) := Mux(fire, src, 0.U)
    for (i <- 1 until GlobalConfig.triRefMemDpiLatency) {
      srcPipe(i) := srcPipe(i - 1)
    }
    srcPipe.last
  }

  if (singlePortMode) {
    val arbA = Module(new RRArbiter(new RefReq, 1))
    arbA.io.in(0).valid := io.req(0).valid
    arbA.io.in(0).bits.addr := io.req(0).bits
    arbA.io.in(0).bits.src := 0.U
    io.req(0).ready := arbA.io.in(0).ready

    mem.io.addr_a := arbA.io.out.bits.addr
    mem.io.en_a := arbA.io.out.valid
    arbA.io.out.ready := true.B
    mem.io.addr_b := 0.U
    mem.io.en_b := false.B

    val srcA = pipeSrc(arbA.io.out.fire, arbA.io.out.bits.src)
    for (i <- 0 until numPorts) {
      io.resp(i).valid := mem.io.valid_a && srcA === i.U
      io.resp(i).bits := mem.io.data_a
      io.resp_addr(i) := mem.io.addr_q_a
    }
  } else {
    val halfPorts = numPorts / 2
    val arbA = Module(new RRArbiter(new RefReq, halfPorts))
    val arbB = Module(new RRArbiter(new RefReq, numPorts - halfPorts))

    for (i <- 0 until halfPorts) {
      arbA.io.in(i).valid := io.req(i).valid
      arbA.io.in(i).bits.addr := io.req(i).bits
      arbA.io.in(i).bits.src := i.U
      io.req(i).ready := arbA.io.in(i).ready
    }
    for (i <- 0 until (numPorts - halfPorts)) {
      val port = i + halfPorts
      arbB.io.in(i).valid := io.req(port).valid
      arbB.io.in(i).bits.addr := io.req(port).bits
      arbB.io.in(i).bits.src := port.U
      io.req(port).ready := arbB.io.in(i).ready
    }

    mem.io.addr_a := arbA.io.out.bits.addr
    mem.io.en_a := arbA.io.out.valid
    arbA.io.out.ready := true.B
    mem.io.addr_b := arbB.io.out.bits.addr
    mem.io.en_b := arbB.io.out.valid
    arbB.io.out.ready := true.B

    val srcA = pipeSrc(arbA.io.out.fire, arbA.io.out.bits.src)
    val srcB = pipeSrc(arbB.io.out.fire, arbB.io.out.bits.src)

    for (i <- 0 until numPorts) {
      io.resp(i).valid := (mem.io.valid_a && srcA === i.U) || (mem.io.valid_b && srcB === i.U)
      io.resp(i).bits := Mux(mem.io.valid_a && srcA === i.U, mem.io.data_a, mem.io.data_b)
      io.resp_addr(i) := Mux(mem.io.valid_a && srcA === i.U, mem.io.addr_q_a, mem.io.addr_q_b)
    }
  }
}
