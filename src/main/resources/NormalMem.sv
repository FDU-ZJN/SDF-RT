module NormalMem #(
  parameter int ADDR_WIDTH = 16,
  parameter int DATA_WIDTH = 96,
  parameter int LATENCY = 2,
  parameter int MAX_ENTRIES = 8554,
  parameter string INIT_FILE = "normal_mem.mem"
) (
  input  logic                   clk,
  input  logic                   reset,
  input  logic [ADDR_WIDTH-1:0]  addr,
  input  logic                   en,
  output logic [DATA_WIDTH-1:0]  data,
  output logic                   valid,
  output logic [ADDR_WIDTH-1:0]  addr_q
);

  localparam int FIXED_LATENCY      = LATENCY;
  localparam int MEM_ADDR_WIDTH     = (MAX_ENTRIES <= 1) ? 1 : $clog2(MAX_ENTRIES);

  logic [FIXED_LATENCY-1:0]    valid_pipe;
  logic [ADDR_WIDTH-1:0]       addr_pipe [0:FIXED_LATENCY-1];
  logic [MEM_ADDR_WIDTH-1:0]   rd_addr_s0;
  logic                        rd_in_range_s0;
  logic [FIXED_LATENCY-1:0]    rd_ok_pipe;
  logic [DATA_WIDTH-1:0]       bram_dout;
  integer i;

  initial begin
    if (LATENCY < 2) begin
      $error("[NormalMem] LATENCY=%0d is invalid, expected >= 2", LATENCY);
    end
  end

  assign rd_addr_s0 = addr[MEM_ADDR_WIDTH-1:0];
  assign rd_in_range_s0 = (addr < MAX_ENTRIES);

  always_ff @(posedge clk) begin
    if (reset) begin
      valid_pipe  <= '0;
      for (i = 0; i < FIXED_LATENCY; i = i + 1) begin
        addr_pipe[i] <= '0;
      end
      rd_ok_pipe <= '0;
    end else begin
      valid_pipe[0] <= en;
      addr_pipe[0]  <= addr;
      for (i = 1; i < FIXED_LATENCY; i = i + 1) begin
        valid_pipe[i] <= valid_pipe[i - 1];
        addr_pipe[i]  <= addr_pipe[i - 1];
      end

      rd_ok_pipe[0] <= rd_in_range_s0;
      for (i = 1; i < FIXED_LATENCY; i = i + 1) begin
        rd_ok_pipe[i] <= rd_ok_pipe[i - 1];
      end
    end
  end

  normal_mem normal_mem_inst (
    .clka  (clk),
    .addra (rd_addr_s0),
    .douta (bram_dout)
  );

  assign data   = rd_ok_pipe[FIXED_LATENCY - 1] ? bram_dout : '0;
  assign valid  = valid_pipe[FIXED_LATENCY - 1];
  assign addr_q = addr_pipe[FIXED_LATENCY - 1];

endmodule
