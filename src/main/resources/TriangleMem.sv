module TriangleMem #(
  parameter int ADDR_WIDTH = 32,
  parameter int DATA_WIDTH = 1152,
  parameter int LATENCY = 2,
  parameter int NUM_PES = 4,
  parameter int MAX_ENTRIES = 3551
) (
  input  logic                   clk,
  input  logic                   reset,
  input  logic [ADDR_WIDTH-1:0]  addr,
  input  logic                   req_valid,
  input  logic [NUM_PES-1:0]     req_mask,
  output logic [DATA_WIDTH-1:0]  data,
  output logic                   valid,
  output logic [NUM_PES-1:0]     valid_mask,
  output logic [ADDR_WIDTH-1:0]  addr_q,
  output logic                   req_ready
);

  localparam int FIXED_LATENCY = 2;
  localparam int BRAM_ADDR_WIDTH = $clog2(MAX_ENTRIES);

  logic [ADDR_WIDTH-1:0] tri_addr;
  logic [BRAM_ADDR_WIDTH-1:0] tri_addr_bram;
  assign tri_addr = addr >> 2;
  assign tri_addr_bram = tri_addr[BRAM_ADDR_WIDTH-1:0];

  assign req_ready = 1'b1;

  logic                  valid_pipe [0:FIXED_LATENCY-1];
  logic [NUM_PES-1:0]    mask_pipe [0:FIXED_LATENCY-1];
  logic [ADDR_WIDTH-1:0] addr_pipe [0:FIXED_LATENCY-1];

  logic [DATA_WIDTH-1:0] data_raw;

  integer i;

  initial begin
    if (LATENCY != FIXED_LATENCY) begin
      $warning("[TriangleMem] LATENCY=%0d is ignored, fixed latency is %0d", LATENCY, FIXED_LATENCY);
    end
  end

  tri_mem trimem_inst (
    .clka(clk),
    .addra(tri_addr_bram),
    .douta(data_raw)
  );

  always_ff @(posedge clk) begin
    if (reset) begin
      for (i = 0; i < FIXED_LATENCY; i = i + 1) begin
        valid_pipe[i] <= 1'b0;
        mask_pipe[i]  <= '0;
        addr_pipe[i]  <= '0;
      end
    end else begin
      valid_pipe[0] <= req_valid;
      addr_pipe[0]  <= addr;
      mask_pipe[0]  <= req_mask;

      for (i = 1; i < FIXED_LATENCY; i = i + 1) begin
        valid_pipe[i] <= valid_pipe[i-1];
        mask_pipe[i]  <= mask_pipe[i-1];
        addr_pipe[i]  <= addr_pipe[i-1];
      end
    end
  end

  assign data       = data_raw;
  assign valid      = valid_pipe[FIXED_LATENCY-1];
  assign valid_mask = mask_pipe[FIXED_LATENCY-1];
  assign addr_q     = addr_pipe[FIXED_LATENCY-1];

endmodule
