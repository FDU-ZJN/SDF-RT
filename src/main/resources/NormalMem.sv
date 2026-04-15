module NormalMem #(
  parameter int ADDR_WIDTH = 16,
  parameter int DATA_WIDTH = 96,
  parameter int LATENCY = 4,
  parameter int MAX_ENTRIES = 14204
) (
  input  logic                   clk,
  input  logic                   reset,
  input  logic [ADDR_WIDTH-1:0]  addr,
  input  logic                   en,
  output logic [DATA_WIDTH-1:0]  data,
  output logic                   valid,
  output logic [ADDR_WIDTH-1:0]  addr_q
);

  localparam int FIXED_LATENCY = 4;

  logic [FIXED_LATENCY-1:0]    valid_pipe;
  logic [ADDR_WIDTH-1:0]       addr_pipe [0:FIXED_LATENCY-1];
  logic [ADDR_WIDTH-1:0]       normal_id;

  
  integer i;

  initial begin
    if (LATENCY != FIXED_LATENCY) begin
      $warning("[NormalMem] LATENCY=%0d is ignored, fixed latency is %0d", LATENCY, FIXED_LATENCY);
    end
  end

  always_ff @(posedge clk) begin
    if (reset) begin
      valid_pipe <= '0;
      for (i = 0; i < FIXED_LATENCY; i = i + 1) begin
        addr_pipe[i] <= '0;
      end
    end else begin
      valid_pipe[0] <= en;
      addr_pipe[0]  <= addr;


      for (i = 1; i < FIXED_LATENCY; i = i + 1) begin
        valid_pipe[i] <= valid_pipe[i - 1];
        addr_pipe[i]  <= addr_pipe[i - 1];
      end
    end
  end

  flat_idx_mem flat_idx_mem_inst (
    .clka(clk),
    .addra(addr),
    .douta(normal_id)
  );

  normal_mem normal_mem_inst (
    .clka(clk),
    .addra(normal_id),
    .douta(data)
  );

  assign valid  = valid_pipe[FIXED_LATENCY - 1];
  assign addr_q = addr_pipe[FIXED_LATENCY - 1];

endmodule
