module SubgridMetaMem #(
  parameter int ADDR_WIDTH = 32,
  parameter int GLOBALRES = 8,
  parameter int SUBRES = 1,
  parameter int LATENCY = 2,
  parameter int MAX_ENTRIES = 512
) (
  input  logic                   clk,
  input  logic                   reset,
  input  logic [ADDR_WIDTH-1:0]  globalIdx,
  input  logic [ADDR_WIDTH-1:0]  subIdx,
  input  logic                   en,
  output logic [15:0]            triStart,
  output logic [15:0]            triCount,
  output logic                   valid
);

  localparam int FIXED_LATENCY = 2;
  localparam int GLOBAL_ADDR_WIDTH = $clog2(GLOBALRES) * 3;
  localparam int SUB_ADDR_WIDTH    = $clog2(SUBRES) * 3;
  localparam int BRAM_ADDR_WIDTH   = $clog2(MAX_ENTRIES);

  logic [31:0] lookup_addr;
  logic [BRAM_ADDR_WIDTH-1:0] addr;
  logic [31:0] data;

  logic [FIXED_LATENCY-1:0] valid_pipe;

  integer i;

  initial begin
    if (LATENCY != FIXED_LATENCY) begin
      $warning("[SubgridMetaMem] LATENCY=%0d is ignored, fixed latency is %0d", LATENCY, FIXED_LATENCY);
    end
  end

  generate
    if (SUB_ADDR_WIDTH > 0) begin : gen_combined_addr
      logic [GLOBAL_ADDR_WIDTH + SUB_ADDR_WIDTH - 1:0] combined_addr;
      assign combined_addr = {globalIdx[GLOBAL_ADDR_WIDTH-1:0], subIdx[SUB_ADDR_WIDTH-1:0]};
      assign lookup_addr = {{(32-(GLOBAL_ADDR_WIDTH + SUB_ADDR_WIDTH)){1'b0}}, combined_addr};
    end else begin : gen_global_only_addr
      assign lookup_addr = {{(32-GLOBAL_ADDR_WIDTH){1'b0}}, globalIdx[GLOBAL_ADDR_WIDTH-1:0]};
    end
  endgenerate

  assign addr = lookup_addr[BRAM_ADDR_WIDTH-1:0];

  dda_mem subgridmem_inst (
    .clka(clk),
    .addra(addr),
    .douta(data)
  );

  always_ff @(posedge clk) begin
    if (reset) begin
      valid_pipe <= '0;
    end else begin
      valid_pipe[0] <= en;
      for (i = 1; i < FIXED_LATENCY; i = i + 1) begin
        valid_pipe[i] <= valid_pipe[i - 1];
      end
    end
  end

  assign triStart = data[31:16];
  assign triCount = data[15:0];
  assign valid    = valid_pipe[FIXED_LATENCY - 1];

endmodule
