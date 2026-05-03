module SubgridMetaMemResourceBB #(
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
  localparam int FIXED_LATENCY    = 2;
  localparam int GLOBAL_ADDR_WIDTH = $clog2(GLOBALRES) * 3;
  localparam int SUB_ADDR_WIDTH    = $clog2(SUBRES) * 3;
  localparam int LOOKUP_ADDR_WIDTH = GLOBAL_ADDR_WIDTH + SUB_ADDR_WIDTH;

  logic [31:0] lookup_addr;
  logic [31:0] data_pipe [0:FIXED_LATENCY-1];
  logic [FIXED_LATENCY-1:0] valid_pipe;
  integer i;

  reg [31:0] subgrid_meta_mem [0:MAX_ENTRIES-1];

  initial begin
    if (LATENCY != FIXED_LATENCY) begin
      $warning("[SubgridMetaMem] LATENCY=%0d is ignored, fixed latency is %0d", LATENCY, FIXED_LATENCY);
    end

    string mem_file;
    if ($value$plusargs("SUBGRID_META_MEM_FILE=%s", mem_file)) begin
      $display("[SubgridMetaMem] Loading subgrid meta memory from %s", mem_file);
      $readmemh(mem_file, subgrid_meta_mem);
    end else begin
      $display("[SubgridMetaMem] Warning: SUBGRID_META_MEM_FILE not specified, using empty memory");
    end
  end

  generate
    if (SUB_ADDR_WIDTH > 0) begin : gen_combined_addr
      logic [LOOKUP_ADDR_WIDTH-1:0] combined_addr;
      assign combined_addr = {globalIdx[GLOBAL_ADDR_WIDTH-1:0], subIdx[SUB_ADDR_WIDTH-1:0]};
      assign lookup_addr = {{(32-LOOKUP_ADDR_WIDTH){1'b0}}, combined_addr};
    end else begin : gen_global_only_addr
      assign lookup_addr = {{(32-GLOBAL_ADDR_WIDTH){1'b0}}, globalIdx[GLOBAL_ADDR_WIDTH-1:0]};
    end
  endgenerate

  always_ff @(posedge clk) begin
    if (reset) begin
      valid_pipe <= '0;
      for (i = 0; i < FIXED_LATENCY; i = i + 1) begin
        data_pipe[i] <= '0;
      end
    end else begin
      valid_pipe[0] <= en;
      if (en) begin
        if (lookup_addr < MAX_ENTRIES) begin
          data_pipe[0] <= subgrid_meta_mem[lookup_addr];
        end else begin
          data_pipe[0] <= '0;
        end
      end else begin
        data_pipe[0] <= '0;
      end

      for (i = 1; i < FIXED_LATENCY; i = i + 1) begin
        valid_pipe[i] <= valid_pipe[i - 1];
        data_pipe[i]  <= data_pipe[i - 1];
      end
    end
  end

  assign triStart = data_pipe[FIXED_LATENCY - 1][31:16];
  assign triCount = data_pipe[FIXED_LATENCY - 1][15:0];
  assign valid    = valid_pipe[FIXED_LATENCY - 1];
endmodule
