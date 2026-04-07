module SubgridMetaMemResourceBB #(
  parameter int ADDR_WIDTH = 32,
  parameter int GLOBALRES = 8,
  parameter int SUBRES = 1,
  parameter int LATENCY = 2
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
localparam int Global_ADDR_WIDTH = $clog2(GLOBALRES)*3;
localparam int SUB_ADDR_WIDTH = $clog2(SUBRES)*3;
localparam int DRAM_ADDR_WIDTH = Global_ADDR_WIDTH+SUB_ADDR_WIDTH;
logic [LATENCY-1:0]    valid_pipe;

// Memory storage for $readmemh initialization
// Packed format: [31:16] = triStart[15:0], [15:0] = triCount[15:0]
// This allows $readmemh to load both values from a single 32-bit word
localparam int MAX_ENTRIES = 65536; // 2^16 subgrids

reg [31:0] subgrid_meta_mem [0:MAX_ENTRIES-1];
reg mem_loaded = 1'b0;

// Pipeline registers for address
logic [ADDR_WIDTH-1:0] globalIdx_pipe [LATENCY-1:0];
logic [ADDR_WIDTH-1:0] subIdx_pipe [LATENCY-1:0];

// Initialize memory from file using $readmemh
initial begin
  string mem_file;
  if ($value$plusargs("SUBGRID_META_MEM_FILE=%s", mem_file)) begin
    $display("[SubgridMetaMem] Loading subgrid meta memory from %s", mem_file);
    $readmemh(mem_file, subgrid_meta_mem);
    mem_loaded = 1'b1;
  end else begin
    $display("[SubgridMetaMem] Warning: SUBGRID_META_MEM_FILE not specified, using empty memory");
  end
end

generate
  if (SUB_ADDR_WIDTH > 0) begin : gen_combined_addr
    logic [Global_ADDR_WIDTH + SUB_ADDR_WIDTH - 1:0] combined_addr;
    assign combined_addr = {globalIdx[Global_ADDR_WIDTH-1:0], subIdx[SUB_ADDR_WIDTH-1:0]};
    
    always_ff @(posedge clk) begin
      if (reset) begin
        valid_pipe <= '0;
        for (int i = 0; i < LATENCY; i++) begin
          globalIdx_pipe[i] <= '0;
          subIdx_pipe[i] <= '0;
        end
      end else begin
        globalIdx_pipe[0] <= globalIdx;
        subIdx_pipe[0] <= subIdx;
        valid_pipe[0] <= en;
        for (int i = 1; i < LATENCY; i++) begin
          globalIdx_pipe[i] <= globalIdx_pipe[i - 1];
          subIdx_pipe[i] <= subIdx_pipe[i - 1];
          valid_pipe[i] <= valid_pipe[i - 1];
        end
      end
    end
    
    // Read from memory and extract packed values
    always_ff @(posedge clk) begin
      if (en) begin
        if (mem_loaded && (combined_addr < MAX_ENTRIES)) begin
          // Packed format: [31:16] = triStart, [15:0] = triCount
          triStart <= subgrid_meta_mem[combined_addr][31:16];
          triCount <= subgrid_meta_mem[combined_addr][15:0];
        end else begin
          triStart <= '0;
          triCount <= '0;
        end
      end else begin
        triStart <= '0;
        triCount <= '0;
      end
    end
  end
  else begin : gen_global_only_addr
    always_ff @(posedge clk) begin
      if (reset) begin
        valid_pipe <= '0;
        for (int i = 0; i < LATENCY; i++) begin
          globalIdx_pipe[i] <= '0;
          subIdx_pipe[i] <= '0;
        end
      end else begin
        globalIdx_pipe[0] <= globalIdx;
        subIdx_pipe[0] <= subIdx;
        valid_pipe[0] <= en;
        for (int i = 1; i < LATENCY; i++) begin
          globalIdx_pipe[i] <= globalIdx_pipe[i - 1];
          subIdx_pipe[i] <= subIdx_pipe[i - 1];
          valid_pipe[i] <= valid_pipe[i - 1];
        end
      end
    end
    
    // Read from memory and extract packed values
    always_ff @(posedge clk) begin
      if (en) begin
        if (mem_loaded && (globalIdx[Global_ADDR_WIDTH-1:0] < MAX_ENTRIES)) begin
          // Packed format: [31:16] = triStart, [15:0] = triCount
          triStart <= subgrid_meta_mem[globalIdx[Global_ADDR_WIDTH-1:0]][31:16];
          triCount <= subgrid_meta_mem[globalIdx[Global_ADDR_WIDTH-1:0]][15:0];
        end else begin
          triStart <= '0;
          triCount <= '0;
        end
      end else begin
        triStart <= '0;
        triCount <= '0;
      end
    end
  end
endgenerate

assign valid = valid_pipe[LATENCY - 1];
endmodule

