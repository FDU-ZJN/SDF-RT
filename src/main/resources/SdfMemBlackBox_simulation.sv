// Simplified SDF Memory BlackBox for Vivado simulation
// Uses $readmemh for initialization instead of complex URAM banks
// Note: This is a simplified version for simulation only, not for synthesis

module SdfMemResourceBB #(
  parameter int ADDR_WIDTH = 32,
  parameter int DATA_WIDTH = 32,
  parameter int GLOBAL_ADDR_BITS = 12,
  parameter int BANK_DEPTH = 4096,
  parameter int URAM_COUNT = 64,
  parameter int LOCAL_GRID_SIZE = 64
) (
  input  logic                   clk,
  input  logic                   reset,
  input  logic [ADDR_WIDTH-1:0]  globalIdx,
  input  logic [ADDR_WIDTH-1:0]  localIdx,
  input  logic                   en,
  output logic [DATA_WIDTH-1:0]  data,
  output logic                   valid
);

  // Simplified memory storage for simulation
  localparam int MAX_GLOBAL_ENTRIES = 4096;  // 2^12
  localparam int MAX_LOCAL_ENTRIES = 1048576; // 2^20
  
  reg [31:0] sdf_global_mem [0:MAX_GLOBAL_ENTRIES-1];
  reg [31:0] sdf_local_mem [0:MAX_LOCAL_ENTRIES-1];
  reg global_mem_loaded = 1'b0;
  reg local_mem_loaded = 1'b0;
  
  // Pipeline registers
  logic [ADDR_WIDTH-1:0] gidx_pipe [1:0];
  logic [ADDR_WIDTH-1:0] lidx_pipe [1:0];
  logic                  vld_pipe [1:0];
  
  // Initialize global SDF memory from file
  initial begin
    string mem_file;
    if ($value$plusargs("SDF_GLOBAL_MEM_FILE=%s", mem_file)) begin
      $display("[SdfMem] Loading global SDF memory from %s", mem_file);
      $readmemh(mem_file, sdf_global_mem);
      global_mem_loaded = 1'b1;
    end else begin
      $display("[SdfMem] Warning: SDF_GLOBAL_MEM_FILE not specified, using empty memory");
    end
  end
  
  // Initialize local SDF memory from file
  initial begin
    string mem_file;
    if ($value$plusargs("SDF_LOCAL_MEM_FILE=%s", mem_file)) begin
      $display("[SdfMem] Loading local SDF memory from %s", mem_file);
      $readmemh(mem_file, sdf_local_mem);
      local_mem_loaded = 1'b1;
    end else begin
      $display("[SdfMem] Warning: SDF_LOCAL_MEM_FILE not specified, using empty memory");
    end
  end
  
  // Check if we should access local SDF (based on meta information)
  // Simplified: assume local_idx bit 31 indicates local access
  function logic is_local_access;
    input logic [ADDR_WIDTH-1:0] gidx;
    begin
      // Simplified heuristic: if global index has bit 31 set, use local
      is_local_access = gidx[31];
    end
  endfunction
  
  // Pipeline stage 0: capture inputs
  always_ff @(posedge clk) begin
    if (reset) begin
      gidx_pipe[0] <= '0;
      lidx_pipe[0] <= '0;
      vld_pipe[0] <= 1'b0;
    end else begin
      gidx_pipe[0] <= globalIdx;
      lidx_pipe[0] <= localIdx;
      vld_pipe[0] <= en;
    end
  end
  
  // Pipeline stage 1: read from memory
  logic [31:0] data_s1;
  logic        valid_s1;
  
  always_ff @(posedge clk) begin
    if (reset) begin
      data_s1 <= '0;
      valid_s1 <= 1'b0;
    end else if (en) begin
      // Simplified addressing logic
      // In real hardware, this uses meta to determine local/global
      // For simulation, we use a simple heuristic
      if (global_mem_loaded && !is_local_access(gidx_pipe[0])) begin
        if (gidx_pipe[0] < MAX_GLOBAL_ENTRIES) begin
          data_s1 <= sdf_global_mem[gidx_pipe[0]];
          valid_s1 <= 1'b1;
        end else begin
          data_s1 <= '0;
          valid_s1 <= 1'b0;
        end
      end else if (local_mem_loaded && is_local_access(gidx_pipe[0])) begin
        // Use localIdx for local SDF access
        if (lidx_pipe[0] < MAX_LOCAL_ENTRIES) begin
          data_s1 <= sdf_local_mem[lidx_pipe[0]];
          valid_s1 <= 1'b1;
        end else begin
          data_s1 <= '0;
          valid_s1 <= 1'b0;
        end
      end else begin
        data_s1 <= '0;
        valid_s1 <= 1'b0;
      end
    end else begin
      data_s1 <= '0;
      valid_s1 <= 1'b0;
    end
  end
  
  // Pipeline stage 2: output
  always_ff @(posedge clk) begin
    if (reset) begin
      data <= '0;
      valid <= 1'b0;
    end else begin
      data <= data_s1[DATA_WIDTH-1:0];
      valid <= valid_s1;
    end
  end

endmodule
