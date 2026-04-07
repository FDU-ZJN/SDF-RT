module BVHMemResourceBB #(
  parameter int ADDR_WIDTH = 32,
  parameter int DATA_WIDTH = 256,
  parameter int LATENCY = 1
) (
  input  logic                   clk,
  input  logic                   reset,
  input  logic [ADDR_WIDTH-1:0]  addr,
  input  logic                   en,
  output logic [DATA_WIDTH-1:0]  data,
  output logic                   valid,
  output logic [ADDR_WIDTH-1:0]  addr_q
);
  logic [DATA_WIDTH-1:0] data_pipe   [0:LATENCY-1];
  logic [ADDR_WIDTH-1:0] addr_pipe   [0:LATENCY-1];
  logic [LATENCY-1:0]    valid_pipe;
  integer i;
  
  // Memory storage for $readmemh initialization
  // DATA_WIDTH = 256 bits = 8 words (6 floats bounds + 4 int32 node info, but packed to 8 words)
  localparam int NUM_WORDS = DATA_WIDTH / 32; // = 8
  localparam int MAX_ENTRIES = 65536; // 2^16 BVH nodes
  
  reg [31:0] bvh_mem [0:MAX_ENTRIES-1][0:NUM_WORDS-1];
  reg mem_loaded = 1'b0;
  
  // Initialize memory from file using $readmemh
  initial begin
    string mem_file;
    if ($value$plusargs("BVH_MEM_FILE=%s", mem_file)) begin
      $display("[BVHMem] Loading BVH memory from %s", mem_file);
      $readmemh(mem_file, bvh_mem);
      mem_loaded = 1'b1;
    end else begin
      $display("[BVHMem] Warning: BVH_MEM_FILE not specified, using empty memory");
    end
  end

  always_ff @(posedge clk) begin
    if (reset) begin
      valid_pipe <= '0;
      for (i = 0; i < LATENCY; i = i + 1) begin
        data_pipe[i] <= '0;
        addr_pipe[i] <= '0;
      end
    end else begin
      valid_pipe[0] <= en;
      if (en) begin
        addr_pipe[0] <= addr;
        // Read from initialized memory
        if (mem_loaded && (addr < MAX_ENTRIES)) begin
          for (i = 0; i < NUM_WORDS; i = i + 1) begin
            data_pipe[0][i*32 +: 32] <= bvh_mem[addr][i];
          end
        end else begin
          data_pipe[0] <= '0;
        end
      end
      for (i = 1; i < LATENCY; i = i + 1) begin
        valid_pipe[i] <= valid_pipe[i - 1];
        data_pipe[i] <= data_pipe[i - 1];
        addr_pipe[i] <= addr_pipe[i - 1];
      end
    end
  end

  assign data = data_pipe[LATENCY - 1];
  assign valid = valid_pipe[LATENCY - 1];
  assign addr_q = addr_pipe[LATENCY - 1];
endmodule

