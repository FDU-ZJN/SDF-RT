module TriangleMemResourceBB #(
  parameter int ADDR_WIDTH = 32,
  parameter int DATA_WIDTH = 1152,
  parameter int LATENCY = 2
) (
  input  logic                   clk,
  input  logic                   reset,
  input  logic [ADDR_WIDTH-1:0]  addr,
  input  logic                   en,
  output logic [DATA_WIDTH-1:0]  data,
  output logic                   valid,
  output logic [ADDR_WIDTH-1:0]  addr_q
);
  logic [LATENCY-1:0]             valid_pipe;
  logic [ADDR_WIDTH-1:0]          addr_pipe [LATENCY-1:0];
  integer i;
  
  // Memory storage for $readmemh initialization
  // DATA_WIDTH bits = 1152 bits = 36 floats (4 PEs * 9 floats/tri)
  localparam int NUM_FLOATS = DATA_WIDTH / 32;
  localparam int MAX_ENTRIES = 4096; // Adjustable based on needs
  
  reg [31:0] triangle_mem [0:MAX_ENTRIES-1][0:NUM_FLOATS-1];
  reg mem_loaded = 1'b0;
  
  // Initialize memory from file using $readmemh
  initial begin
    string mem_file;
    if ($value$plusargs("TRI_MEM_FILE=%s", mem_file)) begin
      $display("[TriangleMem] Loading triangle memory from %s", mem_file);
      $readmemh(mem_file, triangle_mem);
      mem_loaded = 1'b1;
    end else begin
      $display("[TriangleMem] Warning: TRI_MEM_FILE not specified, using empty memory");
    end
  end

  always_ff @(posedge clk) begin
    if (reset) begin
      valid_pipe <= '0;
      for (i = 0; i < LATENCY; i = i + 1)
        addr_pipe[i] <= '0;
    end else begin
      valid_pipe[0]  <= en;
      addr_pipe[0]   <= addr;
      for (i = 1; i < LATENCY; i = i + 1) begin
        valid_pipe[i] <= valid_pipe[i-1];
        addr_pipe[i]  <= addr_pipe[i-1];
      end
    end
  end

  // Read from initialized memory
  always_ff @(posedge clk) begin
    if (en) begin
      if (mem_loaded && (addr < MAX_ENTRIES)) begin
        for (i = 0; i < NUM_FLOATS; i = i + 1) begin
          data[i*32 +: 32] <= triangle_mem[addr][i];
        end
      end else begin
        data <= '0;
      end
    end else begin
      data <= '0;
    end
  end

  assign valid  = valid_pipe[LATENCY-1];
  assign addr_q = addr_pipe[LATENCY-1];

endmodule

