module NormalMemResourceBB #(
  parameter int ADDR_WIDTH = 16,
  parameter int DATA_WIDTH = 96,
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
  logic [LATENCY-1:0]    valid_pipe;
  logic [ADDR_WIDTH-1:0] addr_pipe [LATENCY-1:0];
  integer i;
  
  // Memory storage for $readmemh initialization
  // DATA_WIDTH = 96 bits = 3 floats (normal x, y, z)
  localparam int NUM_FLOATS = DATA_WIDTH / 32; // = 3
  localparam int MAX_ENTRIES = 65536; // 2^16 addresses
  
  reg [31:0] normal_mem [0:MAX_ENTRIES-1][0:NUM_FLOATS-1];
  reg mem_loaded = 1'b0;
  
  // Initialize memory from file using $readmemh
  initial begin
    string mem_file;
    if ($value$plusargs("NORMAL_MEM_FILE=%s", mem_file)) begin
      $display("[NormalMem] Loading normal memory from %s", mem_file);
      $readmemh(mem_file, normal_mem);
      mem_loaded = 1'b1;
    end else begin
      $display("[NormalMem] Warning: NORMAL_MEM_FILE not specified, using empty memory");
    end
  end

  always_ff @(posedge clk) begin
    if (reset) begin
      valid_pipe <= '0;
      for (i = 0; i < LATENCY; i = i + 1) begin
        addr_pipe[i] <= '0;
      end
    end else begin
      valid_pipe[0] <= en;
      addr_pipe[0] <= addr;
      for (i = 1; i < LATENCY; i = i + 1) begin
        valid_pipe[i] <= valid_pipe[i - 1];
        addr_pipe[i]  <= addr_pipe[i - 1];
      end
    end
  end
  
  // Read from initialized memory
  always_ff @(posedge clk) begin
    if (en) begin
      if (mem_loaded && (addr < MAX_ENTRIES)) begin
        for (i = 0; i < NUM_FLOATS; i = i + 1) begin
          data[i*32 +: 32] <= normal_mem[addr][i];
        end
      end else begin
        data <= '0;
      end
    end else begin
      data <= '0;
    end
  end

  assign valid  = valid_pipe[LATENCY - 1];
  assign addr_q = addr_pipe[LATENCY - 1];
endmodule