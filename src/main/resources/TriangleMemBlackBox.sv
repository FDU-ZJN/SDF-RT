module TriangleMemResourceBB #(
  parameter int ADDR_WIDTH = 32,
  parameter int DATA_WIDTH = 1152,
  parameter int LATENCY = 2,
  parameter int NUM_PES = 4,
  parameter int MAX_ENTRIES = 65536
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
  logic [LATENCY-1:0]             valid_pipe;
  logic [ADDR_WIDTH-1:0]          addr_pipe [LATENCY-1:0];
  logic [DATA_WIDTH-1:0]          data_pipe [LATENCY-1:0];
  logic [NUM_PES-1:0]             mask_pipe [LATENCY-1:0];
  integer i;
  logic [ADDR_WIDTH-1:0]          tri_addr;
    assign tri_addr = addr>>2;  // addr is already aligned

  assign req_ready = 1'b1;

  // Memory storage for $readmemh initialization
  // DATA_WIDTH bits = 1152 bits = 36 floats (4 PEs * 9 floats/tri)
  localparam int NUM_FLOATS = DATA_WIDTH / 32;

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
      for (i = 0; i < LATENCY; i = i + 1) begin
        addr_pipe[i] <= '0;
        data_pipe[i] <= '0;
        mask_pipe[i] <= '0;
      end
    end else begin
      valid_pipe[0]  <= req_valid;
      addr_pipe[0]   <= addr;
      if (req_valid) begin
        if (mem_loaded && (tri_addr < MAX_ENTRIES)) begin
          for (i = 0; i < NUM_FLOATS; i = i + 1) begin
            data_pipe[0][i*32 +: 32] <= triangle_mem[tri_addr][i];
          end
          mask_pipe[0] <= req_mask;
        end else begin
          data_pipe[0] <= '0;
          mask_pipe[0] <= '0;
        end
      end else begin
        data_pipe[0] <= '0;
        mask_pipe[0] <= '0;
      end
      for (i = 1; i < LATENCY; i = i + 1) begin
        valid_pipe[i] <= valid_pipe[i-1];
        addr_pipe[i]  <= addr_pipe[i-1];
        data_pipe[i]  <= data_pipe[i-1];
        mask_pipe[i]  <= mask_pipe[i-1];
      end
    end
  end

  assign data       = data_pipe[LATENCY-1];
  assign valid      = valid_pipe[LATENCY-1];
  assign valid_mask = mask_pipe[LATENCY-1];
  assign addr_q     = addr_pipe[LATENCY-1];

endmodule

