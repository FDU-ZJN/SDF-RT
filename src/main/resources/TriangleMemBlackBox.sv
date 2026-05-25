module TriangleMemResourceBB #(
  parameter int ADDR_WIDTH = 32,
  parameter int DATA_WIDTH = 288,
  parameter int LATENCY = 3,
  parameter int NUM_PES = 1,
  parameter int BANK_ID = 0,
  parameter int NUM_BANKS = 1,
  parameter int MAX_ENTRIES = 1,
  parameter string INIT_FILE = ""
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
  localparam int MEM_ADDR_WIDTH = (MAX_ENTRIES <= 1) ? 1 : $clog2(MAX_ENTRIES);
  localparam int MEM_DEPTH = 1 << MEM_ADDR_WIDTH;
  localparam int FIXED_LATENCY = LATENCY;
  localparam int IP_READ_LATENCY = FIXED_LATENCY - 1;

  logic [MEM_ADDR_WIDTH-1:0] mem_addr;
  logic [MEM_ADDR_WIDTH-1:0] mem_addr_reg;
  logic [DATA_WIDTH-1:0]     data_raw;
  logic [DATA_WIDTH-1:0]     data_pipe [0:IP_READ_LATENCY-1];
  logic [FIXED_LATENCY-1:0]  valid_pipe;
  logic [NUM_PES-1:0]        mask_pipe [0:FIXED_LATENCY-1];
  logic [ADDR_WIDTH-1:0]     addr_pipe [0:FIXED_LATENCY-1];
  integer i;

  assign mem_addr = addr[MEM_ADDR_WIDTH-1:0];
  assign req_ready = 1'b1;

  logic [DATA_WIDTH-1:0] triangle_mem [0:MEM_DEPTH-1];
  logic mem_loaded = 1'b0;
  string triangle_mem_file;

  initial begin
    for (i = 0; i < MEM_DEPTH; i = i + 1) begin
      triangle_mem[i] = '0;
    end

    case (BANK_ID)
      0: mem_loaded = $value$plusargs("TRI_MEM_BANK0_FILE=%s", triangle_mem_file);
      1: mem_loaded = $value$plusargs("TRI_MEM_BANK1_FILE=%s", triangle_mem_file);
      2: mem_loaded = $value$plusargs("TRI_MEM_BANK2_FILE=%s", triangle_mem_file);
      3: mem_loaded = $value$plusargs("TRI_MEM_BANK3_FILE=%s", triangle_mem_file);
      4: mem_loaded = $value$plusargs("TRI_MEM_BANK4_FILE=%s", triangle_mem_file);
	      5: mem_loaded = $value$plusargs("TRI_MEM_BANK5_FILE=%s", triangle_mem_file);
	      6: mem_loaded = $value$plusargs("TRI_MEM_BANK6_FILE=%s", triangle_mem_file);
	      7: mem_loaded = $value$plusargs("TRI_MEM_BANK7_FILE=%s", triangle_mem_file);
	      8: mem_loaded = $value$plusargs("TRI_MEM_BANK8_FILE=%s", triangle_mem_file);
	      9: mem_loaded = $value$plusargs("TRI_MEM_BANK9_FILE=%s", triangle_mem_file);
	      10: mem_loaded = $value$plusargs("TRI_MEM_BANK10_FILE=%s", triangle_mem_file);
	      11: mem_loaded = $value$plusargs("TRI_MEM_BANK11_FILE=%s", triangle_mem_file);
	      12: mem_loaded = $value$plusargs("TRI_MEM_BANK12_FILE=%s", triangle_mem_file);
	      13: mem_loaded = $value$plusargs("TRI_MEM_BANK13_FILE=%s", triangle_mem_file);
	      14: mem_loaded = $value$plusargs("TRI_MEM_BANK14_FILE=%s", triangle_mem_file);
	      15: mem_loaded = $value$plusargs("TRI_MEM_BANK15_FILE=%s", triangle_mem_file);
	      default: mem_loaded = $value$plusargs("TRI_MEM_FILE=%s", triangle_mem_file);
	    endcase

    if (!mem_loaded && (INIT_FILE != "")) begin
      triangle_mem_file = INIT_FILE;
      mem_loaded = 1'b1;
    end

    if (mem_loaded) begin
      $display("[TriangleMem] Loading bank %0d/%0d from %s", BANK_ID, NUM_BANKS, triangle_mem_file);
      $readmemh(triangle_mem_file, triangle_mem);
    end else begin
      $display("[TriangleMem] Warning: no memory file specified for bank %0d", BANK_ID);
    end
  end

  always_ff @(posedge clk) begin
    mem_addr_reg <= mem_addr;
  end

  always_ff @(posedge clk) begin
    data_pipe[0] <= triangle_mem[mem_addr_reg];
    for (i = 1; i < IP_READ_LATENCY; i = i + 1) begin
      data_pipe[i] <= data_pipe[i-1];
    end
  end

  assign data_raw = data_pipe[IP_READ_LATENCY-1];

  always_ff @(posedge clk) begin
    if (reset) begin
      valid_pipe <= '0;
      for (i = 0; i < FIXED_LATENCY; i = i + 1) begin
        addr_pipe[i] <= '0;
        mask_pipe[i] <= '0;
      end
    end else begin
      valid_pipe[0] <= req_valid;
      addr_pipe[0]  <= addr;
      mask_pipe[0]  <= req_mask;
      for (i = 1; i < FIXED_LATENCY; i = i + 1) begin
        valid_pipe[i] <= valid_pipe[i-1];
        addr_pipe[i]  <= addr_pipe[i-1];
        mask_pipe[i]  <= mask_pipe[i-1];
      end
    end
  end

  assign data       = data_raw;
  assign valid      = valid_pipe[FIXED_LATENCY-1];
  assign valid_mask = mask_pipe[FIXED_LATENCY-1];
  assign addr_q     = addr_pipe[FIXED_LATENCY-1];

endmodule
