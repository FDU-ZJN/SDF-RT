module TriangleMemResourceBB #(
  parameter int ADDR_WIDTH = 32,
  parameter int DATA_WIDTH = 288,
  parameter int LATENCY = 2,
  parameter int NUM_PES = 1,
  parameter int BANK_ID = 0,
  parameter int NUM_BANKS = 1,
  parameter int MAX_ENTRIES = 1
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
  logic [LATENCY-1:0]         valid_pipe;
  logic [ADDR_WIDTH-1:0]      addr_pipe [LATENCY-1:0];
  logic [DATA_WIDTH-1:0]      data_pipe [LATENCY-1:0];
  logic [NUM_PES-1:0]         mask_pipe [LATENCY-1:0];
  integer i;

  assign req_ready = 1'b1;

  reg [DATA_WIDTH-1:0] triangle_mem [0:MAX_ENTRIES-1];
  reg mem_loaded = 1'b0;

  initial begin
    string mem_file;
    case (BANK_ID)
      0: mem_loaded = $value$plusargs("TRI_MEM_BANK0_FILE=%s", mem_file);
      1: mem_loaded = $value$plusargs("TRI_MEM_BANK1_FILE=%s", mem_file);
      2: mem_loaded = $value$plusargs("TRI_MEM_BANK2_FILE=%s", mem_file);
      3: mem_loaded = $value$plusargs("TRI_MEM_BANK3_FILE=%s", mem_file);
      4: mem_loaded = $value$plusargs("TRI_MEM_BANK4_FILE=%s", mem_file);
      5: mem_loaded = $value$plusargs("TRI_MEM_BANK5_FILE=%s", mem_file);
      6: mem_loaded = $value$plusargs("TRI_MEM_BANK6_FILE=%s", mem_file);
      7: mem_loaded = $value$plusargs("TRI_MEM_BANK7_FILE=%s", mem_file);
      default: mem_loaded = $value$plusargs("TRI_MEM_FILE=%s", mem_file);
    endcase

    if (mem_loaded) begin
      $display("[TriangleMem] Loading bank %0d/%0d from %s", BANK_ID, NUM_BANKS, mem_file);
      $readmemh(mem_file, triangle_mem);
    end else begin
      $display("[TriangleMem] Warning: no memory file specified for bank %0d", BANK_ID);
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
      valid_pipe[0] <= req_valid;
      addr_pipe[0]  <= addr;
      if (req_valid) begin
        if (mem_loaded && (addr < MAX_ENTRIES)) begin
          data_pipe[0] <= triangle_mem[addr];
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
