module TriRefMemResourceBB #(
  parameter int ADDR_WIDTH = 32,
  parameter int DATA_WIDTH = 256,
  parameter int LATENCY = 3,
  parameter int MAX_ENTRIES = 1
) (
  input  logic                   clk,
  input  logic                   reset,
  input  logic [ADDR_WIDTH-1:0]  addr_a,
  input  logic                   en_a,
  output logic [DATA_WIDTH-1:0]  data_a,
  output logic                   valid_a,
  output logic [ADDR_WIDTH-1:0]  addr_q_a,
  input  logic [ADDR_WIDTH-1:0]  addr_b,
  input  logic                   en_b,
  output logic [DATA_WIDTH-1:0]  data_b,
  output logic                   valid_b,
  output logic [ADDR_WIDTH-1:0]  addr_q_b
);
  localparam int MEM_ADDR_WIDTH = (MAX_ENTRIES <= 1) ? 1 : $clog2(MAX_ENTRIES);
  localparam int MEM_DEPTH = 1 << MEM_ADDR_WIDTH;
  localparam int FIXED_LATENCY = LATENCY;
  localparam int IP_READ_LATENCY = FIXED_LATENCY - 1;

  logic [MEM_ADDR_WIDTH-1:0] mem_addr_a;
  logic [MEM_ADDR_WIDTH-1:0] mem_addr_b;
  logic [MEM_ADDR_WIDTH-1:0] mem_addr_a_reg;
  logic [MEM_ADDR_WIDTH-1:0] mem_addr_b_reg;
  logic [DATA_WIDTH-1:0] data_raw_a;
  logic [DATA_WIDTH-1:0] data_raw_b;
  logic [DATA_WIDTH-1:0] data_pipe_a [0:IP_READ_LATENCY-1];
  logic [DATA_WIDTH-1:0] data_pipe_b [0:IP_READ_LATENCY-1];
  logic [FIXED_LATENCY-1:0] valid_pipe_a;
  logic [FIXED_LATENCY-1:0] valid_pipe_b;
  logic [ADDR_WIDTH-1:0] addr_pipe_a [0:FIXED_LATENCY-1];
  logic [ADDR_WIDTH-1:0] addr_pipe_b [0:FIXED_LATENCY-1];
  integer i;

  logic [DATA_WIDTH-1:0] tri_ref_mem [0:MEM_DEPTH-1];
  string tri_ref_mem_file;

  initial begin
    for (i = 0; i < MEM_DEPTH; i = i + 1) begin
      tri_ref_mem[i] = '0;
    end
    if ($value$plusargs("TRI_REF_MEM_FILE=%s", tri_ref_mem_file)) begin
      $display("[TriRefMem] Loading triangle ref memory from %s", tri_ref_mem_file);
      $readmemh(tri_ref_mem_file, tri_ref_mem);
    end else begin
      $display("[TriRefMem] Warning: TRI_REF_MEM_FILE not specified, using empty memory");
    end
  end

  assign mem_addr_a = addr_a[MEM_ADDR_WIDTH-1:0];
  assign mem_addr_b = addr_b[MEM_ADDR_WIDTH-1:0];

  always_ff @(posedge clk) begin
    mem_addr_a_reg <= mem_addr_a;
    mem_addr_b_reg <= mem_addr_b;
  end

  always_ff @(posedge clk) begin
    data_pipe_a[0] <= tri_ref_mem[mem_addr_a_reg];
    data_pipe_b[0] <= tri_ref_mem[mem_addr_b_reg];
    for (i = 1; i < IP_READ_LATENCY; i = i + 1) begin
      data_pipe_a[i] <= data_pipe_a[i-1];
      data_pipe_b[i] <= data_pipe_b[i-1];
    end
  end

  assign data_raw_a = data_pipe_a[IP_READ_LATENCY-1];
  assign data_raw_b = data_pipe_b[IP_READ_LATENCY-1];

  always_ff @(posedge clk) begin
    if (reset) begin
      valid_pipe_a <= '0;
      valid_pipe_b <= '0;
      for (i = 0; i < FIXED_LATENCY; i = i + 1) begin
        addr_pipe_a[i] <= '0;
        addr_pipe_b[i] <= '0;
      end
    end else begin
      valid_pipe_a[0] <= en_a;
      valid_pipe_b[0] <= en_b;
      addr_pipe_a[0] <= addr_a;
      addr_pipe_b[0] <= addr_b;
      for (i = 1; i < FIXED_LATENCY; i = i + 1) begin
        valid_pipe_a[i] <= valid_pipe_a[i-1];
        valid_pipe_b[i] <= valid_pipe_b[i-1];
        addr_pipe_a[i] <= addr_pipe_a[i-1];
        addr_pipe_b[i] <= addr_pipe_b[i-1];
      end
    end
  end

  assign data_a = data_raw_a;
  assign valid_a = valid_pipe_a[FIXED_LATENCY - 1];
  assign addr_q_a = addr_pipe_a[FIXED_LATENCY - 1];
  assign data_b = data_raw_b;
  assign valid_b = valid_pipe_b[FIXED_LATENCY - 1];
  assign addr_q_b = addr_pipe_b[FIXED_LATENCY - 1];
endmodule
