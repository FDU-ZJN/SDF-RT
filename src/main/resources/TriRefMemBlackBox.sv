module TriRefMemResourceBB #(
  parameter int ADDR_WIDTH = 32,
  parameter int DATA_WIDTH = 256,
  parameter int LATENCY = 1,
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
  localparam int FIXED_LATENCY = 1;
  logic [DATA_WIDTH-1:0] data_pipe_a [0:FIXED_LATENCY-1];
  logic [DATA_WIDTH-1:0] data_pipe_b [0:FIXED_LATENCY-1];
  logic [ADDR_WIDTH-1:0] addr_pipe_a [0:FIXED_LATENCY-1];
  logic [ADDR_WIDTH-1:0] addr_pipe_b [0:FIXED_LATENCY-1];
  logic [FIXED_LATENCY-1:0] valid_pipe_a;
  logic [FIXED_LATENCY-1:0] valid_pipe_b;
  integer i;

  reg [DATA_WIDTH-1:0] tri_ref_mem [0:MAX_ENTRIES-1];

  initial begin
    if (LATENCY != FIXED_LATENCY) begin
      $warning("[TriRefMem] LATENCY=%0d is ignored, fixed latency is %0d", LATENCY, FIXED_LATENCY);
    end

    string mem_file;
    if ($value$plusargs("TRI_REF_MEM_FILE=%s", mem_file)) begin
      $display("[TriRefMem] Loading triangle ref memory from %s", mem_file);
      $readmemh(mem_file, tri_ref_mem);
    end else begin
      $display("[TriRefMem] Warning: TRI_REF_MEM_FILE not specified, using empty memory");
    end
  end

  always_ff @(posedge clk) begin
    if (reset) begin
      valid_pipe_a <= '0;
      valid_pipe_b <= '0;
      for (i = 0; i < FIXED_LATENCY; i = i + 1) begin
        data_pipe_a[i] <= '0;
        data_pipe_b[i] <= '0;
        addr_pipe_a[i] <= '0;
        addr_pipe_b[i] <= '0;
      end
    end else begin
      valid_pipe_a[0] <= en_a;
      valid_pipe_b[0] <= en_b;
      data_pipe_a[0] <= (en_a && addr_a < MAX_ENTRIES) ? tri_ref_mem[addr_a] : '0;
      data_pipe_b[0] <= (en_b && addr_b < MAX_ENTRIES) ? tri_ref_mem[addr_b] : '0;
      addr_pipe_a[0] <= addr_a;
      addr_pipe_b[0] <= addr_b;
    end
  end

  assign data_a = data_pipe_a[FIXED_LATENCY - 1];
  assign valid_a = valid_pipe_a[FIXED_LATENCY - 1];
  assign addr_q_a = addr_pipe_a[FIXED_LATENCY - 1];
  assign data_b = data_pipe_b[FIXED_LATENCY - 1];
  assign valid_b = valid_pipe_b[FIXED_LATENCY - 1];
  assign addr_q_b = addr_pipe_b[FIXED_LATENCY - 1];
endmodule
