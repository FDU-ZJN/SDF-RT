module NormalMemResourceBB #(
  parameter int ADDR_WIDTH = 16,
  parameter int DATA_WIDTH = 96,
  parameter int LATENCY = 2,
  parameter int MAX_ENTRIES = 8554
) (
  input  logic                   clk,
  input  logic                   reset,
  input  logic [ADDR_WIDTH-1:0]  addr,
  input  logic                   en,
  output logic [DATA_WIDTH-1:0]  data,
  output logic                   valid,
  output logic [ADDR_WIDTH-1:0]  addr_q,
  input  logic                   wr_en,
  input  logic [31:0]            wr_addr,
  input  logic [31:0]            wr_data
);
  localparam int WORDS_PER_NORMAL = 4;
  localparam int NORMAL_MEM_WORDS = MAX_ENTRIES * WORDS_PER_NORMAL;

  logic [LATENCY-1:0]    valid_pipe;
  logic [ADDR_WIDTH-1:0] addr_pipe [LATENCY-1:0];
  logic [DATA_WIDTH-1:0] data_pipe [LATENCY-1:0];
  reg   [DATA_WIDTH-1:0] normal_mem [0:MAX_ENTRIES-1];
  integer i;

  initial begin
    string mem_file;
    if ($value$plusargs("NORMAL_MEM_FILE=%s", mem_file)) begin
      $display("[NormalMem] Loading original normal memory from %s", mem_file);
      $readmemh(mem_file, normal_mem);
    end else begin
      $display("[NormalMem] NORMAL_MEM_FILE not specified, expecting bus writes");
    end
  end

  always_ff @(posedge clk) begin
    if (reset) begin
      valid_pipe <= '0;
      for (i = 0; i < LATENCY; i = i + 1) begin
        addr_pipe[i] <= '0;
        data_pipe[i] <= '0;
      end
    end else begin
      valid_pipe[0] <= en;
      addr_pipe[0]  <= addr;

      if (wr_en && (wr_addr < NORMAL_MEM_WORDS) && (wr_addr[1:0] != 2'b11)) begin
        normal_mem[wr_addr[31:2]][wr_addr[1:0] * 32 +: 32] <= wr_data;
      end

      if (en) begin
        if (addr < MAX_ENTRIES) begin
          data_pipe[0] <= normal_mem[addr];
        end else begin
          data_pipe[0] <= '0;
        end
      end else begin
        data_pipe[0] <= '0;
      end

      for (i = 1; i < LATENCY; i = i + 1) begin
        valid_pipe[i] <= valid_pipe[i - 1];
        addr_pipe[i]  <= addr_pipe[i - 1];
        data_pipe[i]  <= data_pipe[i - 1];
      end
    end
  end

  assign data   = data_pipe[LATENCY - 1];
  assign valid  = valid_pipe[LATENCY - 1];
  assign addr_q = addr_pipe[LATENCY - 1];
endmodule
