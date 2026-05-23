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
  output logic [ADDR_WIDTH-1:0]  addr_q
);
  logic [LATENCY-1:0]    valid_pipe;
  logic [ADDR_WIDTH-1:0] addr_pipe [LATENCY-1:0];
  logic [DATA_WIDTH-1:0] data_pipe [LATENCY-1:0];
  localparam int MEM_ADDR_WIDTH = $clog2(MAX_ENTRIES);
  localparam int WORDS_PER_NORMAL = 3;
  localparam int NORMAL_MEM_WORDS = MAX_ENTRIES * WORDS_PER_NORMAL;
  localparam int NORMAL_WORD_ADDR_WIDTH = $clog2(NORMAL_MEM_WORDS);
  localparam logic [ADDR_WIDTH-1:0] MAX_ENTRIES_ADDR = MAX_ENTRIES[ADDR_WIDTH-1:0];
  logic [MEM_ADDR_WIDTH-1:0] mem_addr;
  logic [NORMAL_WORD_ADDR_WIDTH-1:0] mem_addr_word;
  logic [NORMAL_WORD_ADDR_WIDTH-1:0] word_base;
  reg   [31:0] normal_mem_words [0:NORMAL_MEM_WORDS-1];
  integer i;
  string normal_mem_file;

  initial begin
    if ($value$plusargs("NORMAL_MEM_FILE=%s", normal_mem_file)) begin
      $display("[NormalMem] Loading original normal memory from %s", normal_mem_file);
      $readmemh(normal_mem_file, normal_mem_words);
    end else begin
      $display("[NormalMem] Warning: NORMAL_MEM_FILE not specified, using empty memory");
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

      if (en) begin
        if (addr < MAX_ENTRIES_ADDR) begin
          data_pipe[0] <= {
            normal_mem_words[word_base + 2],
            normal_mem_words[word_base + 1],
            normal_mem_words[word_base]
          };
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
  assign mem_addr = addr[MEM_ADDR_WIDTH-1:0];
  assign mem_addr_word = {{(NORMAL_WORD_ADDR_WIDTH - MEM_ADDR_WIDTH){1'b0}}, mem_addr};
  assign word_base = mem_addr_word + (mem_addr_word << 1);
endmodule
