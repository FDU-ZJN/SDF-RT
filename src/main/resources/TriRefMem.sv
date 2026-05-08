module TriRefMem #(
  parameter int ADDR_WIDTH = 32,
  parameter int DATA_WIDTH = 256,
  parameter int LATENCY = 1,
  parameter int MAX_ENTRIES = 1,
  parameter string INIT_FILE = "triangle_ref_mem.mem"
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
  localparam int MEM_SIZE_BITS = MAX_ENTRIES * DATA_WIDTH;

  logic [MEM_ADDR_WIDTH-1:0] mem_addr_a;
  logic [MEM_ADDR_WIDTH-1:0] mem_addr_b;
  logic [DATA_WIDTH-1:0] data_raw_a;
  logic [DATA_WIDTH-1:0] data_raw_b;
  logic [LATENCY-1:0] valid_pipe_a;
  logic [LATENCY-1:0] valid_pipe_b;
  logic [ADDR_WIDTH-1:0] addr_pipe_a [0:LATENCY-1];
  logic [ADDR_WIDTH-1:0] addr_pipe_b [0:LATENCY-1];
  integer i;


  assign mem_addr_a = addr_a[MEM_ADDR_WIDTH-1:0];
  assign mem_addr_b = addr_b[MEM_ADDR_WIDTH-1:0];

  xpm_memory_dprom #(
    .ADDR_WIDTH_A       (MEM_ADDR_WIDTH),
    .ADDR_WIDTH_B       (MEM_ADDR_WIDTH),
    .MEMORY_SIZE        (MEM_SIZE_BITS),
    .MEMORY_PRIMITIVE   ("block"),
    .MEMORY_INIT_FILE   (INIT_FILE),
    .MEMORY_INIT_PARAM  (""),
    .USE_MEM_INIT       (1),
    .READ_DATA_WIDTH_A  (DATA_WIDTH),
    .READ_DATA_WIDTH_B  (DATA_WIDTH),
    .READ_LATENCY_A     (LATENCY),
    .READ_LATENCY_B     (LATENCY),
    .ECC_MODE           ("no_ecc"),
    .AUTO_SLEEP_TIME    (0),
    .CASCADE_HEIGHT     (5),
    .SIM_ASSERT_CHK     (0),
    .WAKEUP_TIME        ("disable_sleep"),
    .READ_RESET_VALUE_A ("0"),
    .READ_RESET_VALUE_B ("0"),
    .RST_MODE_A         ("SYNC"),
    .RST_MODE_B         ("SYNC")
  ) trirefmem_xpm_inst (
    .sleep          (1'b0),
    .clka           (clk),
    .rsta           (reset),
    .ena            (en_a),
    .regcea         (1'b1),
    .addra          (mem_addr_a),
    .douta          (data_raw_a),
    .clkb           (clk),
    .rstb           (reset),
    .enb            (en_b),
    .regceb         (1'b1),
    .addrb          (mem_addr_b),
    .doutb          (data_raw_b)
  );

  always_ff @(posedge clk) begin
    if (reset) begin
      valid_pipe_a <= '0;
      valid_pipe_b <= '0;
      for (i = 0; i < LATENCY; i = i + 1) begin
        addr_pipe_a[i] <= '0;
        addr_pipe_b[i] <= '0;
      end
    end else begin
      valid_pipe_a[0] <= en_a;
      valid_pipe_b[0] <= en_b;
      addr_pipe_a[0] <= addr_a;
      addr_pipe_b[0] <= addr_b;
      for (i = 1; i < LATENCY; i = i + 1) begin
        valid_pipe_a[i] <= valid_pipe_a[i-1];
        valid_pipe_b[i] <= valid_pipe_b[i-1];
        addr_pipe_a[i] <= addr_pipe_a[i-1];
        addr_pipe_b[i] <= addr_pipe_b[i-1];
      end
    end
  end

  assign data_a = data_raw_a;
  assign valid_a = valid_pipe_a[LATENCY - 1];
  assign addr_q_a = addr_pipe_a[LATENCY - 1];
  assign data_b = data_raw_b;
  assign valid_b = valid_pipe_b[LATENCY - 1];
  assign addr_q_b = addr_pipe_b[LATENCY - 1];

endmodule
