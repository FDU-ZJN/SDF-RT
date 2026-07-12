module TriMemRefillLineArrayBB #(
  parameter int DATA_WIDTH = 1152,
  parameter int DEPTH = 4,
  parameter int ADDR_WIDTH = (DEPTH <= 1) ? 1 : $clog2(DEPTH)
) (
  input  logic                  clk,
  input  logic                  wr_en,
  input  logic [ADDR_WIDTH-1:0] wr_addr,
  input  logic [DATA_WIDTH-1:0] wr_data,
  input  logic                  rd_en,
  input  logic [ADDR_WIDTH-1:0] rd_addr,
  output logic [DATA_WIDTH-1:0] rd_data
);
  xpm_memory_sdpram #(
    .ADDR_WIDTH_A       (ADDR_WIDTH),
    .ADDR_WIDTH_B       (ADDR_WIDTH),
    .AUTO_SLEEP_TIME    (0),
    .BYTE_WRITE_WIDTH_A (DATA_WIDTH),
    .CLOCKING_MODE      ("common_clock"),
    .ECC_MODE           ("no_ecc"),
    .MEMORY_INIT_FILE   ("none"),
    .MEMORY_INIT_PARAM  ("0"),
    .MEMORY_OPTIMIZATION("true"),
    .MEMORY_PRIMITIVE   ("block"),
    .MEMORY_SIZE        (DEPTH * DATA_WIDTH),
    .MESSAGE_CONTROL    (0),
    .READ_DATA_WIDTH_B  (DATA_WIDTH),
    .READ_LATENCY_B     (1),
    .READ_RESET_VALUE_B ("0"),
    .RST_MODE_A         ("no_reset"),
    .RST_MODE_B         ("no_reset"),
    .USE_EMBEDDED_CONSTRAINT(0),
    .USE_MEM_INIT       (0),
    .WAKEUP_TIME        ("disable_sleep"),
    .WRITE_DATA_WIDTH_A (DATA_WIDTH),
    .WRITE_MODE_B       ("read_first")
  ) refill_line_bram_inst (
    .clka           (clk),
    .ena            (wr_en),
    .wea            (wr_en),
    .addra          (wr_addr),
    .dina           (wr_data),
    .injectsbiterra (1'b0),
    .injectdbiterra (1'b0),
    .clkb           (clk),
    .enb            (rd_en),
    .addrb          (rd_addr),
    .rstb           (1'b0),
    .regceb         (1'b1),
    .doutb          (rd_data),
    .sbiterrb       (),
    .dbiterrb       (),
    .sleep          (1'b0)
  );
endmodule
