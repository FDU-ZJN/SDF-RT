module TriCacheDataArray #(
  parameter int DATA_WIDTH = 288,
  parameter int DEPTH = 256,
  parameter int ADDR_WIDTH = (DEPTH <= 1) ? 1 : $clog2(DEPTH)
) (
  input  logic                  clk,
  input  logic                  reset,
  input  logic                  rd_en,
  input  logic [ADDR_WIDTH-1:0] rd_addr,
  output logic [DATA_WIDTH-1:0] rd_data,
  input  logic                  wr_en,
  input  logic [ADDR_WIDTH-1:0] wr_addr,
  input  logic [DATA_WIDTH-1:0] wr_data
);

  logic                  mem_en;
  logic [ADDR_WIDTH-1:0] mem_addr;

  assign mem_en = rd_en | wr_en;
  assign mem_addr = wr_en ? wr_addr : rd_addr;

  xpm_memory_spram #(
    .ADDR_WIDTH_A       (ADDR_WIDTH),
    .MEMORY_SIZE        (DEPTH * DATA_WIDTH),
    .WRITE_DATA_WIDTH_A (DATA_WIDTH),
    .READ_DATA_WIDTH_A  (DATA_WIDTH),
    .BYTE_WRITE_WIDTH_A (DATA_WIDTH),
    .READ_LATENCY_A     (1),
    .MEMORY_PRIMITIVE   ("block"),
    .WRITE_MODE_A       ("read_first"),
    .ECC_MODE           ("no_ecc"),
    .AUTO_SLEEP_TIME    (0),
    .CASCADE_HEIGHT     (0),
    .SIM_ASSERT_CHK     (0),
    .USE_MEM_INIT       (0),
    .WAKEUP_TIME        ("disable_sleep"),
    .RST_MODE_A         ("SYNC"),
    .READ_RESET_VALUE_A ("0")
  ) cache_bram_inst (
    .sleep          (1'b0),
    .clka           (clk),
    .rsta           (reset),
    .ena            (mem_en),
    .regcea         (1'b1),
    .addra          (mem_addr),
    .dina           (wr_data),
    .wea            (wr_en),
    .injectsbiterra (1'b0),
    .injectdbiterra (1'b0),
    .douta          (rd_data),
    .sbiterra       (),
    .dbiterra       ()
  );

endmodule
