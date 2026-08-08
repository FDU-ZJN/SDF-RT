set design_name design_1
create_bd_design $design_name
current_bd_design $design_name

set ps [create_bd_cell -type ip -vlnv xilinx.com:ip:zynq_ultra_ps_e:3.5 zynq_ultra_ps_e_0]
apply_bd_automation -rule xilinx.com:bd_rule:zynq_ultra_ps_e \
  -config {apply_board_preset "1"} $ps
set_property -dict [list \
  CONFIG.PSU__CRL_APB__PL0_REF_CTRL__FREQMHZ {200} \
  CONFIG.PSU__USE__M_AXI_GP0 {1} \
  CONFIG.PSU__USE__M_AXI_GP1 {0} \
  CONFIG.PSU__MAXIGP0__DATA_WIDTH {128} \
  CONFIG.PSU__USE__S_AXI_GP2 {1} \
  CONFIG.PSU__SAXIGP2__DATA_WIDTH {128} \
] $ps

set rst [create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 rst_ps8_0_96M]

set fpga_top [create_bd_cell -type ip -vlnv user.org:user:fpga_top:2.3 fpga_top_0]

set control_smc [create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect:1.0 axi_smc]
set_property -dict [list CONFIG.NUM_MI {2} CONFIG.NUM_SI {1}] $control_smc

set memory_smc [create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect:1.0 axi_smc_1]
set_property -dict [list CONFIG.NUM_MI {1} CONFIG.NUM_SI {3}] $memory_smc

set tri_datamover [create_bd_cell -type ip -vlnv xilinx.com:ip:axi_datamover:5.1 tri_datamover]
set_property -dict [list \
  CONFIG.c_addr_width {40} \
  CONFIG.c_enable_mm2s {1} \
  CONFIG.c_enable_s2mm {0} \
  CONFIG.c_include_mm2s {Full} \
  CONFIG.c_include_mm2s_dre {0} \
  CONFIG.c_m_axi_mm2s_data_width {128} \
  CONFIG.c_m_axi_mm2s_id_width {2} \
  CONFIG.c_m_axis_mm2s_tdata_width {128} \
  CONFIG.c_mm2s_btt_used {8} \
  CONFIG.c_mm2s_burst_size {4} \
  CONFIG.c_mm2s_include_sf {1} \
] $tri_datamover

set ray_dma [create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma:7.1 ray_dma]
set_property -dict [list \
  CONFIG.c_include_mm2s {1} \
  CONFIG.c_include_s2mm {1} \
  CONFIG.c_include_sg {0} \
  CONFIG.c_sg_length_width {26} \
  CONFIG.c_m_axi_mm2s_data_width {256} \
  CONFIG.c_m_axi_s2mm_data_width {128} \
  CONFIG.c_m_axis_mm2s_tdata_width {256} \
  CONFIG.c_mm2s_burst_size {16} \
  CONFIG.c_s2mm_burst_size {16} \
  CONFIG.c_s_axis_s2mm_tdata_width {128} \
] $ray_dma

connect_bd_intf_net [get_bd_intf_pins $ps/M_AXI_HPM0_FPD] \
                    [get_bd_intf_pins $control_smc/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins $control_smc/M00_AXI] \
                    [get_bd_intf_pins $fpga_top/s_axi]
connect_bd_intf_net [get_bd_intf_pins $control_smc/M01_AXI] \
                    [get_bd_intf_pins $ray_dma/S_AXI_LITE]

connect_bd_intf_net [get_bd_intf_pins $tri_datamover/M_AXI_MM2S] \
                    [get_bd_intf_pins $memory_smc/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins $ray_dma/M_AXI_MM2S] \
                    [get_bd_intf_pins $memory_smc/S01_AXI]
connect_bd_intf_net [get_bd_intf_pins $ray_dma/M_AXI_S2MM] \
                    [get_bd_intf_pins $memory_smc/S02_AXI]
connect_bd_intf_net [get_bd_intf_pins $memory_smc/M00_AXI] \
                    [get_bd_intf_pins $ps/S_AXI_HP0_FPD]

connect_bd_intf_net [get_bd_intf_pins $ray_dma/M_AXIS_MM2S] \
                    [get_bd_intf_pins $fpga_top/s_axis_ray]
connect_bd_intf_net [get_bd_intf_pins $fpga_top/m_axis_ray_result] \
                    [get_bd_intf_pins $ray_dma/S_AXIS_S2MM]
connect_bd_intf_net [get_bd_intf_pins $fpga_top/m_axis_tri_cmd] \
                    [get_bd_intf_pins $tri_datamover/S_AXIS_MM2S_CMD]
connect_bd_intf_net [get_bd_intf_pins $tri_datamover/M_AXIS_MM2S] \
                    [get_bd_intf_pins $fpga_top/s_axis_tri_data]
connect_bd_intf_net [get_bd_intf_pins $tri_datamover/M_AXIS_MM2S_STS] \
                    [get_bd_intf_pins $fpga_top/s_axis_tri_status]

set pl_clk [get_bd_pins $ps/pl_clk0]
connect_bd_net $pl_clk \
  [get_bd_pins $control_smc/aclk] \
  [get_bd_pins $memory_smc/aclk] \
  [get_bd_pins $ray_dma/s_axi_lite_aclk] \
  [get_bd_pins $ray_dma/m_axi_mm2s_aclk] \
  [get_bd_pins $ray_dma/m_axi_s2mm_aclk] \
  [get_bd_pins $rst/slowest_sync_clk] \
  [get_bd_pins $tri_datamover/m_axi_mm2s_aclk] \
  [get_bd_pins $tri_datamover/m_axis_mm2s_cmdsts_aclk] \
  [get_bd_pins $ps/maxihpm0_fpd_aclk] \
  [get_bd_pins $ps/saxihp0_fpd_aclk] \
  [get_bd_pins $fpga_top/clk]

connect_bd_net [get_bd_pins $ps/pl_resetn0] [get_bd_pins $rst/ext_reset_in]
set peripheral_resetn [get_bd_pins $rst/peripheral_aresetn]
connect_bd_net $peripheral_resetn \
  [get_bd_pins $control_smc/aresetn] \
  [get_bd_pins $memory_smc/aresetn] \
  [get_bd_pins $ray_dma/axi_resetn] \
  [get_bd_pins $tri_datamover/m_axi_mm2s_aresetn] \
  [get_bd_pins $tri_datamover/m_axis_mm2s_cmdsts_aresetn] \
  [get_bd_pins $fpga_top/rst_n]

assign_bd_address -offset 0x000400000000 -range 0x00400000 \
  -target_address_space [get_bd_addr_spaces $ps/Data] \
  [get_bd_addr_segs $fpga_top/s_axi/reg0] -force
assign_bd_address -offset 0xA0010000 -range 0x00010000 \
  -target_address_space [get_bd_addr_spaces $ps/Data] \
  [get_bd_addr_segs $ray_dma/S_AXI_LITE/Reg] -force

foreach address_space [list \
  [get_bd_addr_spaces $tri_datamover/Data_MM2S] \
  [get_bd_addr_spaces $ray_dma/Data_MM2S] \
  [get_bd_addr_spaces $ray_dma/Data_S2MM] \
] {
  assign_bd_address -offset 0x00000000 -range 0x80000000 \
    -target_address_space $address_space \
    [get_bd_addr_segs $ps/SAXIGP2/HP0_DDR_LOW] -force
}

foreach address_space [list \
  [get_bd_addr_spaces $tri_datamover/Data_MM2S] \
  [get_bd_addr_spaces $ray_dma/Data_MM2S] \
  [get_bd_addr_spaces $ray_dma/Data_S2MM] \
] {
  foreach segment_name {HP0_DDR_HIGH HP0_LPS_OCM HP0_QSPI} {
    set segment [get_bd_addr_segs -quiet $ps/SAXIGP2/$segment_name]
    if {[llength $segment]} {
      exclude_bd_addr_seg -target_address_space $address_space $segment
    }
  }
}

validate_bd_design
save_bd_design
puts "CREATED_BLOCK_DESIGN $design_name"
