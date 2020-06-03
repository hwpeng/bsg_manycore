/**
 *  bsg_manycore_drlp_master_tile.v
 *
 */

module bsg_manycore_drlp_master_tile
  import bsg_noc_pkg::*; // { P=0, W,E,N,S }
  import bsg_manycore_pkg::*;
  #(parameter dmem_size_p = "inv"
    , parameter vcache_size_p ="inv"
    , parameter icache_entries_p = "inv"
    , parameter icache_tag_width_p = "inv"
    , parameter x_cord_width_p = "inv"
    , parameter y_cord_width_p = "inv"
    , parameter num_tiles_x_p="inv"
    
    , parameter data_width_p = "inv"
    , parameter addr_width_p = "inv"
    , parameter load_id_width_p = "inv"
    , parameter epa_byte_addr_width_p = "inv"
    , parameter dram_ch_addr_width_p = "inv"

    , parameter vcache_block_size_in_words_p="inv"
    , parameter vcache_sets_p="inv"

    , localparam dirs_lp = 4

    , parameter dram_ch_start_col_p  = 0
    , parameter stub_p = {dirs_lp{1'b0}}           // {s,n,e,w}
    , parameter repeater_output_p = {dirs_lp{1'b0}} // {s,n,e,w}
    , parameter hetero_type_p = 0
    , parameter debug_p = 0

    , parameter branch_trace_en_p = 0

    , localparam link_sif_width_lp =
    `bsg_manycore_link_sif_width(addr_width_p,data_width_p,x_cord_width_p,y_cord_width_p,
      load_id_width_p)
  )
  (
    input clk_i
    , input reset_i

    , input  [link_sif_width_lp-1:0][S:W] link_in
    , output [link_sif_width_lp-1:0][S:W] link_out

    , input [x_cord_width_p-1:0] my_x_i
    , input [y_cord_width_p-1:0] my_y_i

    
    // DRLP signals
    , input all_slave_done_i
    , input all_pe_ready_i
    , output logic [15:0] pe_data_v_o
    , output logic [32*288-1:0] all_wgt_o
    , output logic [32*16-1:0] all_bias_o
    , output logic sld_o
    , output logic [12:0] imem_r_addr_o
    , output logic dw_wgt_start_o
  );

  logic [link_sif_width_lp-1:0] proc_link_sif_li;
  logic [link_sif_width_lp-1:0] proc_link_sif_lo;

  //-------------------------------------------
  //As the manycore will distribute across large area, it will take long
  //time for the reset signal to propgate. We should register the reset
  //signal in each tile
  logic reset_r;
  always_ff @ (posedge clk_i) begin
    reset_r <= reset_i;
  end

  bsg_manycore_mesh_node #(
    .stub_p(stub_p)
    ,.x_cord_width_p(x_cord_width_p)
    ,.y_cord_width_p(y_cord_width_p)
    ,.data_width_p(data_width_p)
    ,.addr_width_p(addr_width_p)
    ,.load_id_width_p(load_id_width_p)
    ,.debug_p(debug_p)
    ,.repeater_output_p(repeater_output_p) // select buffer for this particular node
  ) rtr (
    .clk_i(clk_i)
    ,.reset_i(reset_r)
    ,.links_sif_i(link_in)
    ,.links_sif_o(link_out)
    ,.proc_link_sif_i(proc_link_sif_li)
    ,.proc_link_sif_o(proc_link_sif_lo)
    ,.my_x_i(my_x_i)
    ,.my_y_i(my_y_i)
  );

  manycore_drlp_master #(
    .x_cord_width_p(x_cord_width_p)
    ,.y_cord_width_p(y_cord_width_p)
    ,.data_width_p(data_width_p)
    ,.addr_width_p(addr_width_p)
    ,.load_id_width_p(load_id_width_p)

    ,.dmem_size_p(dmem_size_p)
    ,.vcache_size_p(vcache_size_p)
    ,.icache_entries_p(icache_entries_p)
    ,.icache_tag_width_p(icache_tag_width_p)
    ,.max_out_credits_p(200)
    ,.epa_byte_addr_width_p(epa_byte_addr_width_p)
    ,.dram_ch_addr_width_p(dram_ch_addr_width_p)
    ,.dram_ch_start_col_p(dram_ch_start_col_p)
    ,.hetero_type_p(hetero_type_p)
    ,.num_tiles_x_p(num_tiles_x_p)
    ,.vcache_block_size_in_words_p(vcache_block_size_in_words_p)
    ,.vcache_sets_p(vcache_sets_p)

    ,.branch_trace_en_p(branch_trace_en_p)

    ,.debug_p(debug_p)
  ) proc (
    .clk_i(clk_i)
    ,.reset_i(reset_r)

    ,.link_sif_i(proc_link_sif_lo)
    ,.link_sif_o(proc_link_sif_li)

    ,.my_x_i(my_x_i)
    ,.my_y_i(my_y_i)

    ,.all_slave_done_i(all_slave_done_i)
    ,.all_pe_ready_i(all_pe_ready_i)
    ,.pe_data_v_o(pe_data_v_o)
    ,.all_wgt_o(all_wgt_o)
    ,.all_bias_o(all_bias_o)
    ,.sld_o(sld_o)
    ,.imem_r_addr_o(imem_r_addr_o)
    ,.dw_wgt_start_o(dw_wgt_start_o)
  );

endmodule
