// Copyright 2021 ETH Zurich and University of Bologna.
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51

// Description: Scrambles the address in such a way, that part of the memory is accessed
// sequentially and part is interleaved.
// Current constraints:

// Author: Samuel Riedel <sriedel@iis.ee.ethz.ch>
// Author: Marco Bertuletti <mbertuletti@iis.ee.ethz.ch>

module address_scrambler #(
  parameter int unsigned AddrWidth         = 32,
  parameter int unsigned DataWidth         = 32,
  parameter int unsigned ByteOffset        = 2,
  parameter bit          Bypass            = 0,
  parameter int unsigned NumTiles          = 2,
  parameter int unsigned NumBanksPerTile   = 2,
  parameter int unsigned TCDMSizePerBank   = 1024,
  parameter int unsigned SeqMemSizePerTile = 4096,
  parameter int unsigned NumDASPartitions  = 4,
  // Dependant parameters, do not change
  parameter int unsigned MemSizePerTile    = NumBanksPerTile*TCDMSizePerBank,
  parameter int unsigned MemSizePerRow     = (1 << ByteOffset)*NumBanksPerTile*NumTiles
) (
  input  logic [AddrWidth-1:0]                            address_i,
  input  logic [NumDASPartitions-1:0][$clog2(NumTiles):0] group_factor_i,
  input  logic [NumDASPartitions-1:0][$clog2(NumTiles):0] allocated_size_i,
  input  logic [NumDASPartitions-1:0][DataWidth-1:0]      start_addr_scheme_i,
  output logic [AddrWidth-1:0]                            address_o
);
  // Stack Sequential Settings
  localparam int unsigned BankOffsetBits    = $clog2(NumBanksPerTile);
  localparam int unsigned TileIdBits        = $clog2(NumTiles);
  localparam int unsigned SeqPerTileBits    = $clog2(SeqMemSizePerTile);
  localparam int unsigned SeqTotalBits      = SeqPerTileBits+TileIdBits;
  localparam int unsigned ConstantBitsLSB   = ByteOffset + BankOffsetBits;
  localparam int unsigned ScrambleBits      = SeqPerTileBits-ConstantBitsLSB;

  if (Bypass || NumTiles < 2) begin
    assign address_o = address_i;
  end else begin
    // ------ Stack Region Logic ------ //
    logic [ScrambleBits-1:0]    scramble;    // Address bits that have to be shuffled around
    logic [TileIdBits-1:0]      tile_id;     // Which tile does  this address region belong to

    // Scramble the middle part
    // Bits that would have gone to different tiles but now go to increasing lines in the same tile
    assign scramble = address_i[SeqPerTileBits-1:ConstantBitsLSB]; // Bits that would
    // Bits that would have gone to increasing lines in the same tile but now go to different tiles
    assign tile_id  = address_i[SeqTotalBits-1:SeqPerTileBits];

    // ------ Heap Sequential Signals ------ //
    
    // `tile_index` : how many bits to shift for TileID bits in each partition
    // `row_index`: how many bits need to swap within Row Index
    logic [NumDASPartitions-1:0][$clog2($clog2(NumTiles)+1)-1:0] tile_index;
    logic [NumDASPartitions-1:0][$clog2($clog2(NumTiles)+1)-1:0] row_index;

    for (genvar i = 0; i < NumDASPartitions; i++) begin : gen_shift_index
      lzc #(
        .WIDTH ($clog2(NumTiles)+1),
        .MODE  (1'b0              )
      ) i_log_tile_index (
        .in_i    (group_factor_i[i]),
        .cnt_o   (tile_index[i]    ),
        .empty_o (/* Unused */     )
      );
      lzc #(
        .WIDTH ($clog2(NumTiles)+1),
        .MODE  (1'b0            )
      ) i_log_row_index (
        .in_i    (allocated_size_i[i][$clog2(NumTiles):0]),
        .cnt_o   (row_index[i]                           ),
        .empty_o (/* Unused */                           )
      );
    end

    always_comb begin
      // Default: Unscrambled
      address_o[ConstantBitsLSB-1:0] = address_i[ConstantBitsLSB-1:0];
      address_o[SeqTotalBits-1:ConstantBitsLSB] = {tile_id, scramble};
      address_o[AddrWidth-1:SeqTotalBits] = address_i[AddrWidth-1:SeqTotalBits];

      // Stack Region
      if (address_i < (NumTiles * SeqMemSizePerTile)) begin
        address_o[SeqTotalBits-1:ConstantBitsLSB] = {scramble, tile_id};

      // DAS address scrambling
      end else begin

        for (int p = 0; p < NumDASPartitions; p++) begin
          if ( (address_i >= start_addr_scheme_i[0]) && (address_i < start_addr_scheme_i[0]+MemSizePerRow*allocated_size_i[0]) ) begin
            address_o = '0;
            address_o |= address_i & ((1 << (tile_index[0]+ConstantBitsLSB)) - 1);
            address_o |= ((address_i >> (row_index[0]+tile_index[0]+ConstantBitsLSB)) << (tile_index[0]+ConstantBitsLSB)) & ((1 << (TileIdBits+ConstantBitsLSB)) - 1);
            address_o |= ((address_i >> (tile_index[0]+ConstantBitsLSB)) << (TileIdBits + ConstantBitsLSB)) & ((1 << (row_index[0]+TileIdBits+ConstantBitsLSB)) - 1);
            address_o |= address_i & ~((1 << (row_index[0]+TileIdBits+ConstantBitsLSB)) - 1);
          end
        end

      end 
    end
  end

  // Check for unsupported configurations
  if (NumBanksPerTile < 2)
    $fatal(1, "NumBanksPerTile must be greater than 2. The special case '1' is currently not supported!");
  if (SeqMemSizePerTile % (2**ByteOffset*NumBanksPerTile) != 0)
    $fatal(1, "SeqMemSizePerTile must be a multiple of BankWidth*NumBanksPerTile!");
endmodule : address_scrambler
