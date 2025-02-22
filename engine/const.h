#pragma once
#include <string>
#include <unordered_map>

enum : int {
  NO_PIECE_TYPE, PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING,
  PTYPE_NB
};

enum : int {
  BLACK, WHITE, BOTH,
  COLOR_NB
};

enum : int {
  NO_PIECE,
  B_PAWN,                         B_KNIGHT, B_BISHOP, B_ROOK, B_QUEEN, B_KING,
  W_PAWN = (WHITE << 3) + B_PAWN, W_KNIGHT, W_BISHOP, W_ROOK, W_QUEEN, W_KING,
  PIECE_NB
};

enum : int {
  NO_CASTLE = 0,
  WHITE_OO =  0b0001,
  WHITE_OOO = 0b0010,
  BLACK_OO =  0b0100,
  BLACK_OOO = 0b1000,

  KING_SIDE_CASTLE = WHITE_OO | BLACK_OO,
  QUEEN_SIDE_CASTLE = WHITE_OOO | BLACK_OOO,
  WHITE_CASTLE = WHITE_OO | WHITE_OOO,
  BLACK_CASTLE = BLACK_OO | BLACK_OOO,
  ALL_CASTLE = WHITE_CASTLE | BLACK_CASTLE,
  CASTLE_NB = 16
};

enum : int {
  NORMAL_MOVE,
  PROMOTION_MOVE = 1 << 16,
  ENPASSANT_MOVE = 2 << 16,
  CASTLING_MOVE  = 3 << 16
};

enum : int {
    NORTH = 8,
    EAST  = 1,
    SOUTH = -NORTH,
    WEST  = -EAST,

    NORTH_EAST = NORTH + EAST,
    SOUTH_EAST = SOUTH + EAST,
    SOUTH_WEST = SOUTH + WEST,
    NORTH_WEST = NORTH + WEST
};

enum : int {
  A8, B8, C8, D8, E8, F8, G8, H8,
  A7, B7, C7, D7, E7, F7, G7, H7,
  A6, B6, C6, D6, E6, F6, G6, H6,
  A5, B5, C5, D5, E5, F5, G5, H5,
  A4, B4, C4, D4, E4, F4, G4, H4,
  A3, B3, C3, D3, E3, F3, G3, H3,
  A2, B2, C2, D2, E2, F2, G2, H2,
  A1, B1, C1, D1, E1, F1, G1, H1, NO_SQ,
  SQ_NB
};

enum : int {
    RANK_8,
    RANK_7,
    RANK_6,
    RANK_5,
    RANK_4,
    RANK_3,
    RANK_2,
    RANK_1,
    RANK_NB
};


const std::unordered_map<char, int> piece_representation = {
  {'P', W_PAWN}, {'N', W_KNIGHT}, {'B', W_BISHOP}, {'R', W_ROOK}, {'Q', W_QUEEN}, {'K', W_KING},
  {'p', B_PAWN}, {'n', B_KNIGHT}, {'b', B_BISHOP}, {'r', B_ROOK}, {'q', B_QUEEN}, {'k', B_KING}
};

const std::string_view square_representation[64] = {
  "a8", "b8", "c8", "d8", "e8", "f8", "g8", "h8",
  "a7", "b7", "c7", "d7", "e7", "f7", "g7", "h7",
  "a6", "b6", "c6", "d6", "e6", "f6", "g6", "h6",
  "a5", "b5", "c5", "d5", "e5", "f5", "g5", "h5",
  "a4", "b4", "c4", "d4", "e4", "f4", "g4", "h4",
  "a3", "b3", "c3", "d3", "e3", "f3", "g3", "h3",
  "a2", "b2", "c2", "d2", "e2", "f2", "g2", "h2",
  "a1", "b1", "c1", "d1", "e1", "f1", "g1", "h1",
};

const std::string_view start_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";