#pragma once
#include <iostream>

namespace Bits {

inline constexpr uint64_t KINGSIDE_CASTLING = 0x9000000000000090ull;
inline constexpr uint64_t QUEENSIDE_CASTLING = 0x1100000000000011ull;

inline constexpr uint64_t NORTH_EDGE_MASK = 0x00000000000000FF;
inline constexpr uint64_t EAST_EDGE_MASK = 0x8080808080808080;
inline constexpr uint64_t SOUTH_EDGE_MASK = 0xFF00000000000000;
inline constexpr uint64_t WEST_EDGE_MASK = 0x0101010101010101;

static constexpr uint64_t RANK_MASKS[9] = {
    0ull,
    0xFF00000000000000ULL,
    0x00FF000000000000ULL,
    0x0000FF0000000000ULL,
    0x000000FF00000000ULL,
    0x00000000FF000000ULL,
    0x0000000000FF0000ULL,
    0x000000000000FF00ULL,
    0x00000000000000FFULL,
};

static constexpr uint64_t FILE_MASKS[9] = {
    0ull,
    0x0101010101010101ULL,
    0x0202020202020202ULL,
    0x0404040404040404ULL,
    0x0808080808080808ULL,
    0x1010101010101010ULL,
    0x2020202020202020ULL,
    0x4040404040404040ULL,
    0x8080808080808080ULL,
};

static void print_bb(uint64_t bb) {
  for(int r=0; r<8; ++r) {
    for(int f=0; f<8; ++f) {
      int s = (r << 3) + f;
      std::cout << ((bb >> s) & 1) << " ";
    }
    std::cout << (8-r) << " \n";
  }
  std::cout << "a b c d e f g h\n\n";
}

inline void set_bit(uint64_t *bb, int s) {
  *bb |= (1ull << s);
}

inline int cnt_bits(uint64_t bb) {
  return __builtin_popcountll(bb);
}

inline int get_lsb(uint64_t bb) {
  return __builtin_ctzll(bb);
}

constexpr int debruijn[64] = {
    0, 47,  1, 56, 48, 27,  2, 60,
   57, 49, 41, 37, 28, 16,  3, 61,
   54, 58, 35, 52, 50, 42, 21, 44,
   38, 32, 29, 23, 17, 11,  4, 62,
   46, 55, 26, 59, 40, 36, 15, 53,
   34, 51, 20, 43, 31, 22, 10, 45,
   25, 39, 14, 33, 19, 30,  9, 24,
   13, 18,  8, 12,  7,  6,  5, 63
};

inline int get_msb(uint64_t bb) {
  const uint64_t debruijn64 = 0x03f79d71b4cb0a89ull;
  bb |= bb >> 1; 
  bb |= bb >> 2;
  bb |= bb >> 4;
  bb |= bb >> 8;
  bb |= bb >> 16;
  bb |= bb >> 32;
  return debruijn[(bb * debruijn64) >> 58];
}

inline int pop_lsb(uint64_t *bb) {
  int l = get_lsb(*bb);
  *bb &= *bb - 1;
  return l;
}

inline uint64_t flip(uint64_t x) {
  return  ( (x << 56)                           ) |
          ( (x << 40) & 0x00ff000000000000ull ) |
          ( (x << 24) & 0x0000ff0000000000ull ) |
          ( (x <<  8) & 0x000000ff00000000ull ) |
          ( (x >>  8) & 0x00000000ff000000ull ) |
          ( (x >> 24) & 0x0000000000ff0000ull ) |
          ( (x >> 40) & 0x000000000000ff00ull ) |
          ( (x >> 56) );
}

inline uint64_t shift_north(uint64_t bb) {
  return bb >> 8;
}

inline uint64_t shift_south(uint64_t bb) {
  return bb << 8;
}

inline uint64_t shift_north_east(uint64_t b) {
  return (b & ~EAST_EDGE_MASK) >> 7;
}

inline uint64_t shift_north_west(uint64_t b) {
  return (b & ~WEST_EDGE_MASK) >> 9;
}

inline uint64_t shift_south_east(uint64_t b) {
  return (b & ~EAST_EDGE_MASK) << 9;
}

inline uint64_t shift_south_west(uint64_t b) {
  return (b & ~WEST_EDGE_MASK) << 7;
}

inline int get_rank(int s) {
  return s >> 3;
}

inline int get_file(int s) {
  return s & 7;
}

inline int get_diagonal(int s) {
  return get_rank(s) + get_file(s);
}

inline int get_antidiagonal(int s) {
  return 7 + get_rank(s) - get_file(s);
}

};