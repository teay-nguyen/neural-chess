#pragma once
#include "const.h"
#include <cassert>
#include <array>

using bitboard = uint64_t;
using key = uint64_t;

template<typename T, std::size_t sz, std::size_t... szs>
struct nd_array_impl { using type = std::array<typename nd_array_impl<T, szs...>::type, sz>; };

template<typename T, std::size_t sz>
struct nd_array_impl<T, sz> { using type = std::array<T, sz>; };

template<typename T, std::size_t... szs>
using nd_array = typename nd_array_impl<T, szs...>::type;

template<typename T, std::size_t sz>
void fill_ndarray(nd_array<T, sz> &arr, T val) { arr.fill(val); }

template<typename T, std::size_t sz, std::size_t... szs>
void fill_ndarray(nd_array<typename nd_array_impl<T, szs...>::type, sz> &arr, T val) {
  for(std::size_t i=0; i<sz; ++i)
    fill_ndarray<T, szs...>(arr[i], val);
}

class move {
public:
  uint32_t data;

  move() = default;
  constexpr explicit move(uint32_t d) : data(d) {}
  constexpr move(int f, int t, int fl = NORMAL_MOVE) : data((fl + (f << 6) + t)) {} // pt is by default KNIGHT

  template<int FL>
  static constexpr move make(int f, int t, int pt = KNIGHT) {
  #ifdef BOB_DEBUG
    assert(pt != PAWN && pt != KING);
  #endif
    return move(FL + ((pt - KNIGHT) << 12) + (f << 6) + t);
  }

  constexpr int where_from() const { return (data >> 6) & 0x3F; }
  constexpr int where_to() const { return data & 0x3F; }
  constexpr int get_move_flag() const { return data & (3 << 16); }
  constexpr int get_promotion_type() const { return ((data >> 12) & 3) + KNIGHT; }
};

class state {
public:
  std::array<bitboard, PIECE_NB> pieces;
  std::array<bitboard, COLOR_NB> colors;
  std::array<int, 64> mailbox;
  std::array<int, 2> king_sqs;

  bool side_to_move;
  int ply;

  int castling_rights;
  int epsq;

  bitboard castling_mask;

  key hash;
  key pawn_hash;

  void reset();
  void validate_state();

  void add_piece(int s, int p);
  void remove_piece(int s);

  void move_piece_noisy(int from, int to);
  void move_piece_quiet(int from, int to);

  void play_move(move m);

  friend std::ostream& operator<<(std::ostream& out, const state& st);

  // templates

  template<bool Do_Undo>
  void do_castling(bool us, int from, int *to, int *rfrom, int *rto);
};


// state helper funcs

template<bool Do_Undo>
void state::do_castling(bool us, int from, int *to, int *rfrom, int *rto) {
  bool king_side = *to > from;
  *rfrom = *to;
  *rto = (king_side ? F1 : D1) ^ (!us * 56);
  *to = (king_side ? G1 : C1) ^ (!us * 56);

  remove_piece(Do_Undo ? from : *to);
  remove_piece(Do_Undo ? *rfrom : *rto);
  add_piece(Do_Undo ? *to : from, (us << 3) | KING);
  add_piece(Do_Undo ? *rto : *rfrom, (us << 3) | ROOK);
}