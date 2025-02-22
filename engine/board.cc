#include "types.h"
#include "bit.h"
#include "rng.h"
#include <cassert>

// state main funcs

std::ostream& operator<<(std::ostream& out, const state& st) {
  std::unordered_map<int,char> flipped;
  for(const auto& pair : piece_representation) flipped[pair.second] = pair.first;
  flipped[NO_PIECE] = ' ';
  out << "\n +---+---+---+---+---+---+---+---+\n";
  for(int r=0; r<8; ++r) {
    for(int f=0; f<8; ++f)
      out << " | " << flipped[st.mailbox[(r << 3) + f]];
    out << " | " << (8 - r) << "\n +---+---+---+---+---+---+---+---+\n";
  }
  out << "   a   b   c   d   e   f   g   h\n\n";
  out << "hash: " << std::hex << st.hash << std::dec << std::endl;
  out << "pawn hash: " << std::hex << st.pawn_hash << std::dec << std::endl;

  return out;
}

void state::validate_state() {
  // check mailbox and bitboards
  for(int p=B_PAWN; p<=B_KING; ++p) {
    bitboard pbb = pieces[p];
    while (pbb) assert(mailbox[Bits::pop_lsb(&pbb)] == p && "piece bitboard not matching");
  }

  for(int p=W_PAWN; p<=W_KING; ++p) {
    bitboard pbb = pieces[p];
    while (pbb) assert(mailbox[Bits::pop_lsb(&pbb)] == p && "piece bitboard not matching");
  }

  bitboard wbb = 0ull, bbb = 0ull;
  for(int p=B_PAWN; p<=B_KING; ++p) bbb |= pieces[p];
  for(int p=W_PAWN; p<=W_KING; ++p) wbb |= pieces[p];

  assert(wbb == colors[WHITE] && "white bitboard not matching");
  assert(bbb == colors[BLACK] && "black bitboard not matching");
  assert((wbb & bbb) == 0ull);
  assert((wbb | bbb) == colors[BOTH] && "global bitboard not matching");

  // check king square
  assert(Bits::get_lsb(pieces[W_KING]) == king_sqs[WHITE]);
  assert(Bits::get_lsb(pieces[B_KING]) == king_sqs[BLACK]);
}

void state::reset() {
  // TODO: reset new properties
  std::fill(pieces.begin(), pieces.end(), 0);
  std::fill(colors.begin(), colors.end(), 0);
  std::fill(mailbox.begin(), mailbox.end(), NO_PIECE);
  side_to_move = WHITE, ply = 1, castling_rights = 0, epsq = NO_SQ;
  hash = pawn_hash = 0ull;

  for(int i=A2; i<=H2; ++i) add_piece(i, W_PAWN);
  for(int i=A7; i<=H7; ++i) add_piece(i, B_PAWN);

  for(int i : std::initializer_list<int>{B1,G1}) {
    add_piece(i, W_KNIGHT);
    add_piece(i^56, B_KNIGHT);
  }

  for(int i : std::initializer_list<int>{C1,F1}) {
    add_piece(i, W_BISHOP);
    add_piece(i^56, B_BISHOP);
  }

  for(int i : std::initializer_list<int>{A1,H1}) {
    add_piece(i, W_ROOK);
    add_piece(i^56, B_ROOK);
  }

  add_piece(D1, W_QUEEN); add_piece(D8, B_QUEEN);
  add_piece(E1, W_KING); add_piece(E8, B_KING);
  
  castling_mask = Bits::KINGSIDE_CASTLING | Bits::QUEENSIDE_CASTLING;
}

void state::add_piece(int s, int p) {
#ifdef BOB_DEBUG
  assert(p != NO_PIECE);
  assert(mailbox[s] == NO_PIECE);
#endif
  const bitboard mask = 1ull << s;
  colors[BOTH] |= mask;
  pieces[p] |= mask;
  colors[p >> 3] |= mask;
  mailbox[s] = p;
  hash ^= RNG::piece_tbl[s][p];
  if((p & 0b111) == PAWN) pawn_hash ^= RNG::piece_tbl[s][p];
  else if((p & 0b111) == KING) king_sqs[p >> 3] = s;
}

void state::remove_piece(int s) {
#ifdef BOB_DEBUG
  assert(mailbox[s] != NO_PIECE);
  assert((mailbox[s] & 0b111) != KING);
#endif
  const bitboard mask = 1ull << s;
  colors[BOTH] &= ~mask;
  pieces[mailbox[s]] &= ~mask;
  colors[mailbox[s] >> 3] &= ~mask;
  hash ^= RNG::piece_tbl[s][mailbox[s]];
  if((mailbox[s] & 0b111) == PAWN) pawn_hash ^= RNG::piece_tbl[s][mailbox[s]];
  mailbox[s] = NO_PIECE;
}

void state::move_piece_noisy(int from, int to) {
#ifdef BOB_DEBUG
  assert(mailbox[from] != NO_PIECE);
  assert(mailbox[to] != NO_PIECE);
  assert((mailbox[to] >> 3) != (mailbox[from] >> 3));
#endif
  hash ^= RNG::piece_tbl[from][mailbox[from]] ^ RNG::piece_tbl[to][mailbox[from]] ^ RNG::piece_tbl[to][mailbox[to]];
  const bitboard mask = (1ull << from) | (1ull << to);

  pieces[mailbox[from]] ^= mask; 
  pieces[mailbox[to]] &= ~(1ull << to); 

  colors[BOTH] &= ~(1ull << to);
  colors[BOTH] ^= mask;

  colors[mailbox[to] >> 3] &= ~(1ull << to);
  colors[mailbox[from] >> 3] ^= mask;

  if((mailbox[from] & 0b111) == KING)
    king_sqs[mailbox[from] >> 3] = to;

  mailbox[to] = mailbox[from];
  mailbox[from] = NO_PIECE;
}

void state::move_piece_quiet(int from, int to) {
#ifdef BOB_DEBUG
  assert(mailbox[from] != NO_PIECE);
  assert(mailbox[to] == NO_PIECE);
#endif
  hash ^= RNG::piece_tbl[from][mailbox[from]] ^ RNG::piece_tbl[to][mailbox[from]];
  const bitboard mask = (1ull << from) | (1ull << to);
  pieces[mailbox[from]] ^= mask;
  colors[BOTH] ^= mask;
  colors[mailbox[from] >> 3] ^= mask;

  if((mailbox[from] & 0b111) == KING)
    king_sqs[mailbox[from] >> 3] = to;

  mailbox[to] = mailbox[from];
  mailbox[from] = NO_PIECE;

}

void state::play_move(move m) {
  bool us = side_to_move;
  bool them = !us;
  int from = m.where_from();
  int to = m.where_to();
  int piece = mailbox[from];
  int captured = m.get_move_flag() == ENPASSANT_MOVE ? ((them << 3) + PAWN) : mailbox[to];

#ifdef BOB_DEBUG
  assert((piece >> 3) == us);
  assert(captured == NO_PIECE || (piece >> 3) == (m.get_move_flag() != CASTLING_MOVE ? them : us));
  assert((piece & 0b111) != KING);
#endif

  if(m.get_move_flag() == CASTLING_MOVE) {
#ifdef BOB_DEBUG
    assert(piece == ((us << 3) + KING));
    assert(captured == ((us << 3) + ROOK));
#endif
    int rfrom, rto;
    do_castling<true>(us, from, &to, &rfrom, &rto);
    // TODO: update hash
  }

  if (captured) {
    int capsq = to;
    if((captured & 0b111) == PAWN) {
      if(m.get_move_flag() == ENPASSANT_MOVE) {
        capsq -= (us == WHITE ? NORTH : SOUTH);
#ifdef BOB_DEBUG
        assert(piece == ((us << 3) | PAWN));
        assert(to == epsq);
        assert(((to >> 3) ^ (!c * 7)) == RANK_3);
        assert(mailbox[to] == NO_PIECE);
        assert(mailbox[capsq] == ((them << 3) | PAWN));
#endif
      }
      // TODO: update pawn hash
    } else {
      // TODO: update hash
    }

    remove_piece(capsq);
    // TODO: (maybe) update hash
    // TODO: update rule 50
  }

  // TODO: holy shit update hash again
}