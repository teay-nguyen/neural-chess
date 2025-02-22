#include <random>
#include "rng.h"

namespace RNG {

nd_array<uint64_t, 64, PIECE_NB> piece_tbl;
nd_array<key, 64> epsq_tbl;
key side_key;

std::mt19937_64 gen(std::random_device{}());

void init() {
  std::uniform_int_distribution<uint64_t> dist;
  for(int i=0; i<64; ++i)
    for(int j=0; j<PIECE_NB; ++j)
      piece_tbl[i][j] = dist(gen);

  for(int i=0; i<64; ++i) epsq_tbl[i] = dist(gen);
  side_key = dist(gen);
}

}