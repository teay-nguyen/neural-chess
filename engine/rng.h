#pragma once
#include "const.h"
#include "types.h"

namespace RNG {

extern nd_array<key, 64, PIECE_NB> piece_tbl;
extern nd_array<key, 64> epsq_tbl; 
extern key side_key;
void init();

}