#include "types.h"
#include "bit.h"
#include "rng.h"
#include <chrono>

int main() {
  auto st = std::chrono::high_resolution_clock::now();

  RNG::init();

  state p;
  p.reset();
  p.validate_state();

  auto et = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double> duration = et - st;

  std::cout << "program finished in " << duration.count() << "s" << std::endl;

}