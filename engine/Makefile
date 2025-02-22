FLAGS = -O2 -Ofast -march=native -flto -mbmi2 -std=c++20

test:
	@clang++ $(FLAGS) main.cc board.cc rng.cc -Wall -g -o bin/main
	@./bin/main