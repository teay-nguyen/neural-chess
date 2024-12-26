#!/usr/bin/env python3.11
from position import Position
import chess.pgn
import os, sys
import numpy as np

def get_dataset(num_samples=None):
  gn = 0
  X, Y = [], []
  vals = {'1/2-1/2':0.5, '0-1':0, '1-0':1}
  for pgn in os.listdir('data'):
    pgn_fp = os.path.join('data', pgn)
    with open(pgn_fp) as games:
      game = chess.pgn.read_game(games)
      while game:
        game = chess.pgn.read_game(games)
        if game is None: continue
        res = game.headers['Result']
        if res not in vals: continue
        val = vals[res]
        board = game.board()
        for i, move in enumerate(game.mainline_moves()):
          board.push(move)
          serialized = Position(board).serialize()
          X.append(serialized)
          Y.append(val)
        print('parsed game %d, got %d samples' % (gn, len(X)))
        if num_samples is not None and len(X) > num_samples: break
        gn += 1
  X = np.array(X)
  Y = np.array(Y)
  return X, Y

if __name__ == '__main__':
  X, Y = get_dataset(1000000)
  print(f'X {X.shape} Y {Y.shape}')
  np.savez('processed/dataset_1M.npz', X, Y)
