#!/usr/bin/env python3.11
import time
import chess
import torch
from position import Position

class Evaluator:
  def __init__(self):
    from train import Net
    vals = torch.load('nets/value.pth', map_location=lambda storage, loc: storage)
    self.model = Net()
    self.model.load_state_dict(vals)
    self.count = 0

  def reset(self):
    self.count = 0

  def __call__(self, s):
    self.count += 1
    board = s.serialize()[None]
    out = self.model(torch.tensor(board).float())
    return float(out.data)

ev = Evaluator()
s = Position()

INF = 10000

def minmax(s, v, depth, a, b, big=False):
  if depth >= 5 or s.board.is_game_over(): return v(s)
  turn = s.board.turn
  if turn == chess.WHITE:
    ret = -INF
  else:
    ret = INF
  if big:
    bret = []

  # prune here with beam search
  isort = []
  for e in s.board.legal_moves:
    s.board.push(e)
    isort.append((v(s), e))
    s.board.pop()
  move = sorted(isort, key=lambda x: x[0], reverse=s.board.turn)

  if depth >= 3:
    move = move[:10]

  # alpha beta search
  for e in [x[1] for x in move]:
    s.board.push(e)
    tval = minmax(s, v, depth+1, a, b)
    s.board.pop()
    if big: bret.append((tval, e))
    if turn == chess.WHITE:
      ret = max(ret, tval)
      a = max(a, ret)
      if a >= b: break
    else:
      ret = min(ret, tval)
      b = min(b, ret)
      if a >= b: break
  if big: return ret, bret
  else: return ret

def explore_leaves(s, v):
  ret = []
  start = time.time()
  v.reset()
  bval = v(s)
  cval, ret = minmax(s, v, 0, a=-INF, b=INF, big=True)
  eta = time.time() - start
  print(f'{bval} -> {cval}: explored {v.count} nodes in {eta} seconds {int(v.count/eta)}/sec')
  return ret

pos = Position()
ev = Evaluator()

def computer_move(s, v):
  move = sorted(explore_leaves(s, v), key=lambda x: x[0], reverse=s.board.turn)
  if len(move) == 0: return 
  print('top 3: ')
  for i,m in enumerate(move[0:3]):
    print('  ', m)
  print(s.board.turn, 'moving', move[0][1])
  s.board.push(move[0][1])

if __name__ == '__main__':
  s = Position()
  while not s.board.is_game_over():
    computer_move(s, ev)
    print(s.board)
  print(s.board.result())
