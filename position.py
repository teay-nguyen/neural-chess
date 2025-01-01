#!/usr/bin/env python
import chess
import numpy as np

class Position:
  def __init__(self, board=None):
    if board is None:
      self.board = chess.Board()
    else:
      self.board = board 

  def serialize(self):
    #pboard = np.zeros((14, 64), np.uint8)
    pboard = np.zeros((15, 64), np.uint8)
    pset = {"P": 0, "N": 1, "B": 2, "R": 3, "Q": 4, "K": 5, \
            "p": 6, "n": 7, "b": 8, "r": 9, "q":10, "k": 11}

    #   piece-wise encoding
    for i in range(64):
      pchar = self.board.piece_at(i)
      if pchar is None: continue
      pboard[pset[pchar.symbol()],i] = 1

    #   encode castling
    if self.board.has_queenside_castling_rights(chess.WHITE):
      assert pboard[pset['R'],0]
      pboard[12,0] = 1
    if self.board.has_kingside_castling_rights(chess.WHITE):
      assert pboard[pset['R'],7]
      pboard[12,7] = 1
    if self.board.has_queenside_castling_rights(chess.BLACK):
      assert pboard[pset['r'],56]
      pboard[12,56] = 1
    if self.board.has_kingside_castling_rights(chess.BLACK):
      assert pboard[pset['r'],63]
      pboard[12,63] = 1

    #   encode enpassant
    if self.board.ep_square is not None:
      assert (pboard[pset['P']] | pboard[pset['p']])[self.board.ep_square] == 0
      pboard[13,self.board.ep_square] = 1

    #    does encoding turns make it worse?
    pboard[14] = (self.board.turn*1.0)

    pboard = pboard.reshape((-1,8,8))
    return pboard 
