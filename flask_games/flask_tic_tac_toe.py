import numpy as np
#import pandas as pd
import random
import time
import sys
import tic_tac_toe as ttt
from flask import Flask, render_template, request, redirect
import json

def game_over(board):
    return ttt.is_full(board) or ttt.winner(board)
class alpha_beta:
    def __init__(self,depth):
        self.depth = depth
    def make_move(self,board,active_turn):
        #print (board,active_turn,self.depth)
        return alpha_beta_move(board,active_turn,self.depth)[1]
def alpha_beta_move(board,active_turn,depth,alpha = 2):
    swap_dict = {1:-1,-1:1}
    dummy_board = np.copy(np.array(board))
    options = ttt.available_moves(dummy_board)
    random.shuffle(options)
    if len(options) == 1:
        dummy_board[options[0]] = active_turn
        if ttt.winner(dummy_board):
            return (1,options[0]+1)
        else:
            return (0,options[0]+1)
    if depth ==0:
        return (0, options[np.random.randint(len(options))]+1)

    best_value = -2
    candidate_move = None
    for x in options:
        dummy_board[x] = active_turn
        if ttt.winner(dummy_board):
            return (1, x+1)
        (opp_value,opp_move) = alpha_beta_move(dummy_board,swap_dict[active_turn],depth-1,-best_value)
        if -opp_value > best_value:
            candidate_move = x+1
            best_value = -opp_value
        if -opp_value >= alpha:
            #print (options, x, best_value, alpha)
            break
        dummy_board[x] = board[x]

    return (best_value, candidate_move)

def play(p1='human',p2='remote',depth=0,heir=None):
    board = np.zeros(9)
    player = 1
    player_type = {1:'human',-1:'human'}
    print 'playing tic tac toe'
    if p1 == "human":
        player1 = ttt.player()
        player_type[1] = 'human'
    else:
        player1 = alpha_beta(depth)
        player_type[1] = 'remote'
    if p2 == "human":
        player2 = ttt.player()
        player_type[-1] =  'human'
    else:
        player2 = alpha_beta(depth)
        player_type[-1] = 'remote'
    print 'starting a game of tic tac toe'
    print [player_type[1], player_type[-1]]
    return render_template('tic_tac_toe.html',board = list(board),
                                   player =player, types = [player_type[1],player_type[-1]], depth = depth,finished = -2)
    
    
    