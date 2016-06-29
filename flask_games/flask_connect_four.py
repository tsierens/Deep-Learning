
import numpy as np
#import pandas as pd
from numpy import random
import time
import sys
import connect_four as cccc
import os
import json
from flask import Flask, render_template, request, redirect, jsonify

app = Flask(__name__)
#app.debug = True
app.board = [0]*42
app.player = 1

def net_value(board):
    t0=time.clock()
    (m1,m2,m3,m4) = np.load('TD_cccc_100_1_6million(1).npz')['arr_0']
    board = np.copy(board).reshape((1,42))
    m5 = np.dot(board, m1) + m2
    answer = np.tanh(np.dot(np.tanh(m5), m3) + m4) + random.random()*0.05 - 0.05
#    print "node evaluation took {:.5}s".format(time.clock()-t0)
    return answer
#    return np.tanh(np.dot(np.vectorize(np.tanh)(m5), m3) + m4) + random.random()*0.1 - 0.05

def game_over(board):
    board = np.array(board).reshape((6,7))
    return cccc.winner(board) or cccc.is_full(board)

inf = float("inf")

def update_move(board, move, turn):
    board[np.where(board[:,move]==0)[0][-1], move] = turn
    return None
    
def unupdate_move(board, move):
    if 0 in board[:,move]:
        board[np.where(board[:,move]==0)[0][-1]+1, move] = 0
    else:
        board[0,move]=0
    return None

def alpha_beta_move(board, turn, depth = 0, alpha = (-inf,-inf), beta = (inf,inf), evaluation = lambda x: 0):
    dummy_board = np.copy(board) # we don't want to change the board state

    swap_player = {1:-1,-1:1} # So we can change whose turn
    options = cccc.available_moves(board) # get legal moves
    random.shuffle(options) # should inherit move order instead of randomizing


#     if len(options) == 1:
#         update_move(board,options[0])
#         if cccc.winner(dummy_board):
#             return (inf,options[0])
#         else:
#             return (0,options[0])   
    
    best_value = (-inf,-inf)
    
    if not options:
        print board, cccc.game_over(board)
        print 'oops, no available moves'
    cand_move = options[0]
    if depth == 0: 
        for x in options:
            update_move(dummy_board,x,turn)
            op_value = (evaluation(dummy_board*swap_player[turn]) , depth)

            if tuple(-1 * el for el in op_value) > best_value:
                cand_move = x
                best_value = tuple(-1 * el for el in op_value)
                alpha = max(alpha, best_value)
    #        print depth,-op_value, best_value, cand_move,alpha,beta
            if alpha >= beta:
    #                print 'pruned'
                break   #alpha-beta cutoff
            unupdate_move(dummy_board,x)
    else:
    
    
    
        for x in options:

    #        dummy_board = np.copy(board)
    #        height= np.where(board[:,x]==0)[0][-1] #connect four only
    #        dummy_board[height, x] = turn
            update_move(dummy_board,x,turn)
        
            if cccc.winner(dummy_board): #should check over and tied too
                return((inf,depth), x)
            
            if cccc.is_full(dummy_board): #This assumes you can't lose on your turn
                return((0,depth) , x)
            
            op_value,_ = alpha_beta_move( dummy_board,
                                            swap_player[turn],
                                            depth-1,
                                            alpha = tuple(-1 * el for el in beta),
                                            beta = tuple(-1 * el for el in alpha),
                                            evaluation = evaluation)

            if tuple(-1 * el for el in op_value) > best_value:
                cand_move = x
                best_value = tuple(-1 * el for el in op_value)
                alpha = max(alpha, best_value)
    #        print depth,-op_value, best_value, cand_move,alpha,beta
            if alpha >= beta:
    #                print 'pruned'
                break   #alpha-beta cutoff
            unupdate_move(dummy_board,x)
    #        dummy_board[height, x] = 0
    return (best_value, cand_move)

def play(types=['human','human'],depths=(0,0),evals = ("random","random")):
    print 'starting a game of connect Four'
    return render_template('connect_four.html',
                           board = list(np.zeros(42)),
                           player = 1, 
                           finished = -2,
                           types = map(str,types),
                           depths = list(depths),
                           evals = map(str,evals))
                           