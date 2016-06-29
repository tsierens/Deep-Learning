
import numpy as np
#import pandas as pd
import random
import time
import sys
import connect_four as cccc
import os
#import theano
#import theano.tensor as T
#import lasagne
from flask import Flask, render_template, request, redirect
#import json

# def flask_player_move(board, active_turn):

# class flask_player:
# #    def __init__(self):
        
#     def make_move(self,board,active_turn):
#         #print (board,active_turn,self.depth)
#         return flask_player_move(board,active_turn,self.depth)[1]
    
app = Flask(__name__)
#app.debug = True
app.GAME = [r"Tic_Tac_Toe", r"Connect_Four"]
app.OPTIONS = [r"Human", r"MiniMax",r"Neural_network"]
app.board = [0]*42
app.player = 1



def net_value(board):
    (m1,m2,m3,m4) = np.load('TD_cccc_100_1_6million(1).npz')['arr_0']
    board = np.copy(board).reshape((1,42))
    m5 = np.dot(board, m1) + m2
    return np.tanh(np.dot(np.vectorize(np.tanh)(m5), m3) + m4) + random.random()*0.2 - 0.1


def game_over(board):
    board = np.array(board).reshape((6,7))
    return cccc.winner(board) or cccc.is_full(board)

def alpha_beta_move(board,active_turn,depth,evaluation = lambda x: 0,alpha = 2):
    swap_dict = {1:-1,-1:1}
    dummy_board = np.copy(board)
    dummy_board = dummy_board.reshape((6,7))
    options = cccc.available_moves(dummy_board)
    random.shuffle(options)
    if len(options) == 1:
        dummy_board[np.where(dummy_board[:,options[0]]==0)[0][-1],options[0]] = active_turn
        if cccc.winner(dummy_board):
            return (1,options[0]+1)
        else:
            return (0,options[0]+1)
    if depth ==0:
        best_value = -2
        for x in options:
            height = np.where(dummy_board[:,x]==0)[0][-1]
            dummy_board[height,x] = active_turn
            eval_board = evaluation(dummy_board*active_turn)
            if  eval_board > best_value:
                best_value = eval_board
                candidate_move = x + 1
            dummy_board[height,x] = 0
        return (best_value, candidate_move)

    best_value = -2
    candidate_move = None
    for x in options:
        height = np.where(dummy_board[:,x]==0)[0][-1]
        dummy_board[height,x] = active_turn
        if cccc.winner(dummy_board):
            return (1, x+1)
        (opp_value,opp_move) = alpha_beta_move(dummy_board,swap_dict[active_turn],depth-1,evaluation,-best_value)
        if -opp_value > best_value:
            candidate_move = x+1
            best_value = -opp_value
        if -opp_value >= alpha:
            #print (options, x, best_value, alpha)
            break
        dummy_board[height,x] = 0

    return (best_value, candidate_move)



@app.route('/')
def main():
    return redirect('/index')
    

@app.route('/clear', methods = ['GET'])
def clear():
    return redirect('/index')
    #return render_template('index.html',op = app.OPTIONS, names = app.NAMES)

        
@app.route('/index', methods = ['GET','POST'])

def go():
    if request.method == 'GET':
        board = app.board
        player = app.player
        print "HOWDY"
        return render_template('connect_four.html', board = board, cplayer = player, finished = -2 )
    if request.method == 'POST':
        player = int(request.form.get("player"))
        board = request.form.get("board")
        board = board.split(",")
        #print board
        board = [int(x) for x in board]
        board = np.array(board)
        #print board,player
        if game_over(np.copy(board)):
            if cccc.winner(board.reshape((6,7)))==1:
                print '1'
                return render_template('connect_four.html', board = list(board), cplayer = player, finished = 1)
            if cccc.winner(board.reshape((6,7))) ==0:
                print '0'
                return render_template('connect_four.html', board = list(board), cplayer = player, finished = 0)
            if cccc.winner(board.reshape((6,7))) == -1:
                print '-1'
                return render_template('connect_four.html', board = list(board), cplayer = player, finished = -1)
        while not game_over(board):
            if player == -1:
                _,move = alpha_beta_move(board,player,depth=2, evaluation = net_value)
                #print move
                #print game_over(board)
                board = board.reshape((6,7))
                board[np.where(board[:,move-1]==0)[0][-1],move-1] = active_turn = player
                player = -1*player
                board = board.reshape(42)
                #print board,player

            elif player == 1:
                #print board, player
                return render_template('connect_four.html', board = list(board), cplayer = player,finished=-2)
        if game_over(np.copy(board)):
            if cccc.winner(board.reshape((6,7)))==1:
                print '1'
                return render_template('connect_four.html', board = list(board), cplayer = player, finished = 1)
            if cccc.winner(board.reshape((6,7))) ==0:
                print '0'
                return render_template('connect_four.html', board = list(board), cplayer = player, finished = 0)
            if cccc.winner(board.reshape((6,7))) == -1:
                print '-1'
                return render_template('connect_four.html', board = list(board), cplayer = player, finished = -1)
       
    
  
    
if __name__ == '__main__':
    # Bind to PORT if defined, otherwise default to 5000.
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
