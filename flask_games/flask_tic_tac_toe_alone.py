import numpy as np
#import pandas as pd
import random
import time
import sys
import tic_tac_toe as ttt
from flask import Flask, render_template, request, redirect
import json

# def flask_player_move(board, active_turn):

# class flask_player:
# #    def __init__(self):
        
#     def make_move(self,board,active_turn):
#         #print (board,active_turn,self.depth)
#         return flask_player_move(board,active_turn,self.depth)[1]
    
app = Flask(__name__)
app.debug = True
app.GAME = [r"Tic_Tac_Toe", r"Connect_Four"]
app.OPTIONS = [r"Human", r"MiniMax",r"Neural_network"]
app.board = [0,0,0,0,0,0,0,0,0]
app.player = 1


def game_over(board):
    return ttt.is_full(board) or ttt.winner(board)

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

@app.route('/')
def main():
    app.board = [0,0,0,0,0,0,0,0,0]
    app.player = 1
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
        return render_template('tic_tac_toe.html', board = board, cplayer = player, finished = -2 )
    if request.method == 'POST':
        player = int(request.form.get("player"))
        board = request.form.get("board")
        board = board.split(",")
        board = [int(x) for x in board]
        board = np.array(board)
        print board,player
        if game_over(np.copy(board)):
            if ttt.winner(board)==1:
                return render_template('tic_tac_toe.html', board = list(board), cplayer = player, finished = 1)
            if ttt.winner(board) ==0:
                return render_template('tic_tac_toe.html', board = list(board), cplayer = player, finished = 0)
            if ttt.winner(board) == -1:
                return render_template('tic_tac_toe.html', board = list(board), cplayer = player, finished = -1)
        while not game_over(board):
            print "HELLO"
            if player == -1:
                _,move = alpha_beta_move(board,player,depth=6)
                print move
                board[move-1] = player
                player = -1*player
                print board,player

            elif player == 1:
                print board, player
                print 'pleaaaaaaaaase render!!!'
                return render_template('tic_tac_toe.html', board = list(board), cplayer = player,finished=-2)
                print 'you shouldnt see this'
        if game_over(np.copy(board)):
            if ttt.winner(board)==1:
                return render_template('tic_tac_toe.html', board = list(board), cplayer = player, finished = 1)
            if ttt.winner(board) ==0:
                return render_template('tic_tac_toe.html', board = list(board), cplayer = player, finished = 0)
            if ttt.winner(board) == -1:
                return render_template('tic_tac_toe.html', board = list(board), cplayer = player, finished = -1)
       
    
  
    
if __name__ == '__main__':
    app.run(port=5000)
