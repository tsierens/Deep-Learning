

import numpy as np
#import pandas as pd
import random
import time
import sys
import tic_tac_toe as ttt
import connect_four as cccc
import os
from flask import Flask, render_template, request, redirect,jsonify
import flask_connect_four as fc4
import flask_tic_tac_toe as ft3
import json

app = Flask(__name__,)
#app.debug = True
#app.GAMES = [r"Tic_Tac_Toe", r"Connect_Four"]
#app.OPTIONS = [r"Human", r"Computer"]
#app.EVALUATION = ["Random", "Neural Network"]
#app.DEPTH = 0
#app.PLAYERS = ['1','2']
               
@app.route('/')
def main():
    return redirect('/index')
    

@app.route('/clear', methods = ['GET'])
def clear():
    return redirect('/index')
    #return render_template('index.html',op = app.OPTIONS, names = app.NAMES)

        
@app.route('/index', methods = ['GET','POST'])
def go():
    print 'HELLO!'
    if request.method == 'GET':
        print 'rendering menu'
        return render_template('menu.html')
    if request.method == 'POST':
        print 'receieved submission'
        game = request.form.get("game")
        p1 = request.form.get("p1type")
        p2 = request.form.get("p2type")
        try:
            p1depth = int(request.form.get("p1depth"))
        except:
            p1depth = 0
        try:
            p2depth = int(request.form.get("p2depth"))
        except:
            p2depth = 0
        try:
            p1eval = request.form.get("p1eval") 
        except:
            p1eval = "random"
        try:
            p2eval = request.form.get("p2eval") 
        except:
            p2eval = "random"
        print 'players', p1,p2
        print 'depths', p1depth,p2depth
        print 'evals', p1eval,p2eval
        print game
        if game == 'ttt':
            print 'calling tic tac toe initialization'
            return ft3.play(p1=p1,p2=p2,depths = (p1depth,p2depth),evals = (p1eval,p2eval))
        if game == 'c4':
            print 'calling connect four initialization'
            return fc4.play(types =(p1,p2),depths = (p1depth,p2depth),evals = (p1eval,p2eval))
    return None
            
@app.route('/flow', methods = ['get'])
def flow():
    return render_template("flow.html")
                           
@app.route('/minimax', methods = ['get'])
def minimax():
    return render_template("minimax.html")

@app.route('/bio', methods = ['get'])
def bio():
    return render_template("bio.html")
    
    
    
@app.route('/ttt',methods = ['POST'])
def play_ttt():
    player_index_dict = {-1:1,1:0}
    player = int(request.form.get("player"))
    depth = int(request.form.get("depth"))
    board = request.form.get("board")
    board = board.split(",")
#    print board
    board = [int(x) for x in board]
    board = np.array(board)
    types = str(request.form.get("types"))
    types = types.split(",")

    print "The contorl methods are" , types
#    print types, board, player, ttt.game_over(np.copy(board))
    print "the board is ", board
    print "the current player is ", player
    if  types[player_index_dict[player]] == 'remote' and not ttt.game_over(np.copy(board)):
        move = ft3.alpha_beta_move(board, -1, depth = depth)[1]
        print "the move is ",move
        board[move-1] = player
        print "the new board is ", board
        player *= -1
    if ttt.game_over(np.copy(board)):
        if ttt.winner(board)==1:
            return render_template('tic_tac_toe.html', board = list(board), player = player, types = types,depth = depth, finished = 1)
        if ttt.winner(board) ==0:
            return render_template('tic_tac_toe.html', board = list(board), player = player, types = types,depth = depth, finished = 0)
        if ttt.winner(board) == -1:
            return render_template('tic_tac_toe.html', board = list(board), player = player, types = types,depth = depth, finished = -1)
    else:
        print 'render next move'
        return render_template('tic_tac_toe.html',board = list(board), player = player, types = types,depth = depth, finished = -2)
    
@app.route('/cccc',methods = ['POST'])
def play_cccc():

    print "requesting move"
    player_index_dict = {-1:1,1:0}
    player = int(request.form.get("player"))
    depths = map(int,request.form.get("depths").split(','))
    print "the depths are ", depths
    board = request.form.get("board")
    board = board.split(",")
    board = [int(x) for x in board]
    board = np.array(board).reshape((6,7))
    print "the board is "
    print board
    types = request.form.get("types")
    types = map(lambda x: x.replace("\"",""),types.split(","))
    evals = request.form.get("evals").split(",")
    print "the eval method is ",evals[player_index_dict[player]]
    if fc4.game_over(np.copy(board).reshape((6,7))):
        finished = cccc.winner(board.reshape((6,7)))
        return jsonify(finished = finished)
    else:
        finished = -2
    if evals[player_index_dict[player]] == 'nn':
        evaluation = fc4.net_value
    else:
        evaluation = lambda x: 0
    #evaluation
    print "the control methods are ", types
#    print types, player, fc4.game_over(np.copy(board)), evals, depths[player_index_dict[player]]
    if  types[player_index_dict[player]] == 'remote' and not fc4.game_over(np.copy(board)):
        move = fc4.alpha_beta_move(board.reshape((6,7)),
                                   player,
                                   depth = depths[player_index_dict[player]],
                                   evaluation = evaluation)[1]
        print "the next move is ",move

        fc4.update_move(board,move,player)
        print "the board is "
        print board
        player *= -1
    board = board.reshape(42)
    print 'next move is', move
    if fc4.game_over(np.copy(board).reshape((6,7))):
        finished = cccc.winner(board.reshape((6,7)))
    else:
        finished = -2
    return jsonify(move=move, player = -1*player, finished = finished)

       
    
#render_template('connect_four.html',
#                                   board = list(board),
#                                   player = player,
#                                   types = map(str,types),
#                                   evals = map(str,evals),
#                                   depths = depths,
#                                   finished = finished)
               
               
               
               
if __name__ == '__main__':
    # Bind to PORT if defined, otherwise default to 5000.
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host='0.0.0.0', port=port)