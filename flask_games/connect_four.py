
# coding: utf-8

# In[1]:

import numpy as np
import sys
import time

def empty_board():
    return np.zeros((6,7))



def get_all_fours(board):
    list_ = []
    for row in xrange(6):
        for column in xrange(4):
            list_.append(board[row,column:column+4])
    for row in xrange(3):
        for column in xrange(7):
            list_.append(board[row:row+4,column])
    for row in xrange(3):
        for column in xrange(4):        
            list_.append(np.array([board[row+i,column+i] for i in xrange(4)]))
            list_.append(np.array([board[5-(row+i),column+i] for i in xrange(4)]))
    return list_

def available_moves(board):
    return [move for move in xrange(7) if not np.product(board[:,move])]


def winner(board):
    rows = board[:,:4]+board[:,1:5]+board[:,2:6]+board[:,3:7]
    if np.any(rows == 4) or np.any(rows == -4):
        return 1 if np.any(rows==4) else -1
    cols = board[:3,:]+board[1:4,:]+board[2:5,:]+board[3:6,:]
    if np.any(cols == 4) or np.any(cols == -4):
        return 1 if np.any(cols==4) else -1
    diag = board[:3,:4]+board[1:4,1:5]+board[2:5,2:6]+board[3:6,3:7]
    if np.any(diag == 4) or np.any(diag == -4):
        return 1 if np.any(diag==4) else -1
    rdiag = board[5:2:-1,:4]+board[4:1:-1,1:5]+board[3:0:-1,2:6]+board[2::-1,3:7]
    if np.any(rdiag == 4) or np.any(rdiag == -4):
        return 1 if np.any(rdiag==4) else -1
    return 0


#def winner(board):
#    possible_wins = get_all_fours(board)
#    result = 0
#    for item in possible_wins:
#        if len(set(item)) == 1 and item[0] != 0:
#            result = item[0]
#            break
#    return result

def is_full(board):
    return not np.any(board==0)

def game_over(board):
    return winner(board) or is_full(board)

def print_board(board):
    active_dict = {-1 : 'O', 0 : ' ', 1 : 'X'}
    dummy = np.copy(board)
    dummy = dummy.reshape(42)
    dummy = [active_dict[int(element)] for element in dummy]
    dummy = tuple(dummy)
    player_dict_r = {1:'X', -1:'O', 0: ' '}    
    print('''
   {} | {} | {} | {} | {} | {} | {}  
  ---------------------------        
   {} | {} | {} | {} | {} | {} | {}  
  ---------------------------       
   {} | {} | {} | {} | {} | {} | {}   
  ---------------------------        
   {} | {} | {} | {} | {} | {} | {}  
  ---------------------------       
   {} | {} | {} | {} | {} | {} | {} 
  ---------------------------       
   {} | {} | {} | {} | {} | {} | {}  
  ---------------------------
   1 | 2 | 3 | 4 | 5 | 6 | 7
    '''.format(*dummy))
    
    
class player:
    def make_move(self, board,active_turn):
        player_dict = {1:'X', -1:'O'}        
        print( 'it is '+player_dict[active_turn]+'\'s turn, please choose a move')

        print_board(board)
        move = 0
        while True:
            move = raw_input()
            if move.isdigit():
                move = int(move)
            else:
                print('Please input an integer')
                continue
            if not (move - 1 in available_moves(board)):
                print('Choice must be an available move')
                continue
            break
        return move
    
class result(object):
    def __init__(self,board,log,t1,t2):
        self.winner = winner(board)
        self.log = log
        self.board = board
        self.p1_time = t1
        self.p2_time = t2

def play(player_1,player_2):
    t1=0
    t2=0
    player_dict = {1:'X', -1:'O', 0: ' '}    
    swap_dict = {'O':'X','X':'O'}
    log=[]
    active_turn = 1
    board = empty_board()
    while not game_over(board):
        times_failed = 0
        while True:
            times_failed +=1
            if times_failed > 100:
                sys.exit('Failed to make a legal move 100 times.')
            if active_turn == 1:
                t0 = time.clock()
                move = player_1.make_move(board,active_turn)
                t1 += time.clock() - t0
            elif active_turn == -1:
                t0 = time.clock()
                move = player_2.make_move(board,active_turn)
                t2 += time.clock() - t0
            if move-1 in available_moves(board):
                board[np.where(board[:,move-1]==0)[0][-1],move-1] = active_turn
                break
        log.append(move)
        active_turn = -1 * active_turn
    '''
    if winner(board):
        print('The winner is ' + player_dict_r[int(winner(board))]+'!')
    else:
        print('Tie game!!!')
    '''
    return result(board,log,t1,t2)

