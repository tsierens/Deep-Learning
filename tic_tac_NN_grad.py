
# coding: utf-8

# In[ ]:

import numpy as np
import pandas as pd
import lasagne
import theano
import theano.tensor as T
import tic_tac_toe as ttt
import random
import time
import gc
import matplotlib.pyplot as plt
import seaborn
get_ipython().magic(u'matplotlib inline')


# In[ ]:

#initialization

generations = []

hidden_units = 36



value_in = lasagne.layers.InputLayer(shape=(None,9))

#l_drop1 = lasagne.layers.DropoutLayer(l_shape,p=0.2)

value_hid1 = lasagne.layers.DenseLayer(value_in, num_units=hidden_units,W=lasagne.init.GlorotUniform(),
                                          nonlinearity = lasagne.nonlinearities.tanh)
value_hid2 = lasagne.layers.DenseLayer(value_hid1, num_units=hidden_units,W=lasagne.init.GlorotUniform(),
                                          nonlinearity = lasagne.nonlinearities.tanh)

value_hid3 = lasagne.layers.DenseLayer(value_hid2, num_units=hidden_units,W=lasagne.init.GlorotUniform(),
                                          nonlinearity = lasagne.nonlinearities.tanh)

value_drop1 = lasagne.layers.DropoutLayer(value_hid3,p=0.5)
#l_drop2 = lasagne.layers.DropoutLayer(l_hid1,p=0.5)

value_out = lasagne.layers.DenseLayer(value_drop1,
                                  num_units=1, nonlinearity = lasagne.nonlinearities.tanh)



policy_in = lasagne.layers.InputLayer(shape=(1,9))

#l_drop1 = lasagne.layers.DropoutLayer(l_shape,p=0.2)

policy_hid1 = lasagne.layers.DenseLayer(policy_in, num_units=hidden_units,W=lasagne.init.GlorotUniform(),
                                          nonlinearity = lasagne.nonlinearities.rectify)

#l_drop2 = lasagne.layers.DropoutLayer(l_hid1,p=0.5)

policy_out = lasagne.layers.DenseLayer(policy_hid1,
                                  num_units=9,
                                  nonlinearity=lasagne.nonlinearities.softmax)


# In[ ]:

def policy_move(board,active_turn,output_fun,exploration):
    board = board.reshape((1,9))
    X_sym = theano.tensor.matrix()
    y_sym = theano.tensor.ivector()

    player_dict = {'X':1, 'O':-1}
    dummy_board = player_dict[active_turn] * board[:] #make 1s good and -1s bad
    

    move_weights = output_fun(dummy_board)
    move_weights = move_weights.reshape(9)

    options = ttt.available_moves(dummy_board)
    
    if exploration > random.random():
        move = random.choice(options) 
    else:
        available_move_weights = np.array([move_weights[i] for i in options])

        move = options[available_move_weights.argmax(-1)]
    
    return move+1

def value_move(board,active_turn,output_fun,exploration):
    board = board.reshape((1,9))
    X_sym = theano.tensor.matrix()
    y_sym = theano.tensor.ivector()

    player_dict = {'X':1, 'O':-1}

    dummy_board = player_dict[active_turn] * board[:]
    options = ttt.available_moves(dummy_board)
    
    
    if exploration > random.random():
        move = random.choice(options) 
    else:
        move_values = np.zeros(9)
        for move in options:
            dummy_board = player_dict[active_turn] * board[:]
            dummy_board[0][move] = 1
            move_values[move] = -1 * output_fun(-1* dummy_board)
        

        available_move_values = np.array([move_values[move] for move in options])
        
        move = options[available_move_values.argmax(-1)]
    return move + 1
    

    
class nn_ai:
    
    def __init__(self,output_fun, net = 'policy',exploration = 0):
        self.output_fun = output_fun
        self.exploration = exploration
        self.net = net
    
    def make_move(self,board,active_turn):
        if self.net == 'policy':
            move = policy_move(board,active_turn,self.output_fun,self.exploration)
        if self.net == 'value':
            move = value_move(board,active_turn,self.output_fun,self.exploration)
        return move

def alpha_beta_move(board,active_turn,depth,alpha = 2):
    swap_dict = {'X':'O','O':'X'}
    dummy_board = np.arange(9)
    dummy_board[:] = board[:]
    options = ttt.available_moves(board)
    random.shuffle(options)
    player_dict = {'X':1, 'O':-1}
    if len(options) == 1:
        dummy_board[options[0]] = player_dict[active_turn]
        if ttt.winner(dummy_board):
            return (1,options[0]+1)
        else:
            return (0,options[0]+1)
    if depth ==0:
        return (0, options[np.random.randint(len(options))]+1)

    best_value = -2
    candidate_move = None
    for x in options:
        dummy_board[x] = player_dict[active_turn]
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

class alpha_beta:
    def __init__(self,depth):
        self.depth = depth
    def make_move(self,board,active_turn):
        #print (board,active_turn,self.depth)
        return alpha_beta_move(board,active_turn,self.depth)[1]


def tourney(output,games = 50,depth = 0):
    tourney_results = {'wins' : 0, 'ties' : 0, 'losses' : 0}
    for _ in range(games):
        results = ttt.play(nn_ai(output),alpha_beta(depth))
        if results.winner ==  1:
            tourney_results['wins'] +=1
        if results.winner ==  0:
            tourney_results['ties'] +=1
        if results.winner == -1:
            tourney_results['losses'] +=1

        results = ttt.play(alpha_beta(depth),nn_ai(output))
        if results.winner == -1:
            tourney_results['wins'] +=1
        if results.winner ==  0:
            tourney_results['ties'] +=1
        if results.winner ==  1:
            tourney_results['losses'] +=1
    return tourney_results

def fitness(score):
    return (1.1*score['wins'] + score['ties'])

def get_inputs(log):
    boards = []
    piece = 1
    board = np.zeros(9)
    boards.append(np.copy(board))
    for move in log:
        board[move-1] = piece
        piece = -piece
        boards.append(np.copy(board))
    return boards
        
def get_max_future(future_board,value_fun):
    options = ttt.available_moves(future_board)
    dummy_board = np.copy(future_board)
    move_values = np.zeros(9)
    for move in options:
        dummy_board = np.copy(future_board)
        dummy_board[move] = -1
        dummy_board = dummy_board.reshape(1,9)
        if ttt.winner(dummy_board):
            move_values[move] = ttt.winner(dummy_board)
        else:
            move_values[move] = value_fun(dummy_board)
    
    available_move_values = np.array([move_values[move] for move in options])
    dummy_board = np.copy(future_board)
    options_index = np.argmin(available_move_values)
    dummy_board[options[options_index]] = -1
    return np.amin(available_move_values), dummy_board

def random_move(board,turn):
    options = ttt.available_moves(board)
    move = random.choice(options)
    dummy_board = np.copy(board)
    dummy_board[move] = turn
    return dummy_board
        
def random_game(board,turn):
    dummy_board = np.copy(board)
    while not (ttt.is_winner(dummy_board) or ttt.is_full(dummy_board)):
        dummy_board = random_move(dummy_board,turn)
        turn = -1*turn
    return ttt.winner(dummy_board)
    
    
    
def monte_carlo_reward(board,trials = 1000):
    reward = 0
    for _ in range(trials):
        reward += random_game(board,1)
    return float(reward) / float(trials)

def next_board(board,move,player):
    dummy_board = np.copy(board)
    dummy_board[move] = player
    return dummy_board

def game_over(board):
    return ttt.winner(board) or ttt.is_full(board)
    
    
    
def mc_step(branch,results,epsilon, cutoff = 10000):
    dummy_board = np.copy(branch[-1])
    #To help convergence we will randomly drop stored values

    #if random.random() < 1/float(cutoff):
    #    results[tuple(dummy_board)] =  {'result':0,'plays':0}     

    
    if not results.get(tuple(dummy_board)):
        results[tuple(dummy_board)] = {'result':0,'plays':0}
        
    board_plays = results[tuple(dummy_board)]['plays']
    board_result = results[tuple(dummy_board)]['result']
    
    if game_over(dummy_board):
        result = ttt.winner(dummy_board)
        
    elif board_plays> cutoff:
        result = results[tuple(dummy_board)]['result'] / float(results[tuple(dummy_board)]['plays'])
        
    else: 
        options = ttt.available_moves(dummy_board)
        future_boards = [next_board(dummy_board,move,1) for move in options]
        if all(results.get(tuple(-1 * b)) for b in future_boards):
            if epsilon(board_plays) > random.random():
                dummy_board = random.choice(future_boards)
            else:
                dummy_board = min(future_boards,key = lambda x :
                                  results[tuple(-1 * x)]['result'] / float(results[tuple(-1 * x)]['plays'])) 
        
        else: 
            dummy_board = random.choice(future_boards)
            
        branch.append(-1 * np.copy(dummy_board))
        result , _ = mc_step(branch,results,epsilon,cutoff)
        result = -1 * result
    
    return result , branch

def monte_carlo_mod(board,results,epsilon,duration = 1, player = 1,cutoff = 10000):
    #To help convergence we will randomly drop stored values

    #if random.random() < 1/float(cutoff):
    #    results[tuple(board)] =  {'result':0,'plays':0}     
        
    t0 = time.clock()
    if not results.get(tuple(board)):
        results[tuple(board)] = {'result':0,'plays':0}
    while (time.clock() - t0 < duration and results[tuple(board)]['plays'] < cutoff):
        branch = [player * np.copy(board)]
        result , branch = mc_step(branch,results,epsilon,cutoff)
    
        for i, b in enumerate(branch):
            results[tuple(b)]['plays'] +=1
            results[tuple(b)]['result'] += (-1) ** i * result
    return results[tuple(board)]['result'] / float(results[tuple(board)]['plays'])


    
def monte_carlo(board,epsilon = 0.5,duration = 1,player=1):
    plays = {}
    results = {}
    t0 = time.clock()
    plays[tuple(board)] = 0
    results[tuple(board)]=0
    
    while time.clock()-t0 < duration:
        current_player = player
        dummy_board = np.copy(board)
        branch = [(np.copy(dummy_board),current_player)]   

        
        while not game_over(dummy_board):
            options = ttt.available_moves(dummy_board)
            future_boards = [next_board(dummy_board,move,current_player) for move in options]
            
            if all(plays.get(tuple(b)) for b in future_boards):
                if random.random() > epsilon:
                    dummy_board = random.choice(future_boards)
                else:
                    #min here because you are maximizing over future boards, which the results are given in terms of the
                    #current player, i.e. the other player.
                    dummy_board = min(future_boards,key = lambda x : results[tuple(x)] / float(plays[tuple(x)])) 
                    
            
            else:
                dummy_board = random.choice(future_boards)
                plays[tuple(dummy_board)] = 0
                results[tuple(dummy_board)]=0
            current_player *= -1    
            branch.append((np.copy(dummy_board),current_player))
        

        for b,p in branch:
            plays[tuple(b)] +=1
            results[tuple(b)] += p * ttt.winner(dummy_board)
            
    return results[tuple(board)] / float(plays[tuple(board)])
            
       
        
    
    


# In[ ]:

X_sym = T.matrix()
y_sym = T.matrix()
s_sym = T.scalar()
z_sym = T.dscalar()
input_history = []
output_history = []
results = {}
move_history = []
output = lasagne.layers.get_output(value_out,X_sym)
output_det = lasagne.layers.get_output(value_out,X_sym,deterministic=True)
value_fun = theano.function([X_sym],output)
value_fun_det = theano.function([X_sym],output_det)
params = lasagne.layers.get_all_params(value_out)
objective = T.mean(lasagne.objectives.squared_error(output,y_sym))
grad = T.grad(objective,params)
exploration = 1
future_discount = 0

def epsilon(N):
    return 1 - 1. / (float(N) / 100 +1)



#flush training sets
input_history = []
output_history = []
results = {}
move_history = []


# In[ ]:

BATCH_SIZE = 256
batches_per_step = 50
training_per_step = 10
train_duration = 500
exploration = 1
exploration_min = 0.05
exploration_max = 0.95
future_discount = 0.05
minimax_str = 0
validation_str = 6
monte_carlo_duration = 0
print_freq = batches_per_step-1
valid_freq = 5
learning_speed = 0.001
updates = lasagne.updates.nesterov_momentum(grad, params, learning_rate=learning_speed,momentum = 0.9)
#updates = lasagne.updates.sgd(grad, params, learning_rate=learning_speed)
train_net = theano.function([X_sym, y_sym], objective, updates=updates)
temporal_index = 0.8
objective_total = 1000.0


# In[ ]:

t0 = time.clock()
epoch_objective_list = []

#gonna have it play against random.
#might make it play against itself

for epoch in range(train_duration):
    param_values = lasagne.layers.get_all_param_values(value_out)
    
    minimum_data = sum([param_values[i].size for i,_ in enumerate(param_values)])
    t1 = time.clock()
    for _ in range(training_per_step):
        exploration = min(exploration_max ,
                      exploration*0.99 + 0.01 *(min(exploration_max-exploration_min,objective_total) + exploration_min))
        future_discount = future_discount*0.99 + 0.01*(1 - min(1,objective_total))
        future_discount = 1
        result_X = ttt.play(nn_ai(value_fun,net = 'value',exploration = exploration),alpha_beta(minimax_str))
        board_list =get_inputs(result_X.log)
        game_length = len(result_X.log)
        input_list = [board_list[2*i] for i in range((game_length+1)/2)]
        output_list = [board_list[2*i+1] for i in range((game_length+1)/2)]
        move_list = [result_X.log[2*i] for i in range((game_length+1)/2)]
        
        
        
        
        # [monte_carlo_mod(-1 * board_list[2*i+1],results,epsilon =
        #epsilon,duration = monte_carlo_duration)  for i in range((game_length+1)/2)]
    #reward_list = [-1*monte_carlo(-1*board_list[2*i+1],epsilon=epsilon,
    #                              duration = monte_carlo_duration) for i in range((game_length+1)/2)]
#    reward_list = [ttt.winner(board_list[2*i+1]) for i in range((game_length+1)/2)]
    
        input_history = input_history+ input_list
        output_history = output_history+output_list
        move_history = move_history+move_list
    #reward_history = reward_history + reward_list


        result_O = ttt.play(alpha_beta(minimax_str),nn_ai(value_fun,net = 'value', exploration = exploration))
        board_list = get_inputs(result_O.log)
        game_length = len(result_O.log)
        input_list = [-1*board_list[2*i+1] for i in range(game_length/2)]
        output_list = [-1*board_list[2*i+2] for i in range(game_length/2)]
        move_list = [result_O.log[2*i+1] for i in range(game_length/2)]
        
        
        
        
        #[monte_carlo_mod(1 * board_list[2*i+2],results,epsilon 
        #= epsilon,duration = monte_carlo_duration)  for i in range((game_length)/2)]
    #reward_list =  [-1*monte_carlo(board_list[2*i+2],epsilon=epsilon,
     #                             duration = monte_carlo_duration) for i in range((game_length)/2)]
#    reward_list = [-1 * ttt.winner(board_list[2*i+2]) for i in range((game_length)/2)]

    
        input_history = input_history+ input_list
        output_history = output_history+output_list
        move_history = move_history+move_list
    #reward_history = reward_history + reward_list
#    for _ in range(len(input_list)):
#        reward_history.append(reward)
  

    
    if len(input_history) > 2*minimum_data:
        target_history = np.zeros(len(output_history))
        print 'Creating Targets for {} data points'.format(len(output_history))
        print '\n'
        t3 = time.clock()
        for i,item in enumerate(output_history):
            output_state = np.copy(output_history[i])
            if ttt.winner(output_state) or ttt.is_full(output_state):
                target_history[i] = ttt.winner(output_state)
            else:
            #minus because the future term is in terms of the valuation for the player, and we need a target for the 
            #opponent
            #    targets[i] = (1-future_discount) * reward_state + future_discount * get_max_future(
            #output_state,value_fun)
            #targets = np.array(targets).reshape(BATCH_SIZE,1)

                #temporal difference method
                target_history[i]= 0
                current_state = np.copy(output_state)

                depth = 0
                player = 1

                while not game_over(np.copy(current_state)):
                    current_value , next_state= get_max_future(current_state,value_fun)

                    #get_max_future calculates the min future for other player moving next
                    # so the negative player is going to want to reverse it
                    current_value = player * current_value
                    depth +=1
                    target_history[i] += (temporal_index**(depth-1))* (1-temporal_index) *current_value
                    current_state = -1 * np.copy(next_state)
                    player *= -1

                target_history[i] += temporal_index**depth * player* ttt.winner(current_state)





            #we reverse the target because we are evaulating the opponenet's position
            target_history[i] = -1 * target_history[i]
        print 'Time to create targets: {}s'.format(time.clock()-t3)
        print '\n'
        for j in range(batches_per_step):
            t2=time.clock()
            targets = np.zeros((BATCH_SIZE,1))
            training = np.zeros((BATCH_SIZE,9))
            index_pool = range(len(input_history))
            random.shuffle(index_pool)
            objectives = []
            
            while len(index_pool) > BATCH_SIZE:
                sample_indices = [index_pool.pop() for _ in xrange(BATCH_SIZE)]
                #Should try to use generators
                

                
            
            
                for k in range(BATCH_SIZE):
                    #train it on output_history evaluated by the opponent
                    training[k] = (-1*np.copy(output_history[sample_indices[k]]))
                    targets[k] = target_history[sample_indices[k]]
                    #training[i] = training[i].reshape(1,9)

        #               reward_state = reward_history[sample_indices[i]]
                    #move_state = move_history[sample_indices[i]]
                    #reward_state= -1 * results[tuple(-1 * output_state)]['result'] / float(
                    #                   results[tuple(-1 * output_state)]['plays'])

                    #needs to evaluate to the result of the opponent's board
                    #If move is a winning move, reward_state will evaluate to 1. Since the board is reversed, this is the
                    #reverse evaluation



                objectives.append(train_net(training,targets))
                #break to only do 1 batch per run.
            objective_total = np.mean(objectives)
            if j%print_freq ==0:
                print (('Epoch {:5d}, Pass number {}, objective: {:0.5f}, exploration: {:0.2f}, '+
                       'step duration: {:1.3f}s').format(
                       epoch,j+1,float(objective_total),exploration,time.clock()-t2))
                
            epoch_objective_list.append([epoch,j+1,objective_total])
        
        print 'epoch duration {:2.2f}s'.format(time.clock()-t1)
        if epoch%valid_freq==0:
            print ' '
            print ' '
            test_result = {'wins':0,'ties':0,'losses':0}
            for j in range(100):
                result = ttt.play(nn_ai(value_fun_det,'value'),alpha_beta(validation_str))
                if result.winner ==1:
                    test_result['wins'] +=1
                if result.winner == 0:
                    test_result['ties'] +=1
                if result.winner == -1:
                    test_result['losses'] +=1


            print 'As X, neural network has a score of {:3d}-{:3d}-{:3d} vs {}-depth minimax'.format(test_result['wins'],
                                                                   test_result['ties'],test_result['losses'],validation_str)
    
            test_result = {'wins':0,'ties':0,'losses':0}

            for j in range(100):
                result = ttt.play(alpha_beta(validation_str),nn_ai(value_fun_det,'value'))
                if result.winner ==1:
                    test_result['losses'] +=1
                if result.winner == 0:
                    test_result['ties'] +=1
                if result.winner == -1:
                    test_result['wins'] +=1

            print 'As O, neural network has a score of {:3d}-{:3d}-{:3d} vs {}-depth minimax'.format(test_result['wins'],
                                                                test_result['ties'],test_result['losses'],validation_str)
            print ' '
            print 'elapsed time: {:3.3f}s'.format(time.clock()-t0)
            print ' '
    else:
        if epoch%print_freq==0:
            print ('Learning step {:5d}, training size {:4d}, step duration: {:1.3f}s'.format(
                    epoch,len(input_history),time.clock()-t1))


print('\n')

#for epoch in range(10):

        
#    t_epoch = time.clock()

#    result_list = [tourney(output) for i in range(BATCH_SIZE)]
#    fitness_list = [fitness(result) for result in result_list]
    
#    score= f_train(fitness_list)
       
#    t1=time.clock()-t_epoch
    
#    print('Epoch {}, duration {:.01f} seconds'.format(
#            epoch+1, t1))
#    print('Record is: {}-{}-{} with a score of {}'.format(sum([result['wins'] for result in result_list]),
#                                                          sum([result['ties'] for result in result_list]),
#                                                          sum([result['losses'] for result in result_list]),
#                                                          sum([fitness for fitness in fitness_list])))
#    print('mean score is {}'.format(score))
print('total time for neural network is {:.01f} seconds'.format(time.clock()-t0))


# In[ ]:

#save the chromes
np.savez('TD_ttt_nn',lasagne.layers.get_all_param_values(value_out))


# In[ ]:

#load the chromes

loaded_param = list(np.load('TD_ttt_nn.npz')['arr_0'])


# In[ ]:

print(ttt.play(ttt.player(),nn_ai(value_fun,'value')).winner)


# In[ ]:

result = ttt.play(nn_ai(value_fun,'value'),ttt.player())
print(result.board,result.winner)


# In[ ]:

minimax_str = 0

for epoch in range(10000):
    t1 = time.clock()
    result_X = ttt.play(alpha_beta(minimax_str),alpha_beta(minimax_str))
    board_list =get_inputs(result_X.log)
    game_length = len(result_X.log)
    input_list = [board_list[2*i] for i in range((game_length+1)/2)]
    output_list = [board_list[2*i+1] for i in range((game_length+1)/2)]
    move_list = [result_X.log[2*i] for i in range((game_length+1)/2)]
#    reward_list = [-1*monte_carlo(-1*board_list[2*i+1],epsilon=epsilon,
#                                  duration = monte_carlo_duration) for i in range((game_length+1)/2)]
    reward_list = [alpha_beta_move(board_list[2*i],'X',7)[0] for i in range((game_length+1)/2)]
    
    input_history = input_history+ input_list
    output_history = output_history+output_list
    move_history = move_history+move_list
    reward_history = reward_history + reward_list


    result_O = ttt.play(alpha_beta(minimax_str),alpha_beta(minimax_str))
    board_list = get_inputs(result_O.log)
    game_length = len(result_O.log)
    input_list = [-1*board_list[2*i+1] for i in range(game_length/2)]
    output_list = [-1*board_list[2*i+2] for i in range(game_length/2)]
    move_list = [result_O.log[2*i+1] for i in range(game_length/2)]
#    reward_list =  [-1*monte_carlo(board_list[2*i+2],epsilon=epsilon,
#                                  duration = monte_carlo_duration) for i in range((game_length)/2)]
    reward_list = [alpha_beta_move(-1*board_list[2*i+1],'X',7)[0] for i in range((game_length)/2)]

    
    input_history = input_history+ input_list
    output_history = output_history+output_list
    move_history = move_history+move_list
    reward_history = reward_history + reward_list
    if epoch%10 ==0:
        print epoch


# In[ ]:

'''
pd.DataFrame.to_csv(pd.DataFrame(input_history),'minimax_input_history.csv')
pd.DataFrame.to_csv(pd.DataFrame(output_history),'minimax_output_history.csv')
pd.DataFrame.to_csv(pd.DataFrame(move_history),'minimax_move_history.csv')
pd.DataFrame.to_csv(pd.DataFrame(reward_history),'minimax_reward_history.csv')
'''


# In[ ]:

BATCH_SIZE = 256
batches_per_step = 1
train_duration = 20000
exploration = 1
exploration_min = 0.05
future_discount = 0
minimax_str = 0
validation_str = 2
epsilon = 0.5
monte_carlo_duration = 1
objective_total = 1000.0
print_freq = 100
valid_freq = 2000
valid_size = 100
learning_speed = 0.01
updates = lasagne.updates.nesterov_momentum(grad, params, learning_rate=learning_speed,momentum = 0.9)
train_net = theano.function([X_sym, y_sym], objective, updates=updates)


# In[ ]:

input_history = list(pd.DataFrame.from_csv('minimax_input_history.csv').values)
output_history = list(pd.DataFrame.from_csv('minimax_output_history.csv').values)
move_history = list(pd.DataFrame.from_csv('minimax_move_history.csv').values)
reward_history = list(pd.DataFrame.from_csv('minimax_reward_history.csv').values)
#reward_history = [item[0] for item in reward_history]


# In[ ]:

t0 = time.clock()

#gonna have it play against random.
#might make it play against itself

for epoch in range(train_duration):
    t1 = time.clock()
    if len(input_history) > 2*BATCH_SIZE:
        for _ in range(batches_per_step):
            targets = [0]*BATCH_SIZE
            inputs = np.zeros((BATCH_SIZE,9))
            sample_indices = random.sample(range(len(input_history)),BATCH_SIZE)
            #Should try to use generators

            for i in range(BATCH_SIZE):
                #this seems confusing, but the rewards are based on output from the net
                inputs[i] = (np.copy(input_history[sample_indices[i]]))
#               inputs[i] = inputs[i].reshape(1,9)
                output_state = output_history[sample_indices[i]]
#               reward_state = reward_history[sample_indices[i]]
                move_state = move_history[sample_indices[i]]
                reward_state=reward_history[sample_indices[i]]
                targets[i] = reward_state
               
            objective_total = train_net(inputs,targets)
        if epoch%print_freq ==0:
            print ('Learning step {:5d}, objective: {:0.5f}, step duration: {:1.3f}s'.format(
                    epoch,float(objective_total),time.clock()-t1))
        
        if epoch%valid_freq==0:
            print ' '
            print ' '
            test_result = {'wins':0,'ties':0,'losses':0}
            for j in range(valid_size):
                result = ttt.play(nn_ai(value_fun,'value'),alpha_beta(validation_str))
                if result.winner ==1:
                    test_result['wins'] +=1
                if result.winner == 0:
                    test_result['ties'] +=1
                if result.winner == -1:
                    test_result['losses'] +=1


            print 'As X, neural network has a score of {:3d}-{:3d}-{:3d} vs {}-depth minimax'.format(test_result['wins'],
                                                                   test_result['ties'],test_result['losses'],validation_str)
    
            test_result = {'wins':0,'ties':0,'losses':0}

            for j in range(valid_size):
                result = ttt.play(alpha_beta(validation_str),nn_ai(value_fun,'value'))
                if result.winner ==1:
                    test_result['losses'] +=1
                if result.winner == 0:
                    test_result['ties'] +=1
                if result.winner == -1:
                    test_result['wins'] +=1

            print 'As O, neural network has a score of {:3d}-{:3d}-{:3d} vs {}-depth minimax'.format(test_result['wins'],
                                                                        test_result['ties'],test_result['losses'],validation_str)
            print ' '
            print 'elapsed time: {:3.3f}s'.format(time.clock()-t0)
            print ' '
    else:
        if epoch%print_freq==0:
            print ('Learning step {:5d}, training size {:4d}, step duration: {:1.3f}s'.format(
                    epoch,len(input_history), time.clock() - t1))


# In[ ]:

#save the chromes
np.savez('supervised_single_ttt_nn',lasagne.layers.get_all_param_values(value_out))


# In[ ]:

loaded_param = list(np.load('supervised_single_ttt_nn.npz')['arr_0'])


# In[ ]:

for i,item in enumerate(output_history):
    output_state = np.copy(output_history[i])
    if ttt.winner(output_state) or ttt.is_full(output_state):
        target_history[i] = ttt.winner(output_state)
    else:
    #minus because the future term is in terms of the valuation for the player, and we need a target for the 
    #opponent
    #    targets[i] = (1-future_discount) * reward_state + future_discount * get_max_future(
    #output_state,value_fun)
    #targets = np.array(targets).reshape(BATCH_SIZE,1)

        #temporal difference method
        target_history[i]= 0
        current_state = np.copy(output_state)

        depth = 0
        player = 1

        while not game_over(np.copy(current_state)):
            current_value , next_state= get_max_future(current_state,value_fun)

            #get_max_future calculates the min future for other player moving next
            # so the negative player is going to want to reverse it
            current_value = player * current_value
            depth +=1
            target_history[i] += (temporal_index**(depth-1))* (1-temporal_index) *current_value
            current_state = -1 * np.copy(next_state)
            player *= -1

        target_history[i] += temporal_index**depth * player* ttt.winner(current_state)





    #we reverse the target because we are evaulating the opponenet's position
    target_history[i] = -1 * target_history[i]


# In[ ]:

output_state = np.copy(output_history[6551])
if ttt.winner(output_state) or ttt.is_full(output_state):
    test = ttt.winner(output_state)
else:
#minus because the future term is in terms of the valuation for the player, and we need a target for the 
#opponent
#    targets[i] = (1-future_discount) * reward_state + future_discount * get_max_future(
#output_state,value_fun)
#targets = np.array(targets).reshape(BATCH_SIZE,1)

    #temporal difference method
    test= 0
    current_state = np.copy(output_state)

    depth = 0
    player = 1

    while not game_over(np.copy(current_state)):
        current_value , next_state= get_max_future(current_state,value_fun)

        #get_max_future calculates the min future for other player moving next
        # so the negative player is going to want to reverse it
        current_value = player * current_value
        depth +=1
        test += (temporal_index**(depth-1))* (1-temporal_index) *current_value
        current_state = -1 * np.copy(next_state)
        player *= -1

    test += temporal_index**depth * player* ttt.winner(current_state)





#we reverse the target because we are evaulating the opponenet's position
test = -1 * test
print test


# In[ ]:

def plot_im(im,num_channels):
    for j in range(num_channels):    
        plt.subplot(1, num_channels, j+1)
        plt.imshow(im[0][j], interpolation='nearest')
        plt.axis('off')
        
input_grid = lasagne.layers.get_output(value_in, X_sym)
layer1_grid = lasagne.layers.get_output(value_hid1, X_sym)
layer2_grid = lasagne.layers.get_output(value_hid2, X_sym)
layer3_grid = lasagne.layers.get_output(value_hid3, X_sym)
output_grid = lasagne.layers.get_output(value_out, X_sym)

training_index = 130
vmin = -1
vmax = +1
color_map = plt.cm.bwr

fig, axes = plt.subplots(1,5,figsize = (20,6))




# plt.setp(axes, xticks=[], xticklabels=[],
#         yticks=[],yticklabels=[])

# plt.xticks([])

# axes.tick_params(
#     axis='x',          # changes apply to the x-axis
#     which='both',      # both major and minor ticks are affected
#     bottom='off',      # ticks along the bottom edge are off
#     top='off',         # ticks along the top edge are off
#     labelbottom='off') # labels along the bottom edge are off

f_filter = theano.function([X_sym], input_grid)
im = f_filter(training[training_index:training_index+1])

axes[0].matshow(im.reshape((3,3)),cmap=color_map,vmin = vmin,vmax=vmax)

axes[0].plot([1.5, 1.5], [-0.75, 2.75], 'black', lw=4)
axes[0].plot([0.5, 0.5], [-0.75, 2.75], 'black', lw=4)
axes[0].plot([-0.75, 2.75], [1.5, 1.5], 'black', lw=4)
axes[0].plot([-0.75, 2.75], [0.5, 0.5], 'black', lw=4)
axes[0].set_title('Tic Tac Toe Position',fontsize = 20)

f_filter = theano.function([X_sym], layer1_grid)
im = f_filter(training[training_index:training_index+1])

axes[1].matshow(im.reshape((6,6)),cmap=color_map,vmin = vmin,vmax=vmax)

f_filter = theano.function([X_sym], layer2_grid)
im = f_filter(training[training_index:training_index+1])

axes[2].matshow(im.reshape((6,6)),cmap=color_map,vmin = vmin,vmax=vmax)


f_filter = theano.function([X_sym], layer3_grid)
im = f_filter(training[training_index:training_index+1])
axes[2].set_title('Network Node Activations',fontsize=20)
axes[3].matshow(im.reshape((6,6)),cmap=color_map,vmin = vmin,vmax=vmax)


f_filter = theano.function([X_sym], output_grid)
im = f_filter(training[training_index:training_index+1])

axes[4].matshow(im,cmap=color_map,vmin = vmin,vmax=vmax)
axes[4].set_title('Predicted Winner',fontsize=20)

for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])
    
plt.savefig('network_weights.png')
plt.show()
print im

