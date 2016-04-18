
# coding: utf-8

# In[ ]:

import numpy as np
import pandas as pd
import lasagne
import theano
import theano.tensor as T
import random
import time
import connect_four as cccc
import seaborn
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')


# In[ ]:

#initialization

generations = []

hidden_units = 16



value_in = lasagne.layers.InputLayer(shape=(None,42))

value_hid1 = lasagne.layers.DenseLayer(value_in, num_units=hidden_units,W=lasagne.init.GlorotUniform(),
                                          nonlinearity = lasagne.nonlinearities.tanh)
value_hid2 = lasagne.layers.DenseLayer(value_hid1, num_units=hidden_units,W=lasagne.init.GlorotUniform(),
                                          nonlinearity = lasagne.nonlinearities.tanh)

value_hid3 = lasagne.layers.DenseLayer(value_hid2, num_units=hidden_units,W=lasagne.init.GlorotUniform(),
                                          nonlinearity = lasagne.nonlinearities.tanh)

value_drop1 = lasagne.layers.DropoutLayer(value_hid3,p=0.5)

value_out = lasagne.layers.DenseLayer(value_drop1,
                                  num_units=1, nonlinearity = lasagne.nonlinearities.tanh)



# policy_in = lasagne.layers.InputLayer(shape=(1,9))

# #l_drop1 = lasagne.layers.DropoutLayer(l_shape,p=0.2)

# policy_hid1 = lasagne.layers.DenseLayer(policy_in, num_units=hidden_units,W=lasagne.init.GlorotUniform(),
#                                           nonlinearity = lasagne.nonlinearities.rectify)

# #l_drop2 = lasagne.layers.DropoutLayer(l_hid1,p=0.5)

# policy_out = lasagne.layers.DenseLayer(policy_hid1,
#                                   num_units=9,
#                                   nonlinearity=lasagne.nonlinearities.softmax)


# In[ ]:

def value_move(board,active_turn,output_fun,exploration = 0):
    board = board.reshape((1,42))
    X_sym = theano.tensor.matrix()
    y_sym = theano.tensor.ivector()

    dummy_board = active_turn * board[:]
    options = cccc.available_moves(dummy_board)
    
    
    if exploration > random.random():
        move = random.choice(options) 
    else:
        move_values = np.zeros(42)
        for move in options:
            dummy_board = active_turn * board[:]
            dummy_board[0][move] = 1
            move_values[move] = -1 * output_fun(-1* dummy_board)
        

        available_move_values = np.array([move_values[move] for move in options])
        
        move = options[available_move_values.argmax(-1)]
    return move + 1


class nn_ai:
    
    def __init__(self,output_fun, net = 'value',exploration = 0):
        self.output_fun = output_fun
        self.exploration = exploration
        self.net = net
    
    def make_move(self,board,active_turn):
#         if self.net == 'policy':
#             move = policy_move(board,active_turn,self.output_fun,self.exploration)
        if self.net == 'value':
            move = value_move(board,active_turn,self.output_fun,self.exploration)
        return move
    
def alpha_beta_move(board,active_turn,depth,alpha = 2):
    swap_dict = {1:-1,-1:1}
    dummy_board = np.zeros((6,7))
    dummy_board[:] = board[:]
    options = cccc.available_moves(board)
    random.shuffle(options)
    if len(options) == 1:
        dummy_board[np.where(dummy_board[:,options[0]]==0)[0][-1],options[0]] = active_turn
        if cccc.winner(dummy_board):
            return (1,options[0]+1)
        else:
            return (0,options[0]+1)
    if depth ==0:
        return (0, options[np.random.randint(len(options))]+1)

    best_value = -2
    candidate_move = None
    for x in options:
        height = np.where(dummy_board[:,x]==0)[0][-1]
        dummy_board[height,x] = active_turn
        if cccc.winner(dummy_board):
            return (1, x+1)
        (opp_value,opp_move) = alpha_beta_move(dummy_board,swap_dict[active_turn],depth-1,-best_value)
        if -opp_value > best_value:
            candidate_move = x+1
            best_value = -opp_value
        if -opp_value >= alpha:
            #print (options, x, best_value, alpha)
            break
        dummy_board[height,x] = 0

    return (best_value, candidate_move)

class alpha_beta:
    def __init__(self,depth):
        self.depth = depth
    def make_move(self,board,active_turn):
        #print (board,active_turn,self.depth)
        return alpha_beta_move(board,active_turn,self.depth)[1]

def get_max_future(future_board,value_fun):
    options = cccc.available_moves(future_board)
    dummy_board = np.copy(future_board)
    move_values = np.zeros(7)
    for move in options:
        dummy_board = np.copy(future_board)
        dummy_board[np.where(dummy_board[:,move]==0)[0][-1],move] = -1
        # dummy_board = dummy_board.reshape(1,42)
        if cccc.winner(dummy_board):
            move_values[move] = cccc.winner(dummy_board)
        else:
            reshapable = np.copy(dummy_board)
            reshapable = reshapable.reshape(1,42)
            move_values[move] = value_fun(reshapable)
    
    available_move_values = np.array([move_values[move] for move in options])
    dummy_board = np.copy(future_board)
    options_index = np.argmin(available_move_values)
    dummy_board[np.where(dummy_board[:,options[options_index]]==0)[0][-1],options[options_index]] = -1
    return np.amin(available_move_values), dummy_board

def get_inputs(log):
    boards = []
    piece = 1
    board = np.zeros((6,7))
    boards.append(np.copy(board))
    for move in log:
        board[np.where(board[:,move-1]==0)[0][-1],move-1] = piece
        piece = -piece
        boards.append(np.copy(board))
    return boards



def game_over(board):
    return cccc.winner(board) or cccc.is_full(board)


# # Reinforcement Learning

# In[ ]:

X_sym = T.matrix()
y_sym = T.matrix()

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

#flush training sets
input_history = []
output_history = []
results = {}
move_history = []


# In[ ]:

BATCH_SIZE = 256
batches_per_step = 500
training_per_step =100
train_duration = 20
exploration = 1
exploration_min = 0.05
exploration_max = 0.95
future_discount = 0.5
minimax_str = 0
validation_str = 2
monte_carlo_duration = 0
print_freq = batches_per_step-1
valid_freq = 5
learning_speed = 0.01
updates = lasagne.updates.nesterov_momentum(grad, params, learning_rate=learning_speed,momentum = 0.9)
#updates = lasagne.updates.sgd(grad, params, learning_rate=learning_speed)
train_net = theano.function([X_sym, y_sym], objective, updates=updates)
temporal_index = 0.8
objective_total = 1000.0


# In[ ]:

t0 = time.clock()
epoch_objective_list = []

param_values = lasagne.layers.get_all_param_values(value_out)
    
minimum_data = sum([param_values[i].size for i,_ in enumerate(param_values)])
while(len(output_history) < 2 * minimum_data):
    exploration = min(exploration_max ,
                  exploration*0.99 + 0.01 *(min(exploration_max-exploration_min,objective_total) + exploration_min))
    future_discount = future_discount*0.99 + 0.01*(1 - min(1,objective_total))
    # future_discount = 1
    result_X = cccc.play(nn_ai(value_fun,net = 'value',exploration = exploration),alpha_beta(minimax_str))
    board_list =get_inputs(result_X.log)
    game_length = len(result_X.log)
    input_list = [board_list[2*i] for i in range((game_length+1)/2)]
    output_list = [board_list[2*i+1] for i in range((game_length+1)/2)]
    move_list = [result_X.log[2*i] for i in range((game_length+1)/2)]



    input_history = input_history+ input_list
    output_history = output_history+output_list
    move_history = move_history+move_list
    #reward_history = reward_history + reward_list


    result_O = cccc.play(alpha_beta(minimax_str),nn_ai(value_fun,net = 'value', exploration = exploration))
    board_list = get_inputs(result_O.log)
    game_length = len(result_O.log)
    input_list = [-1*board_list[2*i+1] for i in range(game_length/2)]
    output_list = [-1*board_list[2*i+2] for i in range(game_length/2)]
    move_list = [result_O.log[2*i+1] for i in range(game_length/2)]
for epoch in range(train_duration):
    
    t1 = time.clock()
     
    if len(input_history) > minimum_data:
        target_history = np.zeros(len(output_history))
        print 'Creating Targets for {} data points'.format(len(output_history))
        print '\n'
        t3 = time.clock()
        for i,item in enumerate(output_history):
            output_state = np.copy(output_history[i])
            if cccc.winner(output_state) or cccc.is_full(output_state):
                target_history[i] = cccc.winner(output_state)
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

                target_history[i] += temporal_index**depth * player* cccc.winner(current_state)





            #we reverse the target because we are evaulating the opponenet's position
            target_history[i] = -1 * target_history[i]
        print 'Time to create targets: {}s'.format(time.clock()-t3)
        print '\n'
        for j in range(batches_per_step):
            t2=time.clock()
            targets = np.zeros((BATCH_SIZE,1))
            training = np.zeros((BATCH_SIZE,42))
            index_pool = range(len(input_history))
            random.shuffle(index_pool)
            objectives = []
            
            while len(index_pool) > BATCH_SIZE:
                sample_indices = [index_pool.pop() for _ in xrange(BATCH_SIZE)]
                #Should try to use generators
                

                
            
            
                for k in range(BATCH_SIZE):
                    #train it on output_history evaluated by the opponent
                    training[k] = (-1*np.copy(output_history[sample_indices[k]])).reshape(1,42)
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
                result = cccc.play(nn_ai(value_fun_det,'value'),alpha_beta(validation_str))
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
                result = cccc.play(alpha_beta(validation_str),nn_ai(value_fun_det,'value'))
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

f = open('TD_connect_four.txt','w')

for item in epoch_objective_list:
    print>>f , item


# # Supervised Learning

# In[ ]:

raw_training = pd.read_csv('connect-4.csv',header = None)


# In[ ]:

training_set = []
targets_set = []
for index,row in raw_training.iterrows():
    input_dict = {'b':0,'x':1,'o':-1}
    game_state_dict = {'win':1,'draw':0,'loss':-1}
    game_state = game_state_dict[row[42]]
    row = row[:-1].apply(lambda x: input_dict[x])
    board = np.flipud(np.reshape(np.array(row),(6,7),order = 'F'))
    if np.sum(board)==1:
        board = -1 * np.copy(board)
    
    training_set.append(np.copy(board).reshape(1,42))
    targets_set.append(game_state)

dummy_list = list(zip(training_set,targets_set))
random.shuffle(dummy_list)
training_set,targets_set = zip(*dummy_list)

val_training_set = training_set[:len(training_set)/2]
val_targets_set = targets_set[:len(targets_set)/2]
training_set = training_set[len(training_set)/2:]
targets_set = targets_set[len(targets_set)/2:]


# In[ ]:

X_sym = T.matrix()
y_sym = T.matrix()

output = lasagne.layers.get_output(value_out,X_sym)
output_det = lasagne.layers.get_output(value_out,X_sym,deterministic=True)
value_fun = theano.function([X_sym],output)
value_fun_det = theano.function([X_sym],output_det)
params = lasagne.layers.get_all_params(value_out)
objective = T.mean(lasagne.objectives.squared_error(output,y_sym))
grad = T.grad(objective,params)



# In[ ]:

BATCH_SIZE = 256
train_duration = 500
minimax_str = 0
validation_str = 2
valid_freq = 100
learning_speed = 0.01
updates = lasagne.updates.nesterov_momentum(grad, params, learning_rate=learning_speed,momentum = 0.9)
#updates = lasagne.updates.sgd(grad, params, learning_rate=learning_speed)
train_net = theano.function([X_sym, y_sym], objective, updates=updates)
val_net = theano.function([X_sym, y_sym], objective)


# In[ ]:

t0 = time.clock()

epoch_objective_list = []

for epoch in range(train_duration):


    t2=time.clock()

    objectives = []
    index_pool = range(len(targets_set))
    random.shuffle(index_pool)
    while len(index_pool) > BATCH_SIZE:
        
        sample_indices = [index_pool.pop() for _ in xrange(BATCH_SIZE)]

        targets = np.zeros((BATCH_SIZE,1))
        training = np.zeros((BATCH_SIZE,42))
        for k in range(BATCH_SIZE):
            training[k] = np.copy(training_set[sample_indices[k]])
            targets[k] = targets_set[sample_indices[k]]

        objectives.append(train_net(training,targets))
        #break to only do 1 batch per run.
    objective_total = np.mean(objectives)


    objectives = []
    index_pool = range(len(val_targets_set))
    random.shuffle(index_pool)
    while len(index_pool) > BATCH_SIZE:
        
        sample_indices = [index_pool.pop() for _ in xrange(BATCH_SIZE)]

        targets = np.zeros((BATCH_SIZE,1))
        training = np.zeros((BATCH_SIZE,42))
        for k in range(BATCH_SIZE):
            training[k] = np.copy(training_set[sample_indices[k]])
            targets[k] = targets_set[sample_indices[k]]

        objectives.append(val_net(training,targets))
        #break to only do 1 batch per run.
    val_total = np.mean(objectives)

    print (('Epoch {:5d},objective: {:0.5f}, validation: {:0.5f}, step duration: {:1.3f}s').format(
            epoch,float(objective_total),float(val_total),time.clock()-t2))
    
        
    
    
    
    epoch_objective_list.append([epoch,objective_total,val_total])

    # print 'epoch duration {:2.2f}s'.format(time.clock()-t2)
    if epoch%valid_freq==0:
        print ' '
        print ' '
        test_result = {'wins':0,'ties':0,'losses':0}
        for j in range(100):
            result = cccc.play(nn_ai(value_fun_det,'value'),alpha_beta(validation_str))
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
            result = cccc.play(alpha_beta(validation_str),nn_ai(value_fun_det,'value'))
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

import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')

f = open('supervised_connect_four.txt','r')
plot_list = f.read().splitlines()
plot_list = np.array([[float(x) for x in item[1:-1].split(',')] for item in plot_list])

f = open('supervised_tictactoe.txt','r')
plot_list2 = f.read().splitlines()
plot_list2 = np.array([[float(x) for x in item[1:-1].split(',')] for item in plot_list2])


# In[ ]:

plt.figure(figsize=(16,9))
plt.plot(plot_list[:,0],plot_list[:,1],color = 'blue',linewidth=3)
plt.plot(plot_list[:,0],plot_list[:,2],color = 'green')
plt.plot(50*plot_list2[:,0]+plot_list2[:,1]-900,plot_list2[:,2],color = 'red')
plt.title('Deep Learning',fontsize=32)
plt.xlabel('Epoch',fontsize=24)
plt.ylabel('Mean Squared Error',fontsize = 24)
plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)
ax = plt.subplot(111)    
ax.spines["top"].set_visible(False)    
# ax.spines["bottom"].set_visible(False)    
ax.spines["right"].set_visible(False)    
# ax.spines["left"].set_visible(False)    
  
# Ensure that the axis ticks only show up on the bottom and left of the plot.    
# Ticks on the right and top of the plot are generally unnecessary chartjunk.    
ax.get_xaxis().tick_bottom()    
ax.get_yaxis().tick_left()
plt.text(410, 0.61,"Supervised learning with connect four - loss",fontsize=20,color='blue')
plt.text(410, 0.56,"Supervised learning with connect four - validation",fontsize=20,color='green')
plt.text(410, 0.51,"Reinforcement learning with tic tac toe - loss",fontsize=20,color='red')
plt.savefig('supervised_c4.png')
plt.show


# In[ ]:

f = open('TD_connect_four.txt','r')
plot_list = f.read().splitlines()
plot_list = np.array([[float(x) for x in item[1:-1].split(',')] for item in plot_list])


# In[ ]:

plt.plot(500*plot_list[:,0]+plot_list[:,1],plot_list[:,2])
plt.show


# In[ ]:

f = open('supervised_tictactoe.txt','r')
plot_list = f.read().splitlines()
plot_list = np.array([[float(x) for x in item[1:-1].split(',')] for item in plot_list])


# In[ ]:

plt.plot(50*plot_list[:,0]+plot_list[:,1],plot_list[:,2],color = 'red')
plt.title('Reinforcement Learning for Tic Tac Toe',fontsize=16)
plt.xlabel('Epoch',fontsize=16)
plt.ylabel('Mean Squared Error',fontsize = 16)
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)
ax = plt.subplot(111)    
ax.spines["top"].set_visible(False)    
# ax.spines["bottom"].set_visible(False)    
ax.spines["right"].set_visible(False)    
# ax.spines["left"].set_visible(False)    
ax.get_xaxis().tick_bottom()    
ax.get_yaxis().tick_left()
plt.savefig('reinforcement_ttt.png')
plt.show
  


# In[ ]:



