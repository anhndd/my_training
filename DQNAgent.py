from keras.layers import Input, Conv2D, Flatten, Dense, LeakyReLU, Average, Add, Dot, Subtract, Multiply
from keras.models import Model
from keras.optimizers import Adam
from collections import deque
import TargetDQNAgent
import numpy as np
import random
import tensorflow as tf
import os
import constants
import time
import math
import keras.backend as K

class DQNAgent:
    def __init__(self, M, action_size, B):
        
        # Duc Anh Implement:
        # self.memory = deque(maxlen=M)
        
        # Duy Do:
        # TODO: config again.
        self.Num_Exploration = constants.Num_Exploration	    # 2000 steps to explore.                                      
        self.Num_Training    = constants.Num_Training           # TODO?
        self.Num_Testing     = constants.Num_Testing
        self.progress = ''
        self.step = 0

        self.minibatch_size = B
        self.start_epsilon = 1
        self.end_epsilon = 0.01
        self.step_epsilon = 10000
        self.epsilon_decay = (self.start_epsilon - self.end_epsilon) / self.step_epsilon
        self.tp = 2000          # pre-training steps
        self.alpha = 0.0001     # target network update rate
        self.gamma = 0.99       # discount factor
        self.epsilon_r = 0.0001 # learning rate
        self.Beta = 0.01        # Leaky ReLU
        self.action_size = action_size
        self.model = self._build_model()
        self.targetDQN = TargetDQNAgent.TargetDQNAgent(self.action_size)

        # Parameters for PER
        self.eps = 0.00001
        self.alpha_per = 0.6
        self.beta_init = 0.4
        self.beta = self.beta_init

        # transform to deque(maxlen=M);
        self.TD_list = np.array([])
        self.Num_batch = 64
        self.max_len_replay_memory = M
        self.replay_memory = deque(maxlen=M)

        # log loss plot
        self.loss_plot = []
        self.step_plot = []

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        input_1 = Input(shape=(60, 60, 2))
        input_2 = Input(shape=(self.action_size, self.action_size))
        x1 = Conv2D(32, (4, 4), strides=(2, 2),padding='Same', activation=LeakyReLU(alpha=self.Beta))(input_1)
        x1 = Conv2D(64, (2, 2), strides=(2, 2),padding='Same', activation=LeakyReLU(alpha=self.Beta))(x1)
        x1 = Conv2D(128, (2, 2), strides=(1, 1),padding='Same', activation=LeakyReLU(alpha=self.Beta))(x1)
        x1 = Flatten()(x1)
        x1 = Dense(128, activation=LeakyReLU(alpha=self.Beta))(x1)
        x1_value = Dense(64, activation=LeakyReLU(alpha=self.Beta))(x1)
        value = Dense(1, activation=LeakyReLU(alpha=self.Beta))(x1_value)
        x1_advantage = Dense(64, activation=LeakyReLU(alpha=self.Beta))(x1)
        advantage = Dense(self.action_size, activation=LeakyReLU(alpha=self.Beta))(x1_advantage)

        A = Dot(axes=1)([input_2, advantage])
        A_subtract = Subtract()([advantage, A])

        Q_value = Add()([value, A_subtract])

        model = Model(inputs=[input_1, input_2], outputs=[Q_value])
        model.compile(optimizer= Adam(lr=self.epsilon_r), loss='mse')

        return model

    def store_tuple(self, state, action, reward, next_state, terminal):
        # if replay_memory and TD_list  are full:
        if len(self.replay_memory) >= self.max_len_replay_memory:
            # step 1: replay_memory (deuque): no need to delete first element.
            # step 2: TD_list: delete first element.
            self.TD_list = np.delete(self.TD_list, 0)

        # if, agent is exploring:
        if self.progress == 'Exploring':
            # save tuple
            self.replay_memory.append([state, action, reward, next_state, terminal])
            # save TD_error (for PER) 
            # note: reward ~ td_error when agent is exploring.
            self.TD_list = np.append(self.TD_list, pow((abs(reward) + self.eps), self.alpha_per))

        # if, agent is trained.
        elif self.progress == 'Training':
            self.replay_memory.append([state, action, reward, next_state, terminal])
			# ################################################## PER ############################################################
            # cal TD_error:
            Q_value_comma = self.model.predict(next_state)[0]                                       # Q-value(s') -- PrimaryModel
            a_comma = np.argmax(Q_value_comma)                                                      # pick a' cause maxQ-value(s')
            Q_target = reward + self.gamma * self.targetDQN.model.predict(next_state)[0][a_comma]   # a number
            target_f = self.model.predict(state)                                                    #  Q value Q(s,a,theta)
            Q_value = target_f[0][action]
            # append TD_error:
            self.TD_list = np.append(self.TD_list, pow((abs(Q_target-Q_value) + self.eps), self.alpha_per))
			# ###################################################################################################################

    # select action random || by model.
    def select_action_v2(self,state, tentative_act_dec):

        print ('self.start_epsilon: '+ str(self.start_epsilon))
        if np.random.rand() <= self.start_epsilon:
            print('action by random')
            choices = np.where(tentative_act_dec[0] == 1)[0]
            print tentative_act_dec[0], choices, random.choice(choices)
            return random.choice(choices)
        else:
            print('action by model')
            act_values = self.model.predict(state)[0]
            choices = np.where(tentative_act_dec[0] == 1)[0]
            output = act_values[choices]
            return np.argmax(output)


    def select_action(self,state, tentative_act_dec):
        # print tentative_act_dec, tentative_act_dec[0], 'max = ',np.argmax(tentative_act_dec[0])
        print ('self.start_epsilon: '+ str(self.start_epsilon))
        if (tentative_act_dec[0][1] == 0):
            return np.argmax(tentative_act_dec[0])
        if np.random.rand() <= self.start_epsilon:
            print('action by random')
            return random.randrange(self.action_size)
        else:
            print('action by model')
            act_values = self.model.predict(state)
            return np.argmax(act_values[0])  # returns action
    
    # TODO: verify this function
    # get minibatch
    def get_prioritized_minibatch(self):
                '''
		        # TD_normalized = calculate probs of each exp (through TD_list) TD_list = [1,2] >> TD_normalized = [0.3333,0.666]
                # TD_list? 
                    + if (exploring):   TD_List.append    (reward+esp)^alpha_per
                    + if (training):    TD_List.append      (td_error + esp)^alpha_per
                '''
                TD_normalized = self.TD_list / np.linalg.norm(self.TD_list, 1)

                # curcumulate sum
                TD_sum = np.cumsum(TD_normalized) 

				# Get importance sampling weights
                # (N*P(i))^-beta
                weight_is = np.power((self.max_len_replay_memory * TD_normalized), - self.beta)
                # cal probs.
                weight_is = weight_is / np.max(weight_is)
                # Select mini batch and importance sampling weights
                minibatch = []
                batch_index = []
                w_batch = []
                for i in range(self.minibatch_size):
                    rand_batch = random.random()                                # (0,1) random
                    temp_array = np.nonzero(TD_sum >= rand_batch)[0]
                    if len(temp_array) > 0:
                        TD_index = temp_array[0]
                    else:
                        print ('WHAT THE FUCK IS GOING ON???????????? \n\n\n\n')
                        time.sleep(1000)
                        TD_index = 0

                    # expected: weight_is.len === replaymemory.len === TD_sum.len
                    if self.step > 20000:
                        print ('len(weight_is): ', str(len(weight_is)))
                        print ('len(replaymemory): ', str(len(self.replay_memory)))
                        print ('len(TD_sum): ', str(len(TD_sum)))
                        print ('TD_index: ', str(TD_index))
                    if TD_index > len(self.replay_memory):
                        print ('len(weight_is): ', str(len(weight_is)))
                        print ('len(replaymemory): ', str(len(self.replay_memory)))
                        print ('len(TD_sum): ', str(len(TD_sum)))
                        print ('TD_index: ', str(TD_index))
                        print('BUG: deuqe index out of range.')

                    batch_index.append(TD_index)
                    w_batch.append(weight_is[TD_index])
                    minibatch.append(self.replay_memory[TD_index])
                return minibatch, w_batch, batch_index
    
    def get_progress(self):
        progress = ''
		# if current_step <= num_exploration
        if self.step <= self.Num_Exploration:
            progress = 'Exploring'
        elif self.step > self.Num_Exploration:
            progress = 'Training'
        elif self.step <= (self.Num_Exploration + self.Num_Training + self.Num_Testing):
            progress = 'Testing'
        else:
            progress = 'Finished'

        return progress

    def replay(self,minibatch, w_batch, batch_index):
        # DucAnh implementation (no PER)
        # minibatch = random.sample(self.replay_memory, self.minibatch_size)
        
        J = 0
        TD_error_batch = []

        for s, a, r, next_s, done in minibatch:
            if not done:
                Q_value_comma = self.model.predict(next_s)[0]               # Q-value(s') -- PrimaryModel
                a_comma = np.argmax(Q_value_comma)                          # pick a' cause maxQ-value(s')
                Q_target = r + self.gamma * self.targetDQN.model.predict(next_s)[0][a_comma]    # a number
                target_f = self.model.predict(s)    # Q value Q(s,a,theta)
                Q_value = target_f[0][a]
                
                # PER: append TD_Error:
                TD_error_batch.append(Q_target-Q_value)

                # calculate loss for log
                TD_error = (Q_target - Q_value)*(Q_target - Q_value)
                J += TD_error

                target_f[0][a] = Q_target
                self.model.fit(s, target_f, epochs=1, verbose=0,batch_size=self.minibatch_size)

        J = J/self.minibatch_size

        self.loss_plot.append(J)
        self.step_plot.append(self.step)
        np.save('array_plot/array_loss.npy', self.loss_plot)
        np.save('array_plot/array_step.npy', self.step_plot)

        # TODO: is it updating Target_NEtwork?
        self.targetDQN.replay(self.model.get_weights())

        # TODO: TD_error_batch?  Ok

        # TODO: Update TD_list.
        for i_batch in range(len(batch_index)):
            self.TD_list[batch_index[i_batch]] = pow((abs(TD_error_batch[i_batch]) + self.eps), self.alpha_per)

        # Update beta.
        self.beta = self.beta + (1 - self.beta_init) / self.Num_Training

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

    def remember(self, state, action, reward, next_state, done):
        self.replay_memory.append((state, action, reward, next_state, done))

    def replay_random_sample(self):
        # DucAnh implementation (no PER)
        minibatch = random.sample(self.replay_memory, self.minibatch_size)

        J = 0
        TD_error_batch = []

        for s, a, r, next_s, done in minibatch:
            if not done:
                Q_value_comma = self.model.predict(next_s)[0]  # Q-value(s') -- PrimaryModel
                a_comma = np.argmax(Q_value_comma)  # pick a' cause maxQ-value(s')
                Q_target = r + self.gamma * self.targetDQN.model.predict(next_s)[0][a_comma]  # a number
                target_f = self.model.predict(s)  # Q value Q(s,a,theta)
                Q_value = target_f[0][a]

                # PER: append TD_Error:
                TD_error_batch.append(Q_target - Q_value)

                # calculate loss for log
                TD_error = (Q_target - Q_value) * (Q_target - Q_value)
                J += TD_error

                target_f[0][a] = Q_target
                self.model.fit(s, target_f, epochs=1, verbose=0, batch_size=self.minibatch_size)

        J = J / self.minibatch_size

        self.loss_plot.append(J)
        self.step_plot.append(self.step)
        np.save('array_plot/array_loss_random_sample.npy', self.loss_plot)
        np.save('array_plot/array_step_random_sample.npy', self.step_plot)

    def store_priority(self, state, action, reward, next_state):
        self.replay_memory.append((state, action, reward, next_state))

    def replay_priority(self):
        priority = []
        minibatch = []
        for i in range(len(self.replay_memory)):
            s, a, r, next_s = self.replay_memory[i]
            Q_value = self.model.predict(s)[0][a]
            Q_target = self.targetDQN.model.predict(s)[a]
            delta = abs(Q_value - Q_target)
            priority.append(pow(delta,self.alpha_per))

        sum = sum(priority)
        priority =  [i / sum for i in priority]

        for i in range(self.minibatch_size):
            max_index = np.argmax(priority)
            minibatch.append(self.replay_memory[max_index])
            del priority[max_index]

        # start replay with minibatch priority
        for s, a, r, next_s in minibatch:
            Q_value_comma = self.model.predict(next_s)[0]  # Q-value(s') -- PrimaryModel
            a_comma = np.argmax(Q_value_comma)  # pick a' cause maxQ-value(s')
            Q_target = r + self.gamma * self.targetDQN.model.predict(next_s)[0][a_comma]  # a number
            target_f = self.model.predict(s)  # Q value Q(s,a,theta)
            Q_value = target_f[0][a]

            # PER: append TD_Error:
            TD_error_batch.append(Q_target - Q_value)

            # calculate loss for log
            TD_error = (Q_target - Q_value) * (Q_target - Q_value)
            J += TD_error

            target_f[0][a] = Q_target
            self.model.fit(s, target_f, epochs=1, verbose=0, batch_size=1)

        J = J / self.minibatch_size

        self.loss_plot.append(J)
        self.step_plot.append(self.step)
        np.save('array_plot/array_loss.npy', self.loss_plot)
        np.save('array_plot/array_step.npy', self.step_plot)


