
from keras.layers import Input, Conv2D, Flatten, Dense, LeakyReLU, Average, Add, Dot, Subtract, Multiply
from keras.models import Model
from keras.optimizers import Adam
from collections import deque
import TargetDQNAgent
import numpy as np
import random
import tensorflow as tf
import os


class DQNAgent:
    def __init__(self, M, action_size, B):
        
        # Duc Anh Implement:
        # self.memory = deque(maxlen=M)
        
        # Duy Do:
        # TODO: config again.
        self.Num_Exploration = 2000	                                            
        self.Num_Training    = 50000
        self.Num_Testing     = 250000
        self.progress = ''
        self.step = 1

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
        self.alpha = 0.6
        self.beta_init = 0.4
        self.beta = self.beta_init
        self.TD_list = np.array([])
        self.Num_batch = 4
        self.Num_replay_memory = 50000
        self.replay_memory = deque(maxlen=M)


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
        #
        # input_3 = Input(shape=(5,))
        # Output = Multiply()([input_3, Q_value])

        model = Model(inputs=[input_1, input_2], outputs=[Q_value])
        model.compile(optimizer= Adam(lr=self.epsilon_r), loss='mse')

        return model

    # def remember(self, state, action, reward, next_state, done):
    #     # Duc Anh implementation:
    #     self.replay_memory.append((state, action, reward, next_state, done))

    def store_tuple(self, state, action, reward, next_state, terminal):
        # Num_replay_memory = 50.000
        # If Replay memory is longer than Num_replay_memory, delete the oldest one
		if len(self.replay_memory) >= self.Num_replay_memory:
			del self.replay_memory[0]
			self.TD_list = np.delete(self.TD_list, 0)

		if self.progress == 'Exploring':
			self.replay_memory.append([state, action, reward, next_state, terminal])
			self.TD_list = np.append(self.TD_list, pow((abs(reward) + self.eps), self.alpha))
			'''
				alpha: 0.6
				eps: 0.00001
			'''

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
			self.TD_list = np.append(self.TD_list, pow((abs(Q_target-Q_value) + self.eps), self.alpha))
			# ###################################################################################################################

    # select action random || by model.
    def select_action(self,state, tentative_act_dec):
        # print tentative_act_dec, tentative_act_dec[0], 'max = ',np.argmax(tentative_act_dec[0])
        if (tentative_act_dec[0][1] == 0):
            return np.argmax(tentative_act_dec[0])
        if np.random.rand() <= self.start_epsilon:
            return random.randrange(self.action_size)
        else:
            act_values = self.model.predict(state)
            return np.argmax(act_values[0])  # returns action
    def prioritized_minibatch(self):
				# Update TD_error list
                TD_normalized = self.TD_list / np.linalg.norm(self.TD_list, 1)
                TD_sum = np.cumsum(TD_normalized)
				# Get importance sampling weights
                weight_is = np.power((self.Num_replay_memory * TD_normalized), - self.beta)
                weight_is = weight_is / np.max(weight_is)
                # Select mini batch and importance sampling weights
                minibatch = []
                batch_index = []
                w_batch = []
                for i in range(self.Num_batch):
                    rand_batch = random.random()
                    TD_index = np.nonzero(TD_sum >= rand_batch)[0][0]
                    batch_index.append(TD_index)
                    w_batch.append(weight_is[TD_index])
                    minibatch.append(self.replay_memory[TD_index])
                return minibatch, w_batch, batch_index
    

    def storeTraining(self, state, action, reward, next_state, done):
        Q_value_comma = self.model.predict(next_state)[0]               # Q-value(s') -- PrimaryModel
        a_comma = np.argmax(Q_value_comma)                              # pick a' cause maxQ-value(s')
        Q_target = reward + self.gamma * self.targetDQN.model.predict(next_state)[0][a_comma]    # a number
        target_f = self.model.predict(state)    #  Q value Q(s,a,theta)
        Q_value = target_f[0][action]

        self.replay_memory.append([state, action, reward, next_state, done])
        self.TD_list = np.append(self.TD_list, pow((abs(Q_target-Q_value) + self.eps), self.alpha))

    def get_progress(self):
		progress = ''
		# print 'Num_Exploration: ', self.Num_Exploration		# 1000
		# print 'Num_Training: ', self.Num_Training				# 500.000
		# if current_step <= num_exploration
		if self.step <= self.Num_Exploration:
			progress = 'Exploring'
		elif self.step <= self.Num_Exploration + self.Num_Training:
			progress = 'Training'
		elif self.step <= self.Num_Exploration + self.Num_Training + self.Num_Testing:
			progress = 'Testing'
		else:
			progress = 'Finished'

		return progress

    def train(self, minibatch, w_batch, batch_index):

		# Save the each batch batch data
		state_batch      = [batch[0] for batch in minibatch]
		action_batch     = [batch[1] for batch in minibatch]
		reward_batch     = [batch[2] for batch in minibatch]
		next_state_batch = [batch[3] for batch in minibatch]
		terminal_batch   = [batch[4] for batch in minibatch]

		# Get y_prediction (list of Q-target in minibatch)
		y_batch = []
		# 
		Q_batch = self.output_target.eval(feed_dict = {self.input_target: next_state_batch})	# len = 32

		# Get Q-target values
		for i in range(len(minibatch)):
			if terminal_batch[i] == True:
				y_batch.append(reward_batch[i])
			else:
				y_batch.append(reward_batch[i] + self.gamma * np.max(Q_batch[i]))	# Q_target = r + maxQ(s')

		# calculate TD_error of mini batch
		_, self.loss, TD_error_batch = self.sess.run([self.train_step, self.loss_train, self.TD_error], feed_dict = {self.action_target: action_batch,
										 										      									self.y_target: y_batch,
										 									  	      									self.input: state_batch,
																														self.w_is: w_batch})
		# print 'TD_error_batch: ', TD_error_batch #
		# print 'TD_error_batch len', len(TD_error_batch) # 32
		
		# Update TD_list
		for i_batch in range(len(batch_index)):
			self.TD_list[batch_index[i_batch]] = pow((abs(TD_error_batch[i_batch]) + self.eps), self.alpha)

		# Update Beta
		self.beta = self.beta + (1 - self.beta_init) / self.Num_Training


    def replay(self,minibatch, w_batch, batch_index):
        
        # DucAnh implementation (no PER)
        # minibatch = random.sample(self.replay_memory, self.minibatch_size)
        
        J = 0
        for s, a, r, next_s, done in minibatch:
            if not done:
                Q_value_comma = self.model.predict(next_s)[0]               # Q-value(s') -- PrimaryModel
                a_comma = np.argmax(Q_value_comma)                          # pick a' cause maxQ-value(s')
                Q_target = r + self.gamma * self.targetDQN.model.predict(next_s)[0][a_comma]    # a number
                target_f = self.model.predict(s)    # Q value Q(s,a,theta)
                Q_value = target_f[0][a]
                TD_error = Q_target - Q_value
                target_f[0][a] = Q_target
                self.model.fit(s, target_f, epochs=1, verbose=0,batch_size=self.minibatch_size)
        
        # TODO: is it updating Target_NEtwork?
        self.targetDQN.replay(self.model.get_weights())

        # TODO: TD-error?
        # TODO: Update TD_list.
        # TODO: Update beta.

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)