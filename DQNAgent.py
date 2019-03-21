
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
        # self.memory = Memory.Memory(M)
        self.PER_e = 0.01  # Hyperparameter that we use to avoid self.memorysome experiences to have 0 probability of being taken
        self.PER_a = 0.6  # Hyperparameter that we use to make a tradeoff between taking only exp with high priority and sampling randomly
        self.PER_b = 0.4  # importance-sampling, from initial value increasing to 1
        self.PER_b_increment_per_sampling = 0.001

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
    
    def experience_replay(self, state, action, reward, next_state, terminal):

                self.replay_memory.append([state, action, reward, next_state, terminal])
                self.TD_list = np.append(self.TD_list, pow((abs(reward) + self.eps), self.alpha))
                '''
                    alpha: 0.6
                    eps: 0.00001
                '''

    def remember(self, state, action, reward, next_state, done):
        
        # Duc Anh implementation:
        self.replay_memory.append((state, action, reward, next_state, done))

        # Duy Do ver1: change MEMORY Implementation:
        # experience = state, action, reward, next_state, done
        # self.memory.store(experience)


        # Duy Do ver 2 (PER)
        # self.experience_replay(state, action, reward, next_state, done)
        

        # add to TD_list.
        # self.TD_list = np.append(self.TD_list, pow((abs(reward) + self.eps), self.alpha))

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



    def replay(self):
        # minibatch, w_batch, batch_index  = self.prioritized_minibatch()
        minibatch = random.sample(self.replay_memory, self.minibatch_size)
        J = 0
        # absolute_errors = []
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
        
        self.targetDQN.replay(self.model.get_weights())

        # TODO: TD-error?
        # TODO: Update TD_list.
        # TODO: Update beta.

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)
    
    def init_sess(self):
        # Initialize variables
		config = tf.ConfigProto()
		# config.gpu_options.per_process_gpu_memory_fraction = self.GPU_fraction

		sess = tf.InteractiveSession(config=config)

		# Make folder for save data
		# os.makedirs('saved_networks/' + self.game_name + '/' + self.date_time + '_' + self.algorithm)

		# Summary for tensorboard
		summary_placeholders, update_ops, summary_op = self.setup_summary()
		# summary_writer = tf.summary.FileWriter('saved_networks/' + self.game_name + '/' + self.date_time + '_' + self.algorithm, sess.graph)

		init = tf.global_variables_initializer()
		sess.run(init)

		# Load the file if the saved file exists
		saver = tf.train.Saver()
		# check_save = 1
		check_save = input('Load Model? (1=yes/2=no): ')

		if check_save == 1:
			# Restore variables from disk.
			saver.restore(sess, self.load_path + "/model.ckpt")
			print("Model restored.")

			check_train = input('Inference or Training? (1=Inference / 2=Training): ')
			if check_train == 1:
				self.Num_Exploration = 0
				self.Num_Training = 0

		return sess, saver, summary_placeholders, update_ops, summary_op
    # Code for tensorboard
    def setup_summary(self):
	    episode_score = tf.Variable(0.)
	    episode_maxQ = tf.Variable(0.)
	    episode_loss = tf.Variable(0.)

	    tf.summary.scalar('Average Score/' + str(self.Num_plot_episode) + ' episodes', episode_score)
	    tf.summary.scalar('Average MaxQ/' + str(self.Num_plot_episode) + ' episodes', episode_maxQ)
	    tf.summary.scalar('Average Loss/' + str(self.Num_plot_episode) + ' episodes', episode_loss)

	    summary_vars = [episode_score, episode_maxQ, episode_loss]

	    summary_placeholders = [tf.placeholder(tf.float32) for _ in range(len(summary_vars))]
	    update_ops = [summary_vars[i].assign(summary_placeholders[i]) for i in range(len(summary_vars))]
	    summary_op = tf.summary.merge_all()
	    return summary_placeholders, update_ops, summary_op
