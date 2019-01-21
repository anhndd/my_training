import os
import sys
import random
import numpy as np
import time
import math
from collections import deque
from keras.layers import Input, Conv2D, Flatten, Dense, LeakyReLU, Average, Add, Dot, Subtract
from keras.models import Model
from keras.optimizers import Adam
import tensorflow as tf

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

sumoBinary = "/usr/bin/sumo-gui"
sumoConfig = "sumoconfig.sumoconfig"
import traci

class TargetDQNAgent:
    def __init__(self, action_size):
        self.tp = 2000          # pre-training steps
        self.alpha = 0.0001     # target network update rate
        self.gamma = 0.99       # discount factor
        self.epsilon_r = 0.0001 # learning rate
        self.Beta = 0.01        # Leaky ReLU
        self.action_size = action_size
        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        input_1 = Input(shape=(60, 60, 2))
        input_2 = Input(shape=(self.action_size, self.action_size))
        # keras.layers.Conv2D(filters, kernel_size, strides=(1, 1), padding='valid', data_format=None,
        #                     dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform',
        #                     bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None,
        #                     activity_regularizer=None, kernel_constraint=None, bias_constraint=None)
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

        model = Model(inputs=[input_1,input_2], outputs=[Q_value])
        # model.compile(optimizer= Adam(lr=self.epsilon_r), loss='mse')

        return model

    # def remember(self, state, action, reward, next_state, done):
    #     self.memory.append((state, action, reward, next_state, done))
    #
    # def act(self, state):
    #     if self.end_epsilon <= self.start_epsilon:
    #         return random.randrange(self.action_size)
    #     act_values = self.model.predict(state)
    #
    #     return np.argmax(act_values)  # returns action
    #
    def replay(self):
        minibatch = random.sample(self.memory, self.minibatch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
    #
    # def load(self, name):
    #     self.model.load_weights(name)
    #
    # def save(self, name):
    #     self.model.save_weights(name)

class DQNAgent:
    def __init__(self, M, action_size):
        self.memory = deque(maxlen=M)
        self.minibatch_size = 64
        self.start_epsilon = 1
        self.end_epsilon = 0.01
        self.step_epsilon = 10000
        self.tp = 2000          # pre-training steps
        self.alpha = 0.0001     # target network update rate
        self.gamma = 0.99       # discount factor
        self.epsilon_r = 0.0001 # learning rate
        self.Beta = 0.01        # Leaky ReLU
        self.action_size = action_size
        self.model = self._build_model()
        self.targetDQN = TargetDQNAgent(self.action_size)

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        input_1 = Input(shape=(60, 60, 2))
        input_2 = Input(shape=(self.action_size, self.action_size))
        # keras.layers.Conv2D(filters, kernel_size, strides=(1, 1), padding='valid', data_format=None,
        #                     dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform',
        #                     bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None,
        #                     activity_regularizer=None, kernel_constraint=None, bias_constraint=None)
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

        model = Model(inputs=[input_1,input_2], outputs=[Q_value])
        model.compile(optimizer= Adam(lr=self.epsilon_r), loss='mse')

        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if self.end_epsilon <= self.start_epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)

        return np.argmax(act_values)  # returns action

    def replay(self):
        minibatch = random.sample(self.memory, self.minibatch_size)
        ######################################################
        Q_value = []
        Q_target = []
        for s, a, r, next_s, done in minibatch:
            if not done:
                Q_value_list = self.model.predict(s)[0]
                Q_value.append(Q_value_list[a])

                next_a = np.argmax(Q_value_list)
                Q_target.append(r + self.gamma * self.targetDQN.model.predict(next_s)[0][next_a])
                # print Q_value, Q_target
                # Q_value = self.model.predict(s)
                # # next_a = np.argmax(Q_value_list)
                # Q_target = r + self.gamma * self.targetDQN.model.predict(next_s)[0]

                # J_temp = Q_target - Q_value
                # J += J_temp * J_temp
                ##############################################
        # J /= self.minibatch_size #.
        #
        #     target = (r + self.gamma *
        #               np.amax(self.model.predict(next_s)[0]))
        # target_f = self.model.predict(s)
        # target_f[0][a] = target
        #         self.model.fit(Q_target, Q_value, epochs=1, verbose=0)
        input= np.array([Q_target, Q_value]).reshape(1,1,1,1)
        self.model.fit(input, epochs=1, verbose=2)

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

class SumoIntersection:
    def __init__(self):
        # we need to import python modules from the $SUMO_HOME/tools directory
        try:
            sys.path.append(os.path.join(os.path.dirname(
                __file__), '..', '..', '..', '..', "tools"))  # tutorial in tests
            sys.path.append(os.path.join(os.environ.get("SUMO_HOME", os.path.join(
                os.path.dirname(__file__), "..", "..", "..")), "tools"))  # tutorial in docs
        except ImportError:
            sys.exit(
                "please declare environment variable 'SUMO_HOME' as the root directory of your sumo installation (it should contain folders 'bin', 'tools' and 'docs')")

    def getState(self, I):
        positionMatrix = []
        velocityMatrix = []

        cellLength = 2.7
        sizeMatrix = 60
        sizeLaneMatric = sizeMatrix/2-3           #27
        offset_in = 500-cellLength*sizeLaneMatric #427.1
        offset_out = cellLength*sizeLaneMatric    #72.9
        speedLimit = 14
	    #print(traci.edge.getWaitingTime('gneE21'))
	
        # junctionPosition = traci.junction.getPosition('0')[0]
        vehicles_road1_in = traci.edge.getLastStepVehicleIDs('L53')
        vehicles_road1_out = traci.edge.getLastStepVehicleIDs('L35')
        vehicles_road2_in = traci.edge.getLastStepVehicleIDs('D3')
        vehicles_road2_out = traci.edge.getLastStepVehicleIDs('D4')
        vehicles_road3_in = traci.edge.getLastStepVehicleIDs('L43')
        vehicles_road3_out = traci.edge.getLastStepVehicleIDs('L34')
        vehicles_road4_in = traci.edge.getLastStepVehicleIDs('L23')
        vehicles_road4_out = traci.edge.getLastStepVehicleIDs('L32')

        for i in range(sizeMatrix):
            positionMatrix.append([])
            velocityMatrix.append([])
            for j in range(sizeMatrix):
                positionMatrix[i].append(0)
                velocityMatrix[i].append(0)

        index = 32
        for v in vehicles_road1_in:
            std = traci.vehicle.getLanePosition(v)-offset_in
            startPos = int(math.ceil((std - traci.vehicle.getLength(v)) / cellLength)-1)
            endPos = int(math.ceil(std / cellLength)-1)
            if endPos > 26:
                endPos = 26
            if startPos > 26:
                startPos = 26
            elif startPos < 0:
                startPos = 0
            if endPos >= 0:
                positionMatrix[index-traci.vehicle.getLaneIndex(v)][startPos] = 1
                positionMatrix[index-traci.vehicle.getLaneIndex(v)][endPos] = 1
                velocityMatrix[index - traci.vehicle.getLaneIndex(v)][startPos] = traci.vehicle.getSpeed(v)
                velocityMatrix[index - traci.vehicle.getLaneIndex(v)][endPos] = traci.vehicle.getSpeed(v)
                for i in range(startPos,endPos):
                    positionMatrix[index - traci.vehicle.getLaneIndex(v)][i] = 1
                    velocityMatrix[index - traci.vehicle.getLaneIndex(v)][i] = traci.vehicle.getSpeed(v)
        for v in vehicles_road2_in:
            std = traci.vehicle.getLanePosition(v)-offset_in
            startPos = int(math.ceil((std - traci.vehicle.getLength(v)) / cellLength)-1)
            endPos = int(math.ceil(std / cellLength)-1)
            if endPos > 26:
                endPos = 26
            if startPos > 26:
                startPos = 26
            elif startPos < 0:
                startPos = 0
            if endPos >= 0:
                # print(traci.vehicle.getVehicleClass(v), traci.vehicle.getLanePosition(v),std,startPos,endPos)
                positionMatrix[sizeMatrix-1-startPos][index-traci.vehicle.getLaneIndex(v)] = 1
                positionMatrix[sizeMatrix-1-endPos][index-traci.vehicle.getLaneIndex(v)] = 1
                velocityMatrix[sizeMatrix-1-startPos][index-traci.vehicle.getLaneIndex(v)] = traci.vehicle.getSpeed(v)
                velocityMatrix[sizeMatrix-1-endPos][index-traci.vehicle.getLaneIndex(v)] = traci.vehicle.getSpeed(v)
                for i in range(startPos,endPos):
                    positionMatrix[sizeMatrix-1-i][index - traci.vehicle.getLaneIndex(v)] = 1
                    velocityMatrix[sizeMatrix - 1 - i][index - traci.vehicle.getLaneIndex(v)] = traci.vehicle.getSpeed(v)
        for v in vehicles_road3_out:
            std = offset_out - traci.vehicle.getLanePosition(v)
            startPos = int(math.ceil((std - traci.vehicle.getLength(v)) / cellLength)-1)
            endPos = int(math.ceil(std / cellLength)-1)
            if endPos > 26:
                endPos = 26
            if startPos > 26:
                startPos = 26
            elif startPos < 0:
                startPos = 0
            if endPos >= 0:
                # print(traci.vehicle.getVehicleClass(v), traci.vehicle.getLanePosition(v),std,startPos,endPos)
                positionMatrix[index - traci.vehicle.getLaneIndex(v)][sizeMatrix-1-startPos] = 1
                positionMatrix[index - traci.vehicle.getLaneIndex(v)][sizeMatrix-1-endPos] = 1
                velocityMatrix[index - traci.vehicle.getLaneIndex(v)][sizeMatrix-1-startPos] = traci.vehicle.getSpeed(v)
                velocityMatrix[index - traci.vehicle.getLaneIndex(v)][sizeMatrix-1-endPos] = traci.vehicle.getSpeed(v)
                for i in range(startPos,endPos):
                    positionMatrix[index - traci.vehicle.getLaneIndex(v)][sizeMatrix-1-i] = 1
                    velocityMatrix[index - traci.vehicle.getLaneIndex(v)][sizeMatrix - 1 - i] = traci.vehicle.getSpeed(v)
        for v in vehicles_road4_out:
            std = offset_out - traci.vehicle.getLanePosition(v)
            startPos = int(math.ceil((std - traci.vehicle.getLength(v)) / cellLength)-1)
            endPos = int(math.ceil(std / cellLength)-1)
            if endPos > 26:
                endPos = 26
            if startPos > 26:
                startPos = 26
            elif startPos < 0:
                startPos = 0
            if endPos >= 0:
                # print(traci.vehicle.getVehicleClass(v), traci.vehicle.getLanePosition(v),std,startPos,endPos)
                positionMatrix[startPos][index - traci.vehicle.getLaneIndex(v)] = 1
                positionMatrix[endPos][index - traci.vehicle.getLaneIndex(v)] = 1
                velocityMatrix[startPos][index - traci.vehicle.getLaneIndex(v)] = traci.vehicle.getSpeed(v)
                velocityMatrix[endPos][index - traci.vehicle.getLaneIndex(v)] = traci.vehicle.getSpeed(v)
                for i in range(startPos,endPos):
                    positionMatrix[i][index - traci.vehicle.getLaneIndex(v)] = 1
                    velocityMatrix[i][index - traci.vehicle.getLaneIndex(v)] = traci.vehicle.getSpeed(v)

        index = 27
        for v in vehicles_road1_out:
            std = offset_out - traci.vehicle.getLanePosition(v)
            startPos = int(math.ceil((std - traci.vehicle.getLength(v)) / cellLength) - 1)
            endPos = int(math.ceil(std / cellLength) - 1)
            if endPos > 26:
                endPos = 26
            if startPos > 26:
                startPos = 26
            elif startPos < 0:
                startPos = 0
            if endPos >= 0:
                positionMatrix[index + traci.vehicle.getLaneIndex(v)][startPos] = 1
                positionMatrix[index + traci.vehicle.getLaneIndex(v)][endPos] = 1
                velocityMatrix[index + traci.vehicle.getLaneIndex(v)][startPos] = traci.vehicle.getSpeed(v)
                velocityMatrix[index + traci.vehicle.getLaneIndex(v)][endPos] = traci.vehicle.getSpeed(v)
                for i in range(startPos, endPos):
                    positionMatrix[index + traci.vehicle.getLaneIndex(v)][i] = 1
                    velocityMatrix[index + traci.vehicle.getLaneIndex(v)][i] = traci.vehicle.getSpeed(v)
        for v in vehicles_road2_out:
            std = offset_out - traci.vehicle.getLanePosition(v)
            startPos = int(math.ceil((std - traci.vehicle.getLength(v)) / cellLength) - 1)
            endPos = int(math.ceil(std / cellLength) - 1)
            if endPos > 26:
                endPos = 26
            if startPos > 26:
                startPos = 26
            elif startPos < 0:
                startPos = 0
            if endPos >= 0:
                positionMatrix[sizeMatrix-1-startPos][index + traci.vehicle.getLaneIndex(v)] = 1
                positionMatrix[sizeMatrix-1-endPos][index + traci.vehicle.getLaneIndex(v)] = 1
                velocityMatrix[sizeMatrix - 1 - startPos][index + traci.vehicle.getLaneIndex(v)] = traci.vehicle.getSpeed(v)
                velocityMatrix[sizeMatrix - 1 - endPos][index + traci.vehicle.getLaneIndex(v)] = traci.vehicle.getSpeed(v)
                for i in range(startPos, endPos):
                    positionMatrix[sizeMatrix-1-i][index + traci.vehicle.getLaneIndex(v)] = 1
                    velocityMatrix[sizeMatrix - 1 - i][index + traci.vehicle.getLaneIndex(v)] = traci.vehicle.getSpeed(v)
        for v in vehicles_road3_in:
            std = traci.vehicle.getLanePosition(v)-offset_in
            startPos = int(math.ceil((std - traci.vehicle.getLength(v)) / cellLength)-1)
            endPos = int(math.ceil(std / cellLength)-1)
            if endPos > 26:
                endPos = 26
            if startPos > 26:
                startPos = 26
            elif startPos < 0:
                startPos = 0
            if endPos >= 0:
                # print(traci.vehicle.getVehicleClass(v), traci.vehicle.getLanePosition(v),std,startPos,endPos)
                positionMatrix[index + traci.vehicle.getLaneIndex(v)][sizeMatrix-1-startPos] = 1
                positionMatrix[index + traci.vehicle.getLaneIndex(v)][sizeMatrix-1-endPos] = 1
                velocityMatrix[index + traci.vehicle.getLaneIndex(v)][sizeMatrix-1-startPos] = traci.vehicle.getSpeed(v)
                velocityMatrix[index + traci.vehicle.getLaneIndex(v)][sizeMatrix-1-endPos] = traci.vehicle.getSpeed(v)
                for i in range(startPos,endPos):
                    positionMatrix[index + traci.vehicle.getLaneIndex(v)][sizeMatrix-1-i] = 1
                    velocityMatrix[index + traci.vehicle.getLaneIndex(v)][sizeMatrix - 1 - i] = traci.vehicle.getSpeed(v)
        for v in vehicles_road4_in:
            std = traci.vehicle.getLanePosition(v)-offset_in
            startPos = int(math.ceil((std - traci.vehicle.getLength(v)) / cellLength)-1)
            endPos = int(math.ceil(std / cellLength)-1)
            if endPos > 26:
                endPos = 26
            if startPos > 26:
                startPos = 26
            elif startPos < 0:
                startPos = 0
            if endPos >= 0:
                # print(traci.vehicle.getVehicleClass(v), traci.vehicle.getLanePosition(v),std,startPos,endPos)
                positionMatrix[startPos][index + traci.vehicle.getLaneIndex(v)] = 1
                positionMatrix[endPos][index + traci.vehicle.getLaneIndex(v)] = 1
                velocityMatrix[startPos][index + traci.vehicle.getLaneIndex(v)] = traci.vehicle.getSpeed(v)
                velocityMatrix[endPos][index + traci.vehicle.getLaneIndex(v)] = traci.vehicle.getSpeed(v)
                for i in range(startPos,endPos):
                    positionMatrix[i][index + traci.vehicle.getLaneIndex(v)] = 1
                    velocityMatrix[i][index + traci.vehicle.getLaneIndex(v)] = traci.vehicle.getSpeed(v)

        # for v in positionMatrix:
        #     print(v)
        # for i in range(0,50):
        #     s = ''
        #     for j in range(0, 60):
        #         s+=str(positionMatrix[i][j])
        #     print s

        # for v in velocityMatrix:
        #     print(v)

        outputMatrix = [positionMatrix, velocityMatrix]
        output = np.transpose(outputMatrix)#np.array(outputMatrix)
        output = output.reshape(1,60,60,2)
        # print output.tolist()

        return [output, I]

    def cal_yellow_phase(self, id_list, a_dec):
        v_on_road = []
        for id in id_list:
            vehicles_road = traci.edge.getLastStepVehicleIDs(id)
            for vehicle in vehicles_road:
                v_on_road.append(traci.vehicle.getSpeed(vehicle))
        if not v_on_road:
            v_on_road.append(0)

        return int(np.amax(v_on_road)/a_dec)

    def cal_waiting_time(self):
        waiting_time = 0
        waiting_time += (traci.edge.getLastStepHaltingNumber('L53')
                         + traci.edge.getLastStepHaltingNumber('L34')
                         + traci.edge.getLastStepHaltingNumber('D3')
                         + traci.edge.getLastStepHaltingNumber('L23'))
        return waiting_time

def main():
    # Control code here
    M = 20000 #size memory
    a_dec = 4.5 # m/s^2
    phase_number = 2
    action_space = phase_number * 2 + 1
    action_policy = [[0, 0], [5, 0], [-5, 0], [0, 5], [0, -5]]
    I = np.full((action_space, action_space), 0.5).reshape(1, action_space, action_space)
    action_time = [33, 33]
    waiting_time_t = 0
    waiting_time_t1 = 0
    total_reward = 0
    reward_t = 0
    step = 0
    i = 0
    agent = DQNAgent(M, action_space)

    sumo_int = SumoIntersection()
    sumo_cmd = [sumoBinary, "-c", sumoConfig]
    traci.start(sumo_cmd)

    while traci.simulation.getMinExpectedNumber() > 0 and step < 100:#7000
        traci.simulationStep()
        waiting_time = 0
        state = sumo_int.getState(I)
        action = agent.act(state)
        for j in range(phase_number):
            action_time[j] += action_policy[action][j]
            if action_time[j] < 0:
                action_time[j] = 0
            elif action_time[j] > 60:
                action_time[j] = 60

        # print action_time[0]
        for j in range(action_time[0]):
            traci.trafficlight.setPhase('3', 0)
            waiting_time += sumo_int.cal_waiting_time()
            traci.simulationStep()

        yellow_time1 =  sumo_int.cal_yellow_phase(['L53','L34'], a_dec)
        # print waiting_time#yellow_time1
        for j in range(yellow_time1):
            traci.trafficlight.setPhase('3', 1)
            waiting_time += sumo_int.cal_waiting_time()
            traci.simulationStep()

        # print waiting_time#action_time[1]
        for j in range(action_time[1]):
            traci.trafficlight.setPhase('3', 2)
            waiting_time += sumo_int.cal_waiting_time()
            traci.simulationStep()

        yellow_time2 =  sumo_int.cal_yellow_phase(['D3','L23'], a_dec)
        # print waiting_time#yellow_time2
        for j in range(yellow_time2):
            traci.trafficlight.setPhase('3', 3)
            waiting_time += sumo_int.cal_waiting_time()
            traci.simulationStep()

        waiting_time_t1 = waiting_time
        reward_t = waiting_time_t - waiting_time_t1
        # print waiting_time_t, waiting_time_t1, reward_t
        waiting_time_t = waiting_time_t1

        new_state = sumo_int.getState(I)
        agent.remember(state, action, reward_t, new_state, False)

        i += 1;
        # if len(agent.memory) > M & i > agent.tp:
        # if len(agent.memory) > 100 & i > 1:
        #     print 'begin'
        #     print agent.replay()
        print('-----------------------end simulation----------------------------')
        step += 1
    traci.close()

if __name__ == '__main__':
    main()
