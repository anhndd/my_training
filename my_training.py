import os
import sys
import random
import numpy as np
import time
import math
import h5py
from collections import deque
from keras.layers import Input, Conv2D, Flatten, Dense, LeakyReLU, Average, Add, Dot, Subtract, Multiply
from keras.models import Model
from keras.optimizers import Adam
import tensorflow as tf
from scipy.spatial.ckdtree import coo_entries

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

sumoBinary = "/usr/bin/sumo"
sumoConfig = "sumoconfig.sumoconfig"
import traci

# count_action_dif_default = 0
class TargetDQNAgent:
    def __init__(self, action_size):
        self.alpha = 0.0001     # target network update rate
        self.Beta = 0.01        # Leaky ReLU
        self.action_size = action_size
        self.model = self._build_model()

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

        input_3 = Input(shape=(5,))
        Output = Multiply()([input_3, Q_value])

        model = Model(inputs=[input_1, input_2, input_3], outputs=[Output])
        # model.compile(optimizer= Adam(lr=self.epsilon_r), loss='mse')

        return model

    def replay(self, primary_network_weights):
        target_network_weights = self.model.get_weights()
        # print target_network_weights[len(target_network_weights)-1]
        # target_network_weights = self.alpha*target_network_weights + (1-self.alpha)*primary_network_weights

        for i in range(len(target_network_weights)):
            target_network_weights[i] = self.alpha*target_network_weights[i] + (1-self.alpha)*primary_network_weights[i]
        # print 'new weight'
        # print target_network_weights[len(target_network_weights)-1], primary_network_weights[len(target_network_weights)-1]
        self.model.set_weights(target_network_weights)

class DQNAgent:
    def __init__(self, M, action_size, B):
        self.memory = deque(maxlen=M)
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
        self.targetDQN = TargetDQNAgent(self.action_size)

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

        input_3 = Input(shape=(5,))
        Output = Multiply()([input_3, Q_value])

        model = Model(inputs=[input_1, input_2, input_3], outputs=[Output])
        model.compile(optimizer= Adam(lr=self.epsilon_r), loss='mse')

        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self,state):
        # print state[2][0]
        if (state[2][0][1] == 0):
            return np.argmax(state[2])
        if np.random.rand() <= self.start_epsilon:
            return random.randrange(self.action_size)
        else:
            act_values = self.model.predict(state)
            # print state[2][0], np.argmax(state[2][0])
            return np.argmax(act_values[0])  # returns action

    def replay(self):
        minibatch = random.sample(self.memory, self.minibatch_size)
        ######################################################
        J = 0
        # S_list = []
        # Q_list = []
        for s, a, r, next_s, done in minibatch:
            if not done:
                Q_value_comma = self.model.predict(next_s)[0]
                # Q_value = self.model.predict(s)[0][a]
                #
                next_a = np.argmax(Q_value_comma)
                Q_target = r + self.gamma * self.targetDQN.model.predict(next_s)[0][next_a]

                # J_temp = (Q_target - Q_value)
                # J_temp += J_temp * J_temp
                # J += J_temp
                ##############################################
                # cach 1...................
                target_f = self.model.predict(s)
                target_f[0][a] = Q_target
                self.model.fit(s, target_f, epochs=1, verbose=2,batch_size=self.minibatch_size)
                # cach 1...................
        # J /= self.minibatch_size
        # self.model.train_on_batch(X, Y)
        # self.model.fit(S_list, Q_list, epochs=1, verbose=0)
        self.targetDQN.replay(self.model.get_weights())

                # import keras
                # keras.callbacks.History()
        # input= np.array([Q_target, Q_value]).reshape(1,1,1,1)
        # self.model.fit(input, epochs=1, verbose=2)

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

    def getState(self, I, action, tentative_action):
        positionMatrix = []
        velocityMatrix = []

        cellLength = 2.7
        sizeMatrix = 60
        sizeLaneMatric = sizeMatrix/2-12           #18
        offset = 500-cellLength*sizeLaneMatric #427.1
        offset_out = cellLength*sizeLaneMatric    #72.9
        speedLimit = 14
	    #print(traci.edge.getWaitingTime('gneE21'))
	
        # junctionPosition = traci.junction.getPosition('0')[0]
        vehicles_road1_in = traci.edge.getLastStepVehicleIDs('gneE21')
        vehicles_road1_out = traci.edge.getLastStepVehicleIDs('gneE22')
        vehicles_road2_in = traci.edge.getLastStepVehicleIDs('gneE86')
        vehicles_road2_out = traci.edge.getLastStepVehicleIDs('gneE87')
        vehicles_road3_in = traci.edge.getLastStepVehicleIDs('gneE89')
        vehicles_road3_out = traci.edge.getLastStepVehicleIDs('gneE88')
        vehicles_road4_in = traci.edge.getLastStepVehicleIDs('gneE85')
        vehicles_road4_out = traci.edge.getLastStepVehicleIDs('gneE84')

        for i in range(sizeMatrix):
            positionMatrix.append([])
            velocityMatrix.append([])
            for j in range(sizeMatrix):
                positionMatrix[i].append(0)
                velocityMatrix[i].append(0)

        index = 42
        offset = 97.34 - cellLength*sizeLaneMatric
        for v in vehicles_road1_in:
            if traci.vehicle.getVehicleClass(v) != 'pedestrian':
                std = traci.vehicle.getLanePosition(v) - offset
                if std >= 0:
                    temp = std / cellLength
                    endPos = int(temp)
                    temp = (temp - endPos) / cellLength

                    temp_y_start = (1.75 - (
                            traci.vehicle.getLateralLanePosition(v) + traci.vehicle.getWidth(v) / 2)) / 0.875
                    yStartPos = int(temp_y_start)

                    temp_y_end = 1.75 - (traci.vehicle.getLateralLanePosition(v) - traci.vehicle.getWidth(v) / 2)
                    temp_y_end = temp_y_end / 0.875
                    yEndPos = int(temp_y_end)
                    if temp > 0.25:
                        endPos += 1
                    if temp_y_start < 0:
                        positionMatrix[index - traci.vehicle.getLaneIndex(v) * 4 + yStartPos][endPos] = 1
                    else:
                        temp_y_start2 = (temp_y_start - yStartPos) / 0.875
                        if temp_y_start2 < 0.75:
                            positionMatrix[index - traci.vehicle.getLaneIndex(v) * 4 + yStartPos][endPos] = 1
                    temp_y_end = (temp_y_end - yEndPos) / 0.875
                    if temp_y_end > 0.25:
                        positionMatrix[index - traci.vehicle.getLaneIndex(v) * 4 + yEndPos][endPos] = 1
                        for i in range(yStartPos, yEndPos + 1):
                            positionMatrix[index - traci.vehicle.getLaneIndex(v) * 4 + i][endPos] = 1
                    else:
                        for i in range(yStartPos, yEndPos):
                            positionMatrix[index - traci.vehicle.getLaneIndex(v) * 4 + i][endPos] = 1

                    temp = (std - traci.vehicle.getLength(v)) / cellLength
                    startPos = int(temp)
                    temp = (temp - startPos) / cellLength
                    if temp < 0.75:
                        if (startPos < 0) & (endPos >= 0):
                            startPos = 0
                        if temp_y_start < 0:
                            positionMatrix[index - traci.vehicle.getLaneIndex(v) * 4 + yStartPos][startPos] = 1
                        else:
                            temp_y_start2 = (temp_y_start - yStartPos) / 0.875
                            if temp_y_start2 < 0.75:
                                positionMatrix[index - traci.vehicle.getLaneIndex(v) * 4 + yStartPos][startPos] = 1
                            else:
                                yStartPos += 1
                        if temp_y_end > 0.25:
                            positionMatrix[index - traci.vehicle.getLaneIndex(v) * 4 + yStartPos][startPos] = 1
                            for i in range(yStartPos, yEndPos + 1):
                                positionMatrix[index - traci.vehicle.getLaneIndex(v) * 4 + i][startPos] = 1
                            for i in range(startPos + 1, endPos):
                                for j in range(yStartPos, yEndPos + 1):
                                    positionMatrix[index - traci.vehicle.getLaneIndex(v) * 4 + j][i] = 1
                        else:
                            for i in range(yStartPos, yEndPos):
                                positionMatrix[index - traci.vehicle.getLaneIndex(v) * 4 + i][startPos] = 1
                            for i in range(startPos + 1, endPos):
                                for j in range(yStartPos, yEndPos):
                                    positionMatrix[index - traci.vehicle.getLaneIndex(v) * 4 + j][i] = 1

        offset = 82.58 - cellLength * sizeLaneMatric
        for v in vehicles_road2_in:
            if traci.vehicle.getVehicleClass(v) != 'pedestrian':
                std = traci.vehicle.getLanePosition(v) - offset
                if std >= 0:
                    # if traci.vehicle.getVehicleClass(v) == 'taxi':
                    temp = std / cellLength
                    endPos = int(temp)
                    temp = (temp - endPos) / cellLength

                    temp_y_start = (1.75 - (
                            traci.vehicle.getLateralLanePosition(v) + traci.vehicle.getWidth(v) / 2)) / 0.875
                    yStartPos = int(temp_y_start)

                    temp_y_end = 1.75 - (
                            traci.vehicle.getLateralLanePosition(v) - traci.vehicle.getWidth(v) / 2)
                    temp_y_end = temp_y_end / 0.875
                    yEndPos = int(temp_y_end)
                    if temp > 0.25:
                        endPos += 1
                    if temp_y_start < 0:
                        positionMatrix[sizeMatrix - 1 - endPos][
                            index - traci.vehicle.getLaneIndex(v) * 4 + yStartPos] = 1
                    else:
                        temp_y_start2 = (temp_y_start - yStartPos) / 0.875
                        if temp_y_start2 < 0.75:
                            positionMatrix[sizeMatrix - 1 - endPos][
                                index - traci.vehicle.getLaneIndex(v) * 4 + yStartPos] = 1
                    temp_y_end = (temp_y_end - yEndPos) / 0.875
                    if temp_y_end > 0.25:
                        positionMatrix[sizeMatrix - 1 - endPos][index - traci.vehicle.getLaneIndex(v) * 4 + yEndPos] = 1
                        for i in range(yStartPos, yEndPos + 1):
                            positionMatrix[sizeMatrix - 1 - endPos][index - traci.vehicle.getLaneIndex(v) * 4 + i] = 1
                    else:
                        for i in range(yStartPos, yEndPos):
                            positionMatrix[sizeMatrix - 1 - endPos][index - traci.vehicle.getLaneIndex(v) * 4 + i] = 1

                    temp = (std - traci.vehicle.getLength(v)) / cellLength
                    startPos = int(temp)
                    temp = (temp - startPos) / cellLength
                    if temp < 0.75:
                        if (startPos < 0) & (endPos >= 0):
                            startPos = 0
                        if temp_y_start < 0:
                            positionMatrix[sizeMatrix - 1 - startPos][
                                index - traci.vehicle.getLaneIndex(v) * 4 + yStartPos] = 1
                        else:
                            temp_y_start2 = (temp_y_start - yStartPos) / 0.875
                            if temp_y_start2 < 0.75:
                                positionMatrix[sizeMatrix - 1 - startPos][
                                    index - traci.vehicle.getLaneIndex(v) * 4 + yStartPos] = 1
                            else:
                                yStartPos += 1
                        if temp_y_end > 0.25:
                            positionMatrix[sizeMatrix - 1 - startPos][
                                index - traci.vehicle.getLaneIndex(v) * 4 + yStartPos] = 1
                            for i in range(yStartPos, yEndPos + 1):
                                positionMatrix[sizeMatrix - 1 - startPos][
                                    index - traci.vehicle.getLaneIndex(v) * 4 + i] = 1
                            for i in range(startPos + 1, endPos):
                                for j in range(yStartPos, yEndPos + 1):
                                    positionMatrix[sizeMatrix - 1 - i][
                                        index - traci.vehicle.getLaneIndex(v) * 4 + j] = 1
                        else:
                            for i in range(yStartPos, yEndPos):
                                positionMatrix[sizeMatrix - 1 - startPos][
                                    index - traci.vehicle.getLaneIndex(v) * 4 + i] = 1
                            for i in range(startPos + 1, endPos):
                                for j in range(yStartPos, yEndPos):
                                    positionMatrix[sizeMatrix - 1 - i][
                                        index - traci.vehicle.getLaneIndex(v) * 4 + j] = 1

        offset = cellLength*sizeLaneMatric
        for v in vehicles_road3_out:
            if traci.vehicle.getVehicleClass(v) != 'pedestrian':
                std = offset - traci.vehicle.getLanePosition(v)
                if std >= 0:
                    temp = (std+ traci.vehicle.getLength(v)) / cellLength
                    endPos = int(temp)
                    temp = (temp - endPos) / cellLength

                    temp_y_start = (1.75 - (
                            traci.vehicle.getLateralLanePosition(v) + traci.vehicle.getWidth(v) / 2)) / 0.875
                    yStartPos = int(temp_y_start)

                    temp_y_end = 1.75 - (traci.vehicle.getLateralLanePosition(v) - traci.vehicle.getWidth(v) / 2)
                    temp_y_end = temp_y_end / 0.875
                    yEndPos = int(temp_y_end)
                    if temp > 0.25:
                        endPos += 1
                    if temp_y_start < 0:
                        positionMatrix[index - traci.vehicle.getLaneIndex(v) * 4 + yStartPos][sizeMatrix - 1 - endPos] = 1
                    else:
                        temp_y_start2 = (temp_y_start - yStartPos) / 0.875
                        if temp_y_start2 < 0.75:
                            positionMatrix[index - traci.vehicle.getLaneIndex(v) * 4 + yStartPos][sizeMatrix - 1 - endPos] = 1
                    temp_y_end = (temp_y_end - yEndPos) / 0.875
                    if temp_y_end > 0.25:
                        positionMatrix[index - traci.vehicle.getLaneIndex(v) * 4 + yEndPos][sizeMatrix - 1 - endPos] = 1
                        for i in range(yStartPos, yEndPos + 1):
                            positionMatrix[index - traci.vehicle.getLaneIndex(v) * 4 + i][sizeMatrix - 1 - endPos] = 1
                    else:
                        for i in range(yStartPos, yEndPos):
                            positionMatrix[index - traci.vehicle.getLaneIndex(v) * 4 + i][sizeMatrix - 1 - endPos] = 1

                    temp = (std) / cellLength
                    startPos = int(temp)
                    temp = (temp - startPos) / cellLength
                    if temp < 0.75:
                        if (startPos < 0) & (endPos >= 0):
                            startPos = 0
                        if temp_y_start < 0:
                            positionMatrix[index - traci.vehicle.getLaneIndex(v) * 4 + yStartPos][sizeMatrix - 1 - startPos] = 1
                        else:
                            temp_y_start2 = (temp_y_start - yStartPos) / 0.875
                            if temp_y_start2 < 0.75:
                                positionMatrix[index - traci.vehicle.getLaneIndex(v) * 4 + yStartPos][sizeMatrix - 1 - startPos] = 1
                            else:
                                yStartPos += 1
                        if temp_y_end > 0.25:
                            positionMatrix[index - traci.vehicle.getLaneIndex(v) * 4 + yStartPos][sizeMatrix - 1 - startPos] = 1
                            for i in range(yStartPos, yEndPos + 1):
                                positionMatrix[index - traci.vehicle.getLaneIndex(v) * 4 + i][sizeMatrix - 1 - startPos] = 1
                            for i in range(startPos + 1, endPos):
                                for j in range(yStartPos, yEndPos + 1):
                                    positionMatrix[index - traci.vehicle.getLaneIndex(v) * 4 + j][sizeMatrix - 1 - i] = 1
                        else:
                            for i in range(yStartPos, yEndPos):
                                positionMatrix[index - traci.vehicle.getLaneIndex(v) * 4 + i][sizeMatrix - 1 - startPos] = 1
                            for i in range(startPos + 1, endPos):
                                for j in range(yStartPos, yEndPos):
                                    positionMatrix[index - traci.vehicle.getLaneIndex(v) * 4 + j][sizeMatrix - 1 - i] = 1

        # offset = cellLength * sizeLaneMatric
        for v in vehicles_road4_out:
            if traci.vehicle.getVehicleClass(v) != 'pedestrian':
                std = offset - traci.vehicle.getLanePosition(v)
                if std >= 0:
                    # if traci.vehicle.getVehicleClass(v) == 'taxi':
                    temp = (std + traci.vehicle.getLength(v)) / cellLength
                    endPos = int(temp)
                    temp = (temp - endPos) / cellLength

                    temp_y_start = (1.75 - (traci.vehicle.getLateralLanePosition(v) + traci.vehicle.getWidth(v) / 2)) / 0.875
                    yStartPos = int(temp_y_start)

                    temp_y_end = 1.75 - (traci.vehicle.getLateralLanePosition(v) - traci.vehicle.getWidth(v) / 2)
                    temp_y_end = temp_y_end / 0.875
                    yEndPos = int(temp_y_end)
                    if temp > 0.25:
                        endPos += 1
                    if temp_y_start < 0:
                        positionMatrix[endPos][index - traci.vehicle.getLaneIndex(v) * 4 + yStartPos] = 1
                    else:
                        temp_y_start2 = (temp_y_start - yStartPos) / 0.875
                        if temp_y_start2 < 0.75:
                            positionMatrix[endPos][index - traci.vehicle.getLaneIndex(v) * 4 + yStartPos] = 1
                    temp_y_end = (temp_y_end - yEndPos) / 0.875
                    if temp_y_end > 0.25:
                        positionMatrix[endPos][index - traci.vehicle.getLaneIndex(v) * 4 + yEndPos] = 1
                        for i in range(yStartPos, yEndPos + 1):
                            positionMatrix[endPos][index - traci.vehicle.getLaneIndex(v) * 4 + i] = 1
                    else:
                        for i in range(yStartPos, yEndPos):
                            positionMatrix[endPos][index - traci.vehicle.getLaneIndex(v) * 4 + i] = 1

                    temp = std / cellLength
                    startPos = int(temp)
                    temp = (temp - startPos) / cellLength
                    if temp < 0.75:
                        if (startPos < 0) & (endPos >= 0):
                            startPos = 0

                        if temp_y_start < 0:
                            positionMatrix[startPos][index - traci.vehicle.getLaneIndex(v) * 4 + yStartPos] = 1
                        else:
                            temp_y_start2 = (temp_y_start - yStartPos) / 0.875
                            if temp_y_start2 < 0.75:
                                positionMatrix[startPos][index - traci.vehicle.getLaneIndex(v) * 4 + yStartPos] = 1
                            else:
                                yStartPos += 1
                        if temp_y_end > 0.25:
                            positionMatrix[startPos][index - traci.vehicle.getLaneIndex(v) * 4 + yStartPos] = 1
                            for i in range(yStartPos, yEndPos + 1):
                                positionMatrix[startPos][index - traci.vehicle.getLaneIndex(v) * 4 + i] = 1
                            for i in range(startPos + 1, endPos):
                                for j in range(yStartPos, yEndPos + 1):
                                    positionMatrix[i][index - traci.vehicle.getLaneIndex(v) * 4 + j] = 1
                        else:
                            for i in range(yStartPos, yEndPos):
                                positionMatrix[startPos][index - traci.vehicle.getLaneIndex(v) * 4 + i] = 1
                            for i in range(startPos + 1, endPos):
                                for j in range(yStartPos, yEndPos):
                                    positionMatrix[i][index - traci.vehicle.getLaneIndex(v) * 4 + j] = 1
        index = 14
        # offset = cellLength * sizeLaneMatric
        for v in vehicles_road1_out:
            if traci.vehicle.getVehicleClass(v) != 'pedestrian':
                std = offset - traci.vehicle.getLanePosition(v)
                if std >= 0:
                    temp = (std+ traci.vehicle.getLength(v)) / cellLength
                    endPos = int(temp)
                    temp = (temp - endPos) / cellLength

                    temp_y_start = (1.75 - (
                            traci.vehicle.getLateralLanePosition(v) + traci.vehicle.getWidth(v) / 2)) / 0.875
                    yStartPos = int(temp_y_start)

                    temp_y_end = 1.75 - (traci.vehicle.getLateralLanePosition(v) - traci.vehicle.getWidth(v) / 2)
                    temp_y_end = temp_y_end / 0.875
                    yEndPos = int(temp_y_end)
                    if temp > 0.25:
                        endPos += 1
                    if temp_y_start < 0:
                        positionMatrix[index + traci.vehicle.getLaneIndex(v) * 4 + 3-yStartPos][endPos] = 1
                    else:
                        temp_y_start2 = (temp_y_start - yStartPos) / 0.875
                        if temp_y_start2 < 0.75:
                            positionMatrix[index + traci.vehicle.getLaneIndex(v) * 4 + 3-yStartPos][endPos] = 1
                    temp_y_end = (temp_y_end - yEndPos) / 0.875
                    if temp_y_end > 0.25:
                        positionMatrix[index + traci.vehicle.getLaneIndex(v) * 4 + 3-yEndPos][endPos] = 1
                        for i in range(yStartPos, yEndPos + 1):
                            positionMatrix[index + traci.vehicle.getLaneIndex(v) * 4 + 3-i][endPos] = 1
                    else:
                        for i in range(yStartPos, yEndPos):
                            positionMatrix[index + traci.vehicle.getLaneIndex(v) * 4 + 3-i][endPos] = 1

                    temp = (std) / cellLength
                    startPos = int(temp)
                    temp = (temp - startPos) / cellLength
                    if temp < 0.75:
                        if (startPos < 0) & (endPos >= 0):
                            startPos = 0
                        if temp_y_start < 0:
                            positionMatrix[index + traci.vehicle.getLaneIndex(v) * 4 + 3-yStartPos][startPos] = 1
                        else:
                            temp_y_start2 = (temp_y_start - yStartPos) / 0.875
                            if temp_y_start2 < 0.75:
                                positionMatrix[index + traci.vehicle.getLaneIndex(v) * 4 + 3-yStartPos][startPos] = 1
                            else:
                                yStartPos += 1
                        if temp_y_end > 0.25:
                            positionMatrix[index + traci.vehicle.getLaneIndex(v) * 4 + 3-yStartPos][startPos] = 1
                            for i in range(yStartPos, yEndPos + 1):
                                positionMatrix[index + traci.vehicle.getLaneIndex(v) * 4 + 3-i][startPos] = 1
                            for i in range(startPos + 1, endPos):
                                for j in range(yStartPos, yEndPos + 1):
                                    positionMatrix[index + traci.vehicle.getLaneIndex(v) * 4 + 3-j][i] = 1
                        else:
                            for i in range(yStartPos, yEndPos):
                                positionMatrix[index + traci.vehicle.getLaneIndex(v) * 4 + 3-i][startPos] = 1
                            for i in range(startPos + 1, endPos):
                                for j in range(yStartPos, yEndPos):
                                    positionMatrix[index + traci.vehicle.getLaneIndex(v) * 4 + 3-j][i] = 1

        for v in vehicles_road2_out:
            if traci.vehicle.getVehicleClass(v) != 'pedestrian':
                std = offset - traci.vehicle.getLanePosition(v)
                if std >= 0:
                    # if traci.vehicle.getVehicleClass(v) == 'taxi':
                    temp = (std+ traci.vehicle.getLength(v)) / cellLength
                    endPos = int(temp)
                    temp = (temp - endPos) / cellLength

                    temp_y_start = (1.75 - (
                            traci.vehicle.getLateralLanePosition(v) + traci.vehicle.getWidth(v) / 2)) / 0.875
                    yStartPos = int(temp_y_start)

                    temp_y_end = 1.75 - (
                            traci.vehicle.getLateralLanePosition(v) - traci.vehicle.getWidth(v) / 2)
                    temp_y_end = temp_y_end / 0.875
                    yEndPos = int(temp_y_end)
                    if temp > 0.25:
                        endPos += 1
                    if temp_y_start < 0:
                        positionMatrix[sizeMatrix - 1 - endPos][
                            index + traci.vehicle.getLaneIndex(v) * 4 + 3-yStartPos] = 1
                    else:
                        temp_y_start2 = (temp_y_start - yStartPos) / 0.875
                        if temp_y_start2 < 0.75:
                            positionMatrix[sizeMatrix - 1 - endPos][
                                index + traci.vehicle.getLaneIndex(v) * 4 + 3-yStartPos] = 1
                    temp_y_end = (temp_y_end - yEndPos) / 0.875
                    if temp_y_end > 0.25:
                        positionMatrix[sizeMatrix - 1 - endPos][index + traci.vehicle.getLaneIndex(v) * 4 + 3-yEndPos] = 1
                        for i in range(yStartPos, yEndPos + 1):
                            positionMatrix[sizeMatrix - 1 - endPos][index + traci.vehicle.getLaneIndex(v) * 4 + 3-i] = 1
                    else:
                        for i in range(yStartPos, yEndPos):
                            positionMatrix[sizeMatrix - 1 - endPos][index + traci.vehicle.getLaneIndex(v) * 4 + 3-i] = 1

                    temp = (std) / cellLength
                    startPos = int(temp)
                    temp = (temp - startPos) / cellLength
                    if temp < 0.75:
                        if (startPos < 0) & (endPos >= 0):
                            startPos = 0

                        if temp_y_start < 0:
                            positionMatrix[sizeMatrix - 1 - startPos][
                                index + traci.vehicle.getLaneIndex(v) * 4 + 3-yStartPos] = 1
                        else:
                            temp_y_start2 = (temp_y_start - yStartPos) / 0.875
                            if temp_y_start2 < 0.75:
                                positionMatrix[sizeMatrix - 1 - startPos][
                                    index + traci.vehicle.getLaneIndex(v) * 4 + 3-yStartPos] = 1
                            else:
                                yStartPos += 1
                        if temp_y_end > 0.25:
                            positionMatrix[sizeMatrix - 1 - startPos][
                                index + traci.vehicle.getLaneIndex(v) * 4 + 3-yStartPos] = 1
                            for i in range(yStartPos, yEndPos + 1):
                                positionMatrix[sizeMatrix - 1 - startPos][
                                    index + traci.vehicle.getLaneIndex(v) * 4 + 3-i] = 1
                            for i in range(startPos + 1, endPos):
                                for j in range(yStartPos, yEndPos + 1):
                                    positionMatrix[sizeMatrix - 1 - i][
                                        index + traci.vehicle.getLaneIndex(v) * 4 + 3-j] = 1
                        else:
                            for i in range(yStartPos, yEndPos):
                                positionMatrix[sizeMatrix - 1 - startPos][
                                    index + traci.vehicle.getLaneIndex(v) * 4 + 3-i] = 1
                            for i in range(startPos + 1, endPos):
                                for j in range(yStartPos, yEndPos):
                                    positionMatrix[sizeMatrix - 1 - i][
                                        index + traci.vehicle.getLaneIndex(v) * 4 + 3-j] = 1

        offset = 107.06 - cellLength*sizeLaneMatric
        for v in vehicles_road3_in:
            if traci.vehicle.getVehicleClass(v) != 'pedestrian':
                std = traci.vehicle.getLanePosition(v) - offset
                if std >= 0:
                    temp = std / cellLength
                    endPos = int(temp)
                    temp = (temp - endPos) / cellLength

                    temp_y_start = (1.75 - (
                            traci.vehicle.getLateralLanePosition(v) + traci.vehicle.getWidth(v) / 2)) / 0.875
                    yStartPos = int(temp_y_start)

                    temp_y_end = 1.75 - (traci.vehicle.getLateralLanePosition(v) - traci.vehicle.getWidth(v) / 2)
                    temp_y_end = temp_y_end / 0.875
                    yEndPos = int(temp_y_end)
                    if temp > 0.25:
                        endPos += 1
                    if temp_y_start < 0:
                        positionMatrix[index + traci.vehicle.getLaneIndex(v) * 4 + 3-yStartPos][sizeMatrix - 1 - endPos] = 1
                    else:
                        temp_y_start2 = (temp_y_start - yStartPos) / 0.875
                        if temp_y_start2 < 0.75:
                            positionMatrix[index + traci.vehicle.getLaneIndex(v) * 4 + 3-yStartPos][sizeMatrix - 1 - endPos] = 1
                    temp_y_end = (temp_y_end - yEndPos) / 0.875
                    if temp_y_end > 0.25:
                        positionMatrix[index + traci.vehicle.getLaneIndex(v) * 4 + 3-yEndPos][sizeMatrix - 1 - endPos] = 1
                        for i in range(yStartPos, yEndPos + 1):
                            positionMatrix[index + traci.vehicle.getLaneIndex(v) * 4 + 3-i][sizeMatrix - 1 - endPos] = 1
                    else:
                        for i in range(yStartPos, yEndPos):
                            positionMatrix[index + traci.vehicle.getLaneIndex(v) * 4 + 3-i][sizeMatrix - 1 - endPos] = 1

                    temp = (std - traci.vehicle.getLength(v)) / cellLength
                    startPos = int(temp)
                    temp = (temp - startPos) / cellLength
                    if temp < 0.75:
                        if (startPos < 0) & (endPos >= 0):
                            startPos = 0
                        if temp_y_start < 0:
                            positionMatrix[index + traci.vehicle.getLaneIndex(v) * 4 + 3-yStartPos][sizeMatrix - 1 - startPos] = 1
                        else:
                            temp_y_start2 = (temp_y_start - yStartPos) / 0.875
                            if temp_y_start2 < 0.75:
                                positionMatrix[index + traci.vehicle.getLaneIndex(v) * 4 + 3-yStartPos][sizeMatrix - 1 - startPos] = 1
                            else:
                                yStartPos += 1
                        if temp_y_end > 0.25:
                            positionMatrix[index + traci.vehicle.getLaneIndex(v) * 4 + 3-yStartPos][sizeMatrix - 1 - startPos] = 1
                            for i in range(yStartPos, yEndPos + 1):
                                positionMatrix[index + traci.vehicle.getLaneIndex(v) * 4 + 3-i][sizeMatrix - 1 - startPos] = 1
                            for i in range(startPos + 1, endPos):
                                for j in range(yStartPos, yEndPos + 1):
                                    positionMatrix[index + traci.vehicle.getLaneIndex(v) * 4 + 3-j][sizeMatrix - 1 - i] = 1
                        else:
                            for i in range(yStartPos, yEndPos):
                                positionMatrix[index + traci.vehicle.getLaneIndex(v) * 4 + 3-i][sizeMatrix - 1 - startPos] = 1
                            for i in range(startPos + 1, endPos):
                                for j in range(yStartPos, yEndPos):
                                    positionMatrix[index + traci.vehicle.getLaneIndex(v) * 4 + 3-j][sizeMatrix - 1 - i] = 1

        offset = 72.64 - cellLength * sizeLaneMatric
        for v in vehicles_road4_in:
            if traci.vehicle.getVehicleClass(v) != 'pedestrian':
                std = traci.vehicle.getLanePosition(v) - offset
                if std >= 0:
                    # if traci.vehicle.getVehicleClass(v) == 'taxi':
                    temp = std / cellLength
                    endPos = int(temp)
                    temp = (temp - endPos) / cellLength

                    temp_y_start = (1.75 - (
                            traci.vehicle.getLateralLanePosition(v) + traci.vehicle.getWidth(v) / 2)) / 0.875
                    yStartPos = int(temp_y_start)

                    temp_y_end = 1.75 - (
                            traci.vehicle.getLateralLanePosition(v) - traci.vehicle.getWidth(v) / 2)
                    temp_y_end = temp_y_end / 0.875
                    yEndPos = int(temp_y_end)
                    if temp > 0.25:
                        endPos += 1
                    if temp_y_start < 0:
                        positionMatrix[endPos][index + traci.vehicle.getLaneIndex(v) * 4 +3-yStartPos] = 1
                    else:
                        temp_y_start2 = (temp_y_start - yStartPos) / 0.875
                        if temp_y_start2 < 0.75:
                            positionMatrix[endPos][
                                index + traci.vehicle.getLaneIndex(v) * 4 +3-yStartPos] = 1
                    temp_y_end = (temp_y_end - yEndPos) / 0.875
                    if temp_y_end > 0.25:
                        positionMatrix[endPos][index + traci.vehicle.getLaneIndex(v) * 4 +3-yEndPos] = 1
                        for i in range(yStartPos, yEndPos + 1):
                            positionMatrix[endPos][index + traci.vehicle.getLaneIndex(v) * 4 +3-i] = 1
                    else:
                        for i in range(yStartPos, yEndPos):
                            positionMatrix[endPos][index + traci.vehicle.getLaneIndex(v) * 4 +3-i] = 1

                    temp = (std - traci.vehicle.getLength(v)) / cellLength
                    startPos = int(temp)
                    temp = (temp - startPos) / cellLength
                    if temp < 0.75:
                        if (startPos < 0) & (endPos >= 0):
                            startPos = 0

                        if temp_y_start < 0:
                            positionMatrix[startPos][
                                index + traci.vehicle.getLaneIndex(v) * 4 +3-yStartPos] = 1
                        else:
                            temp_y_start2 = (temp_y_start - yStartPos) / 0.875
                            if temp_y_start2 < 0.75:
                                positionMatrix[startPos][
                                    index + traci.vehicle.getLaneIndex(v) * 4 +3-yStartPos] = 1
                            else:
                                yStartPos += 1
                        if temp_y_end > 0.25:
                            positionMatrix[startPos][index + traci.vehicle.getLaneIndex(v) * 4 +3-yStartPos] = 1
                            for i in range(yStartPos, yEndPos + 1):
                                positionMatrix[startPos][index + traci.vehicle.getLaneIndex(v) * 4 +3-i] = 1
                            for i in range(startPos + 1, endPos):
                                for j in range(yStartPos, yEndPos + 1):
                                    positionMatrix[i][index + traci.vehicle.getLaneIndex(v) * 4 +3-j] = 1
                        else:
                            for i in range(yStartPos, yEndPos):
                                positionMatrix[startPos][index + traci.vehicle.getLaneIndex(v) * 4 +3-i] = 1
                            for i in range(startPos + 1, endPos):
                                for j in range(yStartPos, yEndPos):
                                    positionMatrix[i][index + traci.vehicle.getLaneIndex(v) * 4 +3-j] = 1

        # for v in positionMatrix:
        #     print(v)
        # for i in range(0,60):
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
        # global count_action_dif_default
        # if count_action_dif_default > 1:
        #     tentative_action_matrix = tentative_action[0]
        #     count_action_dif_default = 0
        # else:
        #     tentative_action_matrix = tentative_action[action]
        # print  tentative_action_matrix
        tentative_action_matrix = tentative_action[action]
        return [output, I, tentative_action_matrix]

    def cal_yellow_phase(self, id_list, a_dec):
        v_on_road = []
        for id in id_list:
            vehicles_road = traci.edge.getLastStepVehicleIDs(id)
            for vehicle in vehicles_road:
                v_on_road.append(traci.vehicle.getSpeed(vehicle))
        if not v_on_road:
            v_on_road.append(0)

        return int(np.amax(v_on_road)/a_dec)

def cal_waiting_time():
    waiting_time = 0
    waiting_time += (traci.edge.getLastStepHaltingNumber('gneE21')
                     + traci.edge.getLastStepHaltingNumber('gneE86')
                     + traci.edge.getLastStepHaltingNumber('gneE89')
                     + traci.edge.getLastStepHaltingNumber('gneE85'))
    return waiting_time

def main():
    # Control code here
    M = 20000 #size memory
    B = 64 #minibatch_size
    a_dec = 4.5 # m/s^2
    phase_number = 2
    action_space = phase_number * 2 + 1
    action_policy = [[0, 0], [5, 0], [-5, 0], [0, 5], [0, -5]]
    tentative_action = [np.asarray([1,1,1,1,1]).reshape(1, action_space),np.asarray([1,0,0,0,0]).reshape(1, action_space),
                        np.asarray([1,0,0,0,0]).reshape(1, action_space),np.asarray([1,0,0,0,0]).reshape(1, action_space),
                        np.asarray([1,0,0,0,0]).reshape(1, action_space)]
    # global count_action_dif_default
    I = np.full((action_space, action_space), 0.5).reshape(1, action_space, action_space)
    idLightControl = '4628048104'
    waiting_time_t = 0
    i = 0
    agent = DQNAgent(M, action_space, B)
    try:
        agent.load('Models/reinf_traf_control_v5.h5')
    except:
        print('No models found')

    sumo_int = SumoIntersection()

    episodes = 2000
    sumo_cmd = [sumoBinary, "-c", sumoConfig, '--start']
    for e in range(episodes):
        traci.start(sumo_cmd)
        action = 0
        # count_action_dif_default = 0
        action_time = [33,33]
        zstep = 0
        state = sumo_int.getState(I, action, tentative_action)
        if i > 20000:
            break
        while (traci.simulation.getMinExpectedNumber() > 0) & (zstep < 700):
            traci.simulationStep()
            waiting_time = 0
            # print '------------------------------------------- ', action,state[2], ' --------------------'
            action = agent.act(state)
            # if action != 0:
            #     count_action_dif_default += 1
            # print '------------------------------------------- ', action, ' --------------------'
            for j in range(phase_number):
                action_time[j] += action_policy[action][j]
                if action_time[j] < 0:
                    action_time[j] = 0
                elif action_time[j] > 60:
                    action_time[j] = 60

            # print action_time[0]
            for j in range(action_time[0]):
                traci.trafficlight.setPhase(idLightControl, 0)
                waiting_time += cal_waiting_time()
                traci.simulationStep()

            yellow_time1 = sumo_int.cal_yellow_phase(['gneE21', 'gneE89'], a_dec)
            # print waiting_time#yellow_time1
            for j in range(yellow_time1):
                traci.trafficlight.setPhase(idLightControl, 1)
                waiting_time += cal_waiting_time()
                traci.simulationStep()

            # print waiting_time#action_time[1]
            for j in range(action_time[1]):
                traci.trafficlight.setPhase(idLightControl, 2)
                waiting_time += cal_waiting_time()
                traci.simulationStep()

            yellow_time2 = sumo_int.cal_yellow_phase(['gneE86', 'gneE85'], a_dec)
            # print waiting_time#yellow_time2
            for j in range(yellow_time2):
                traci.trafficlight.setPhase(idLightControl, 3)
                waiting_time += cal_waiting_time()
                traci.simulationStep()

            waiting_time_t1 = waiting_time
            reward_t = waiting_time_t - waiting_time_t1
            # print waiting_time_t, waiting_time_t1, reward_t
            waiting_time_t = waiting_time_t1

            new_state = sumo_int.getState(I, action, tentative_action)
            agent.remember(state, action, reward_t, new_state, False)
            state = new_state
            i += 1
            zstep+=1
            print '------------------------------------------- ', i, action_time, ' --------------------'
            if (len(agent.memory) > B) & (i > agent.tp):
                # if len(agent.memory) > 100 & (i > 1):
                #     print '-------------------------------------------BEGIN REPLAY------------------------'
                agent.replay()
                agent.start_epsilon -= agent.epsilon_decay
            # print reward_t, 'in step ', i
            # print('-----------------------end simulation----------------------------')
        agent.save('Models/reinf_traf_control_v5.h5')
        traci.close(wait=False)


if __name__ == '__main__':
    main()
sys.stdout.flush()