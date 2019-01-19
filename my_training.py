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

class DQNAgent:
    def __init__(self):
        self.memory = deque(maxlen=20000)
        self.minibatch_size = 64
        self.start_epsilon = 1
        self.end_epsilon = 0.01
        self.step_epsilon = 10000
        self.tp = 2000          # pre-training steps
        self.alpha = 0.0001     # target network update rate
        self.gamma = 0.99       # discount factor
        self.epsilon_r = 0.0001 # learning rate
        self.Beta = 0.01        # Leaky ReLU
        self.action_size = 2
        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        input_1 = Input(shape=(60, 60, 2))
        input_2 = Input(shape=(2, 2))
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
        # if self.end_epsilon <= self.start_epsilon:
        #     return random.randrange(self.action_size)
        act_values = self.model.predict(state)

        return np.argmax(act_values)  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)

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

    def getState(self):
        positionMatrix = []
        velocityMatrix = []

        cellLength = 2.7
        sizeMatrix = 60
        sizeLaneMatric = sizeMatrix/2-3           #27
        offset_in = 500-cellLength*sizeLaneMatric #427.1
        offset_out = cellLength*sizeLaneMatric    #72.9
        speedLimit = 14
	#print(traci.edge.getWaitingTime('gneE21'))
	
        junctionPosition = traci.junction.getPosition('0')[0]
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


            #print('class ', traci.vehicle.getVehicleClass(v), ' with length ', traci.vehicle.getLength(v),' in position ', traci.vehicle.getPosition(v))

            #ind = int(
            #    abs((junctionPosition - traci.vehicle.getPosition(v)[0] - offset_in)) / cellLength)
        #     # if (ind < 12):
        #     #     positionMatrix[2 - traci.vehicle.getLaneIndex(v)][11 - ind] = 1
        #     #     velocityMatrix[2 - traci.vehicle.getLaneIndex(
        #     #         v)][11 - ind] = traci.vehicle.getSpeed(v) / speedLimit
        outputMatrix = [positionMatrix, velocityMatrix]
        output = np.transpose(outputMatrix)#np.array(outputMatrix)
        output = output.reshape(1,60,60,2)
        I = [[0.5, 0.5], [0.5, 0.5]]
        I = np.array(I)
        I = I.reshape(1, 2, 2)
        # print output.tolist()
        # print('---------------------------------------------')

        return [output, I]

waiting_time_last = 0
def calc_reward():
    global waiting_time_last
    vehicles_road1_in = traci.edge.getWaitingTime('L53')
    vehicles_road1_out = traci.edge.getWaitingTime('L35')
    vehicles_road2_in = traci.edge.getWaitingTime('D3')
    vehicles_road2_out = traci.edge.getWaitingTime('D4')
    vehicles_road3_in = traci.edge.getWaitingTime('L43')
    vehicles_road3_out = traci.edge.getWaitingTime('L34')
    vehicles_road4_in = traci.edge.getWaitingTime('L23')
    vehicles_road4_out = traci.edge.getWaitingTime('L32')
    r = waiting_time_last - (vehicles_road1_in +vehicles_road1_out+vehicles_road2_in+vehicles_road2_out
    + vehicles_road3_in + vehicles_road3_out+vehicles_road4_in+vehicles_road4_out)
    waiting_time_last = vehicles_road1_in +vehicles_road1_out+vehicles_road2_in+vehicles_road2_out
    + vehicles_road3_in + vehicles_road3_out+vehicles_road4_in+vehicles_road4_out
    return r


def main():
    # Control code here
    sumoInt = SumoIntersection()
    sumoCmd = [sumoBinary, "-c", sumoConfig]
    traci.start(sumoCmd)
    step = 0
    agent = DQNAgent()
    while step < 10000:
        # print(calc_reward())
        traci.simulationStep()
        state = sumoInt.getState()
        action = agent.act(state)
        print action
        print('-----------------------end simulation----------------------------')
        step += 1
    traci.close()

if __name__ == '__main__':
    main()
