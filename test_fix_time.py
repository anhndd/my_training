import os
import sys
import random
import numpy as np
import time
import math
import h5py
from collections import deque
from keras.layers import Input, Conv2D, Flatten, Dense, LeakyReLU, Average, Add, Dot, Subtract
from keras.models import Model
from keras.optimizers import Adam
import tensorflow as tf
from scipy.spatial.ckdtree import coo_entries

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

sumoBinary = "/usr/bin/sumo-gui"
sumoConfig = "sumoconfig.sumoconfig"
import traci



def cal_waiting_time():
    return (traci.edge.getWaitingTime('gneE21') + traci.edge.getWaitingTime('gneE86') + traci.edge.getWaitingTime(
        'gneE89') + traci.edge.getWaitingTime('gneE85'))  # waiting_time

def cal_waiting_time_average():
    number_vehicle = (traci.edge.getLastStepVehicleNumber('gneE21')+traci.edge.getLastStepVehicleNumber('gneE86')
      +traci.edge.getLastStepVehicleNumber('gneE89')+traci.edge.getLastStepVehicleNumber('gneE85'))
    if number_vehicle == 0:
        return 0
    return (traci.edge.getWaitingTime('gneE21') + traci.edge.getWaitingTime('gneE86') + traci.edge.getWaitingTime(
        'gneE89') + traci.edge.getWaitingTime('gneE85')) / number_vehicle  # waiting_time

def main():
    # Control code here
    log = open('Logs_result/log_fix_time.txt', 'w')
    time_plot = []
    waiting_time_plot = []
    reward_t_plot = []
    time_reward_t_plot = []
    a_dec = 4.5 # m/s^2
    phase_number = 2
    action_space = phase_number * 2 + 1
    action_policy = [[0, 0], [5, 0], [-5, 0], [0, 5], [0, -5]]
    I = np.full((action_space, action_space), 0.5).reshape(1, action_space, action_space)
    time_test = 40
    action_time = [time_test, time_test]
    idLightControl = '4628048104'
    waiting_time_t = 0
    waiting_time_t1 = 0
    reward_t = 0
    i = 0
    # try:
    #     agent.load('Models/reinf_traf_control_v2.h5')
    # except:
    #     print('No models found')

    sumo_cmd = [sumoBinary, "-c", sumoConfig,'--start']
    traci.start(sumo_cmd)

    while (traci.simulation.getMinExpectedNumber() > 0) & (traci.simulation.getTime() < 10000):
        traci.simulationStep()
        waiting_time = 0

        # print action_time[0]
        for j in range(action_time[0]):
            traci.trafficlight.setPhase(idLightControl, 0)
            traci.simulationStep()
            time_plot.append(traci.simulation.getTime())
            waiting_time_plot.append(cal_waiting_time_average())

        yellow_time1 =  4
        # print waiting_time#yellow_time1
        for j in range(yellow_time1):
            traci.trafficlight.setPhase(idLightControl, 1)
            traci.simulationStep()
            time_plot.append(traci.simulation.getTime())
            waiting_time_plot.append(cal_waiting_time_average())

        # print waiting_time#action_time[1]
        for j in range(action_time[1]):
            traci.trafficlight.setPhase(idLightControl, 2)
            traci.simulationStep()
            time_plot.append(traci.simulation.getTime())
            waiting_time_plot.append(cal_waiting_time_average())

        yellow_time2 =  4
        # print waiting_time#yellow_time2
        for j in range(yellow_time2):
            traci.trafficlight.setPhase(idLightControl, 3)
            traci.simulationStep()
            time_plot.append(traci.simulation.getTime())
            waiting_time_plot.append(cal_waiting_time_average())

        waiting_time += cal_waiting_time()
        waiting_time_t1 = waiting_time
        reward_t = waiting_time_t - waiting_time_t1
        reward_t_plot.append(reward_t)
        time_reward_t_plot.append(traci.simulation.getTime())
        waiting_time_t = waiting_time_t1

        # new_state = sumo_int.getState(I)
        # agent.remember(state, action, reward_t, new_state, False)

        i += 1;
        waiting_time_average = cal_waiting_time_average()
        log.write('action - ' + str(i) + ', total waiting time - ' +
                 str(waiting_time_average)  + ', action - ' +'('+str(action_time[0])+','+str(yellow_time1)+','+str(action_time[1])+','+str(yellow_time2)+')'+ ', reward - ' + str(reward_t) +'\n')
        # print '\n--------------------------------------------------- ',waiting_time, 'in step ', i, ' ---------------------------------------------------\n'
    log.close()
    traci.close()
    key = '_10000_' + str(time_test)
    np.save('array_plot/array_time_fix'+key+'.npy', time_plot)
    np.save('array_plot/array_waiting_time_fix'+key+'.npy', waiting_time_plot)

    np.save('array_plot/reward_t_plot_fix'+key+'.npy', reward_t_plot)
    np.save('array_plot/time_reward_t_plot_fix'+key+'.npy', time_reward_t_plot)

if __name__ == '__main__':
    main()
sys.stdout.flush()