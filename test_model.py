import sys
import os
import numpy as np
import time
from scipy.spatial.ckdtree import coo_entries

# split file
import DQNAgent
import SumoIntersection

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

import traci

sumoBinary = "/usr/bin/sumo-gui"
sumoConfig = "sumoconfig.sumoconfig"


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
    log = open('Logs_result/log-model.txt', 'w')
    time_plot = []
    waiting_time_plot = []
    reward_t_plot = []
    time_reward_t_plot = []

    # Control code here
    memory_size = 20000  # size memory
    mini_batch_size = 64  # minibatch_size
    a_dec = 4.5  # m/s^2
    num_of_phase = 2  # 2 phase
    action_space_size = num_of_phase * 2 + 1  # 5 actions
    action_policy = [[0, 0], [5, 0], [-5, 0], [0, 5], [0, -5]]
    tentative_action = [np.asarray([1, 1, 1, 1, 1]).reshape(1, action_space_size),
                        np.asarray([1, 0, 0, 0, 0]).reshape(1, action_space_size),
                        np.asarray([1, 0, 0, 0, 0]).reshape(1, action_space_size),
                        np.asarray([1, 0, 0, 0, 0]).reshape(1, action_space_size),
                        np.asarray([1, 0, 0, 0, 0]).reshape(1, action_space_size)]

    # global count_action_dif_default
    I = np.full((action_space_size, action_space_size), 0.5).reshape(1, action_space_size, action_space_size)
    idLightControl = '4628048104'
    waiting_time_t = 0
    numb_of_cycle = 0

    # new Agent.
    agent = DQNAgent.DQNAgent(memory_size, action_space_size, mini_batch_size)
    try:
        agent.load('Models/reinf_traf_control_v13_random_sample.h5')
    except:
        print('No models found')
    agent.start_epsilon = 0
    # new Sumo Intersection
    sumo_int = SumoIntersection.SumoIntersection()

    # 2000 episodes
    episodes = 2000

    # command to run SUMO
    sumo_cmd = [sumoBinary, "-c", sumoConfig, '--start']

    # run 2000 episodes
    for e in range(episodes):
        # start sumo simulation.
        traci.start(sumo_cmd)

        # init action.
        action = 0

        # time for each phase
        action_time = [33, 33]

        # getState by action.
        state, tentative_act_dec = sumo_int.getState(I, action, tentative_action)

        # break traning and save model.
        if numb_of_cycle > 30000:
            break

        # run a cycle.
        while traci.simulation.getMinExpectedNumber() > 0:
            traci.simulationStep()
            waiting_time = 0

            # get action.
            action = agent.select_action(state, tentative_act_dec)

            #  ============================================================ Perform action ======================:
            for j in range(num_of_phase):
                action_time[j] += action_policy[action][j]
                if action_time[j] < 0:
                    action_time[j] = 0
                elif action_time[j] > 60:
                    action_time[j] = 60
            for j in range(action_time[0]):
                traci.trafficlight.setPhase(idLightControl, 0)
                # waiting_time += cal_waiting_time()
                traci.simulationStep()
                time_plot.append(traci.simulation.getTime())
                waiting_time_plot.append(cal_waiting_time_average())
            yellow_time1 = sumo_int.cal_yellow_phase(['gneE21', 'gneE89'], a_dec)
            for j in range(yellow_time1):
                traci.trafficlight.setPhase(idLightControl, 1)
                # waiting_time += cal_waiting_time()
                traci.simulationStep()
                time_plot.append(traci.simulation.getTime())
                waiting_time_plot.append(cal_waiting_time_average())
            for j in range(action_time[1]):
                traci.trafficlight.setPhase(idLightControl, 2)
                # waiting_time += cal_waiting_time()
                traci.simulationStep()
                time_plot.append(traci.simulation.getTime())
                waiting_time_plot.append(cal_waiting_time_average())
            yellow_time2 = sumo_int.cal_yellow_phase(['gneE86', 'gneE85'], a_dec)
            for j in range(yellow_time2):
                traci.trafficlight.setPhase(idLightControl, 3)
                # waiting_time += cal_waiting_time()
                traci.simulationStep()
                time_plot.append(traci.simulation.getTime())
                waiting_time_plot.append(cal_waiting_time_average())
            #  ============================================================ Finish action ======================:

            # calculate REWARD
            waiting_time += cal_waiting_time()
            waiting_time_t1 = waiting_time
            reward_t = waiting_time_t - waiting_time_t1
            reward_t_plot.append(reward_t)
            time_reward_t_plot.append(traci.simulation.getTime())
            waiting_time_t = waiting_time_t1

            # get NewState by selected-action
            new_state, tentative_act_dec = sumo_int.getState(I, action, tentative_action)

            # reassign
            state = new_state
            numb_of_cycle+=1
            waiting_time_average = cal_waiting_time_average()
            print('action - ' + '(' + str(action_time[0]) + ',' + str(
                yellow_time1) + ',' + str(action_time[1]) + ',' + str(yellow_time2) + ')')
            log.write('action - ' + str(numb_of_cycle) + ', total waiting time - ' +
                      str(waiting_time_average) + ', action - ' + '(' + str(action_time[0]) + ',' + str(
                yellow_time1) + ',' + str(action_time[1]) + ',' + str(yellow_time2) + ')' + ', reward - ' + str(
                reward_t) + '\n')

        traci.close(wait=False)
        log.close()
        key = '_10000'
        np.save('array_plot/array_time'+key+'.npy', time_plot)
        np.save('array_plot/array_waiting_time'+key+'.npy', waiting_time_plot)

        np.save('array_plot/reward_t_plot'+key+'.npy', reward_t_plot)
        np.save('array_plot/time_reward_t_plot'+key+'.npy', time_reward_t_plot)
        break


if __name__ == '__main__':
    main()
sys.stdout.flush()