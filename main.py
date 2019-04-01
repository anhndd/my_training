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

sumoBinary = "/usr/bin/sumo"
sumoConfig = "sumoconfig.sumoconfig"

def cal_waiting_time():
    return (traci.edge.getWaitingTime('gneE21') + traci.edge.getWaitingTime('gneE86') + traci.edge.getWaitingTime('gneE89') + traci.edge.getWaitingTime('gneE85')) # waiting_time

def main():

    # Control code here
    memory_size = 20000                                     # size memory
    mini_batch_size = 64                                    # minibatch_size
    a_dec = 4.5                                             # m/s^2
    num_of_phase = 2                                        # 2 phase
    action_space_size = num_of_phase * 2 + 1                # 5 actions
    action_policy = [[0, 0], [5, 0], [-5, 0], [0, 5], [0, -5]]  
    tentative_action = [np.asarray([1,1,1,1,1]).reshape(1, action_space_size),np.asarray([1,0,0,0,0]).reshape(1, action_space_size),
                        np.asarray([1,0,0,0,0]).reshape(1, action_space_size),np.asarray([1,0,0,0,0]).reshape(1, action_space_size),
                        np.asarray([1,0,0,0,0]).reshape(1, action_space_size)]
    
    # global count_action_dif_default
    I = np.full((action_space_size, action_space_size), 0.5).reshape(1, action_space_size, action_space_size)
    idLightControl = '4628048104'
    waiting_time_t = 0
    numb_of_cycle = 0

    # new Agent.
    agent = DQNAgent.DQNAgent(memory_size, action_space_size, mini_batch_size)
    try:
        agent.load('Models/reinf_traf_control_v12_PER.h5')
    except:
        print('No models found')
    # agent.start_epsilon = 0
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
        action_time = [33,33]

        state, tentative_act_dec = sumo_int.getState(I, action, tentative_action)

        # break traning and save model.
        if numb_of_cycle > 30000:
            break
        
        # run a cycle.
        while traci.simulation.getMinExpectedNumber() > 0:
            
            # run a step on SUMO (~ 1 second).
            traci.simulationStep()

            waiting_time = 0

            # Get progress?
            agent.progress = agent.get_progress()
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
            yellow_time1 = sumo_int.cal_yellow_phase(['gneE21', 'gneE89'], a_dec)
            for j in range(yellow_time1):
                traci.trafficlight.setPhase(idLightControl, 1)
                # waiting_time += cal_waiting_time()
                traci.simulationStep()
            for j in range(action_time[1]):
                traci.trafficlight.setPhase(idLightControl, 2)
                # waiting_time += cal_waiting_time()
                traci.simulationStep()
            yellow_time2 = sumo_int.cal_yellow_phase(['gneE86', 'gneE85'], a_dec)
            for j in range(yellow_time2):
                traci.trafficlight.setPhase(idLightControl, 3)
                # waiting_time += cal_waiting_time()
                traci.simulationStep()
            #  ============================================================ Finish action ======================:

            # calculate REWARD
            waiting_time += cal_waiting_time()
            waiting_time_t1 = waiting_time
            reward_t = waiting_time_t - waiting_time_t1
            waiting_time_t = waiting_time_t1

            # get NewState by selected-action
            new_state, tentative_act_dec = sumo_int.getState(I, action, tentative_action)


            # Case 1: Experience Replay (store tuple) + store TD_error
            agent.store_tuple(state, action, reward_t, new_state, False)
            
            # Case 2: stored EXP/Tuple
            # agent.remember(state, action, reward_t, new_state, False)

            # reassign
            state = new_state
            numb_of_cycle += 1
            agent.step += 1

            print ('-------------------------step - ',numb_of_cycle, numb_of_cycle/300,'% - ', action_time, ' --------------------')
            
            if agent.progress == 'Training':
				# step 1: if agent.step % 100 == 0 then update weights of target_network.
				# ......... thinking ....................

				# step 2: get mini_batch?
                minibatch, w_batch, batch_index  = agent.prioritized_minibatch()

                # step 3: train.
                agent.replay(minibatch, w_batch, batch_index)

                # step 4: update epsilon:
                agent.start_epsilon -= agent.epsilon_decay

        agent.save('Models/reinf_traf_control_v14_PER.h5')
        traci.close(wait=False)

if __name__ == '__main__':
    main()
sys.stdout.flush()