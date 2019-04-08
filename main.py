import sys
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import constants

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

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

def plot_durations(total_reward):
    plt.figure(2)
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    # plt.plot(episode_durations)
    # Take 100 episode averages and plot them too
    plt.plot(total_reward)

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())

def cal_waiting_time_average():
    number_vehicle = (traci.edge.getLastStepVehicleNumber('gneE21')+traci.edge.getLastStepVehicleNumber('gneE86')
      +traci.edge.getLastStepVehicleNumber('gneE89')+traci.edge.getLastStepVehicleNumber('gneE85'))

    if number_vehicle == 0:
        return 0
    return (traci.edge.getWaitingTime('gneE21') + traci.edge.getWaitingTime('gneE86') + traci.edge.getWaitingTime(
        'gneE89') + traci.edge.getWaitingTime('gneE85')) / number_vehicle  # waiting_time

def cal_waiting_time():
    return (traci.edge.getWaitingTime('gneE21') + traci.edge.getWaitingTime('gneE86') + traci.edge.getWaitingTime('gneE89')
            + traci.edge.getWaitingTime('gneE85')) # waiting_time

def main():
    # reward every episode
    total_reward = []
    # Control code here
    memory_size = constants.memory_size                   # size memory
    mini_batch_size = constants.mini_batch_size           # minibatch_size
    a_dec = constants.a_dec                               # m/s^2
    num_of_phase = constants.num_of_phase                 # 2 phase
    action_space_size = num_of_phase * 2 + 1                # 5 actions
    action_policy = constants.action_policy
    tentative_action = [np.asarray([1,1,1,1,1]).reshape(1, action_space_size),np.asarray([1,1,0,0,0]).reshape(1, action_space_size),
                        np.asarray([1,0,1,0,0]).reshape(1, action_space_size),np.asarray([1,0,0,1,0]).reshape(1, action_space_size),
                        np.asarray([1,0,0,0,1]).reshape(1, action_space_size)]

    # global count_action_dif_default
    I = np.full((action_space_size, action_space_size), 0.5).reshape(1, action_space_size, action_space_size)
    idLightControl = constants.idLightControl
    waiting_time_t = 0
    numb_of_cycle = 0

    # new Agent.
    agent = DQNAgent.DQNAgent(memory_size, action_space_size, mini_batch_size)
    try:
        agent.load('Models/reinf_traf_control_v14_loss_real_time.h5')
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
        waiting_time_plot = []
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
        while (traci.simulation.getMinExpectedNumber() > 0):
            
            # run a step on SUMO (~ 1 second).
            traci.simulationStep()

            waiting_time = 0

            # Get progress?
            agent.progress = agent.get_progress()
            action = agent.select_action_v2(state, tentative_act_dec)

            #  ============================================================ Perform action ======================
            for j in range(num_of_phase):
                action_time[j] += action_policy[action][j]
                if action_time[j] < 0:
                    action_time[j] = 0
                elif action_time[j] > 60:
                    action_time[j] = 60
            for j in range(action_time[0]):
                traci.trafficlight.setPhase(idLightControl, 0)
                traci.simulationStep()
                waiting_time_plot.append(cal_waiting_time_average())
            yellow_time1 = sumo_int.cal_yellow_phase(['gneE21', 'gneE89'], a_dec)
            for j in range(yellow_time1):
                traci.trafficlight.setPhase(idLightControl, 1)
                traci.simulationStep()
                waiting_time_plot.append(cal_waiting_time_average())
            for j in range(action_time[1]):
                traci.trafficlight.setPhase(idLightControl, 2)
                traci.simulationStep()
                waiting_time_plot.append(cal_waiting_time_average())
            yellow_time2 = sumo_int.cal_yellow_phase(['gneE86', 'gneE85'], a_dec)
            for j in range(yellow_time2):
                traci.trafficlight.setPhase(idLightControl, 3)
                traci.simulationStep()
                waiting_time_plot.append(cal_waiting_time_average())
            #  ============================================================ Finish action ======================:

            # calculate REWARD
            waiting_time += cal_waiting_time()
            waiting_time_t1 = waiting_time
            reward_t = waiting_time_t - waiting_time_t1
            waiting_time_t = waiting_time_t1

            # get NewState by selected-action
            new_state, tentative_act_dec = sumo_int.getState(I, action, tentative_action)


            # Case 1: Experience Replay (store tuple) + store TD_error
            # agent.store_tuple(state, action, reward_t, new_state, False)
            
            # Case 2: stored EXP/Tuple
            agent.remember(state, action, reward_t, new_state, False)

            # reassign
            state = new_state
            numb_of_cycle += 1
            agent.step += 1
            print ('-------------------------step - ',numb_of_cycle, numb_of_cycle/300,'% - ', action_time, ' --------------------')
            
            if agent.progress == 'Training':
				# step 1: if agent.step % 100 == 0 then update weights of target_network.
				# ......... thinking ....................

				# step 2: get mini_batch?
                # minibatch, w_batch, batch_index  = agent.get_prioritized_minibatch()

                # step 3: train.
                # agent.replay(minibatch, w_batch, batch_index)
                agent.replay_random_sample()

                # step 4: update epsilon:
                agent.start_epsilon -= agent.epsilon_decay

        total_reward.append(-np.mean(waiting_time_plot))
        plot_durations(total_reward)
        agent.save('Models/reinf_traf_control_v14_loss_real_time.h5')
        traci.close(wait=False)

    plt.ioff()
    plt.show()

if __name__ == '__main__':
    main()
sys.stdout.flush()