from numpy.testing._private.utils import HAS_LAPACK64
import traci
import numpy as np
import matplotlib.pyplot as plt 
import random
import timeit
import os
from f_function_arrival_rate import f_function
from reward_by_hand import g_function
# from H_function import H_function
import difflib

# phase codes based on environment.net.xml
PHASE_NS_GREEN = 0  # action 0 code 00
PHASE_NS_YELLOW = 1
PHASE_NSL_GREEN = 2  # action 1 code 01
PHASE_NSL_YELLOW = 3
PHASE_EW_GREEN = 4  # action 2 code 10
PHASE_EW_YELLOW = 5
PHASE_EWL_GREEN = 6  # action 3 code 11
PHASE_EWL_YELLOW = 7


class Simulation:
    def __init__(self, Model, Memory, TrafficGen, sumo_cmd, gamma, max_steps, green_duration, yellow_duration, num_states, num_actions, training_epochs):
        self._Model = Model
        self._Memory = Memory
        self._TrafficGen = TrafficGen
        self._gamma = gamma
        self._step = 0
        self._sumo_cmd = sumo_cmd
        self._max_steps = max_steps
        self._green_duration = green_duration
        self._yellow_duration = yellow_duration
        self._num_states = num_states
        self._num_actions = num_actions
        self._reward_store = []
        self._cumulative_wait_store = []
        self._avg_queue_length_store = []
        self._training_epochs = training_epochs
        

    def run(self, episode, epsilon):
        """
        Runs an episode of simulation, then starts a training session
        """
        start_time = timeit.default_timer()

        # first, generate the route file for this simulation and set up sumo
        self._TrafficGen.generate_routefile(seed=episode)
        traci.start(self._sumo_cmd)
        print("Simulating...")

        # inits
        self._step = 0
        self._waiting_times = {}
        self._sum_neg_reward = 0
        self._sum_queue_length = 0
        self._sum_waiting_time = 0
        self._arrival_rate = {
            "N0":0,
            "N1&N2":0,
            "N3":0,
            "S0":0,
            "S1&S2":0,
            "S3":0,
            "E0":0,
            "E1&E2":0,
            "E3":0,
            "W0":0,
            "W1&W2":0,
            "W3":0
        }
        self._lane_recorder = {
            "N0":[0] * 10,
            "N1&N2":[0] * 10,
            "N3":[0] * 10,
            "S0":[0] * 10,
            "S1&S2":[0] * 10,
            "S3":[0] * 10,
            "E0":[0] * 10,
            "E1&E2":[0] * 10,
            "E3":[0] * 10,
            "W0":[0] * 10,
            "W1&W2":[0] * 10,
            "W3":[0] * 10
        }
        
        old_total_wait = 0
        old_state = -1
        old_action = -1
        actionflag = "one-step"
        similarity_list = []
        

        if actionflag == "traditional":
            print("You are using the traditional pick action method without rollout")
        if actionflag == "one-step":
            print("You are using the one-step lookahead pick action method with rollout")
        if actionflag == "multi-step":
            print("You are using the multi-step lookahead pick action method with rollout")
        if actionflag == "manual":
            print("The action is set manually")

        while self._step < self._max_steps:

            # get current state of the intersection
            current_state = self._get_state()
            # print("lane_recorder situation:", self._lane_recorder)
            # calculate reward of previous action: (change in cumulative waiting time between actions)
            # waiting time = seconds waited by a car since the spawn in the environment, cumulated for every car in incoming lanes
            # current_total_wait = self._collect_waiting_times()
            # reward = old_total_wait - current_total_wait # Difference of accumulated total waiting time
            

            # choose the light phase to activate, based on the current state of the intersection
            # action = self._pick_a_control_rollout(current_state, current_total_wait, old_action)
            # action = self._pick_a_control_rollout_four_step(current_state, current_total_wait, old_action)

            
            if actionflag == "traditional":
                action = self._choose_action(current_state, epsilon) # 0 1 2 3
            if actionflag == "one-step":
                action = self._pick_a_control_rollout(current_state, old_action)
            if actionflag == "multi-step":
                action = self._pick_a_control_rollout_four_step(current_state, old_action)
            if actionflag == "manual":
                if self._step <= 800:
                    action = 0
                else:
                    action = 1
            
            reward = g_function(current_state, action, old_action)
            # print("Current step and state and its reward:", self._step, current_state, reward)
            print("Current step and state:", self._step, current_state)
            
            # saving the data into the memory for training
            if actionflag == "traditional":
                if self._step != 0:
                    self._Memory.add_sample((old_state, old_action, reward, current_state))

            # You can use these lines to see the difference between real and estimated
            if self._step > 215:
                difference = [abs(x1 - x2) for (x1, x2) in zip(current_state, predict_state)]
                similarity = sum(difference) / len(difference)
                similarity_list.append(similarity)
                print("Average prediction error for each lane: ", format(similarity, '.3f'))
                # print(similarity_list)


            if action == 0:
                print("action is North and South Green")
            if action == 1:
                print("action is North and South Left Green")
            if action == 2:
                print("action is East and West Green")
            if action == 3:
                print("action is East and West Left Green")
            if self._step > 200:
                predict_state = f_function(self._arrival_rate, current_state, action, old_action)
                print("predict state based on f function (Compare with next true state):", predict_state)
               
            # print("x_k:", current_state)
            # print("u_k:", action)
            # print("old_action:", old_action)
            
            # If the chosen phase is different from the last phase, activate the yellow phase
            if self._step != 0 and old_action != action:
                self._set_yellow_phase(old_action)
                self._simulate(self._yellow_duration)

            # execute the phase selected before
            self._set_green_phase(action)
            self._simulate(self._green_duration)
            
            # saving variables for later & accumulate reward
            old_state = current_state
            old_action = action
            

            # old_total_wait = current_total_wait

            # saving only the meaningful reward to better see if the agent is behaving correctly
            if reward < 0:
                self._sum_neg_reward += reward

        
        self._save_episode_stats()
        print("Total reward:", self._sum_neg_reward, "- Epsilon:", round(epsilon, 2))
        x1=list(range(0,len(similarity_list)))
        y1=similarity_list
        
        plt.plot(x1,y1,'ro-')
        plt.title('The average prediction error for each lane')
        plt.xlabel('step')
        plt.ylabel('difference')
        plt.legend()
        plt.show()

        
        traci.close()
        simulation_time = round(timeit.default_timer() - start_time, 1)

        print("Training...")
        start_time = timeit.default_timer()
        if actionflag == "traditional":
            for _ in range(self._training_epochs):
                self._replay()
        training_time = round(timeit.default_timer() - start_time, 1)

        return simulation_time, training_time


    def _simulate(self, steps_todo):
        """
        Execute steps in sumo while gathering statistics
        """
        if (self._step + steps_todo) >= self._max_steps:  # do not do more steps than the maximum allowed number of steps
            steps_todo = self._max_steps - self._step

        while steps_todo > 0:
            traci.simulationStep()  # simulate 1 step in sumo
            self._step += 1 # update the step counter
            steps_todo -= 1
            queue_length = self._get_queue_length()
            self._sum_queue_length += queue_length
            self._sum_waiting_time += queue_length # 1 step while waiting in queue means 1 second waited, for each car, therefore queue_length == waited_seconds


    def _collect_waiting_times(self):
        """
        Retrieve the waiting time of every car in the incoming roads
        """
        incoming_roads = ["E2TL", "N2TL", "W2TL", "S2TL"]
        car_list = traci.vehicle.getIDList()
        for car_id in car_list:
            wait_time = traci.vehicle.getAccumulatedWaitingTime(car_id)
            road_id = traci.vehicle.getRoadID(car_id)  # get the road id where the car is located
            if road_id in incoming_roads:  # consider only the waiting times of cars in incoming roads
                self._waiting_times[car_id] = wait_time
            else:
                if car_id in self._waiting_times: # a car that was tracked has cleared the intersection
                    del self._waiting_times[car_id] 
        total_waiting_time = sum(self._waiting_times.values())
        return total_waiting_time


    def _choose_action(self, state, epsilon):
        """
        Decide wheter to perform an explorative or exploitative action, according to an epsilon-greedy policy
        """
        if random.random() < epsilon:
            return random.randint(0, self._num_actions - 1) # random action - exploration
        else:
            # print("argmax:", np.argmax(self._Model.predict_one(state)))
            # temp_state = [0,0,0,0]
            # print(np.argmax(self._Model.predict_one(state)))
            print(self._Model.predict_one(state))
            return np.argmax(self._Model.predict_one(state)) # the best action given the current state - exploitation

    def _pick_a_control_rollout(self, current_state, old_action): # one-step look ahead with Q factor approximation
        """
        Get a control with one step look ahead
        """
        if self._step == 0:
            return 0
        # Without force setting
        # if current_state[2]>=20 or current_state[5]>=20:
        #     return 1
        # elif current_state[8]>=20 or current_state[11]>=20:
        #     return 3
        print("old_action", old_action)
        if old_action == -1:
            old_action = 2
        a_list = []
        
        action1 = 0
        action2 = 1
        action3 = 2
        action4 = 3

        g1 = g_function(current_state, action1, old_action)
        g2 = g_function(current_state, action2, old_action)
        g3 = g_function(current_state, action3, old_action)
        g4 = g_function(current_state, action4, old_action)
        
        next_state_1 = f_function(self._arrival_rate, current_state, action1, old_action)
        print("If NS green, next_state:", next_state_1)
        q_s_a_d1 = self._Model.predict_one(next_state_1)
        H1 = np.amax(q_s_a_d1) # H_{k+1}(x_{k+1}^1)
        q_tilde1 = g1 + self._gamma * H1 # x_k, u_1 evaluation
        
        next_state_2 = f_function(self._arrival_rate, current_state, action2, old_action)
        print("If NS left green, next_state:", next_state_2)
        q_s_a_d2 = self._Model.predict_one(next_state_2)
        H2 = np.amax(q_s_a_d2) 
        q_tilde2 = g2 + self._gamma * H2

        next_state_3 = f_function(self._arrival_rate, current_state, action3, old_action)
        print("If EW green, next_state:", next_state_3)
        q_s_a_d3 = self._Model.predict_one(next_state_3)
        H3 = np.amax(q_s_a_d3) 
        q_tilde3 = g3 + self._gamma * H3
        
        next_state_4 = f_function(self._arrival_rate, current_state, action4, old_action)
        print("If NS left green, next_state:", next_state_4)        
        q_s_a_d4 = self._Model.predict_one(next_state_4)
        H4 = np.amax(q_s_a_d4) 
        q_tilde4 = g4 + self._gamma * H4
        
        a_list.append(q_tilde1)
        a_list.append(q_tilde2)
        a_list.append(q_tilde3)
        a_list.append(q_tilde4)
        # four Q tilde, see which control wins, and we pick that control
        max_index = a_list.index(max(a_list))
        return max_index

    def _pick_a_control_rollout_four_step(self, current_state, old_action): # four-step look ahead with Q factor approximation
        """
        Get a control with four step look ahead
        """
        if self._step == 0:
            return 0
        if old_action == -1:
            old_action = 2
        a_list = []
        
        action1 = 0
        action2 = 1
        action3 = 2
        action4 = 3

        ################################## For action1 ############################################
        g11 = g_function(current_state, action1, old_action) # g(x_k, u_1)
        x_k_plus_1_1 = f_function(self._arrival_rate, current_state, action1, old_action) # x_{k+1}^1
        u_k_plus_1_1_hat = np.argmax(self._Model.predict_one(x_k_plus_1_1))
        
        g12 = g_function(x_k_plus_1_1, u_k_plus_1_1_hat, action1) # g(x_k, u_1)
        x_k_plus_2_1 = f_function(self._arrival_rate, x_k_plus_1_1, u_k_plus_1_1_hat, action1) # x_{k+2}^1
        u_k_plus_2_1_hat = np.argmax(self._Model.predict_one(x_k_plus_2_1))

        g13 = g_function(x_k_plus_2_1, u_k_plus_2_1_hat, u_k_plus_1_1_hat)
        x_k_plus_3_1 = f_function(self._arrival_rate, x_k_plus_2_1, u_k_plus_2_1_hat, u_k_plus_1_1_hat) # x_{k+3}^1
        u_k_plus_3_1_hat = np.argmax(self._Model.predict_one(x_k_plus_3_1))

        g14 = g_function(x_k_plus_3_1, u_k_plus_3_1_hat, u_k_plus_2_1_hat)
        x_k_plus_4_1 = f_function(self._arrival_rate, x_k_plus_3_1, u_k_plus_3_1_hat, u_k_plus_2_1_hat) # x_{k+4}^1
        print("If current NS green, four steps later, the state will be:", x_k_plus_4_1)
        q_hat_1 = self._Model.predict_one(x_k_plus_4_1)
        H1 = np.amax(q_hat_1)
        q_tilde1 = g11 + self._gamma*g12 + self._gamma**2*g13 + self._gamma**3*g14 + self._gamma**4*H1 # evaulation of x_k u_1
        # print("q_tilde1", q_tilde1)

        ############################# For action2 ################################
        g21 = g_function(current_state, action2, old_action) # g(x_k, u_1)
        x_k_plus_1_2 = f_function(self._arrival_rate, current_state, action2, old_action) # x_{k+1}^1
        u_k_plus_1_2_hat = np.argmax(self._Model.predict_one(x_k_plus_1_2))

        g22 = g_function(x_k_plus_1_2, u_k_plus_1_2_hat, action2) # g(x_k, u_1)
        x_k_plus_2_2 = f_function(self._arrival_rate, x_k_plus_1_2, u_k_plus_1_2_hat, action2) # x_{k+2}^1
        u_k_plus_2_2_hat = np.argmax(self._Model.predict_one(x_k_plus_2_2))

        g23 = g_function(x_k_plus_2_2, u_k_plus_2_2_hat, u_k_plus_1_2_hat)
        x_k_plus_3_2 = f_function(self._arrival_rate, x_k_plus_2_2, u_k_plus_2_2_hat, u_k_plus_1_2_hat) # x_{k+3}^1
        u_k_plus_3_2_hat = np.argmax(self._Model.predict_one(x_k_plus_3_2))

        g24 = g_function(x_k_plus_3_2, u_k_plus_3_2_hat, u_k_plus_2_2_hat)
        x_k_plus_4_2 = f_function(self._arrival_rate, x_k_plus_3_2, u_k_plus_3_2_hat, u_k_plus_2_2_hat) # x_{k+3}^1
        print("If current NS Left green, four steps later, the state will be:", x_k_plus_4_2)
        q_hat_2 = self._Model.predict_one(x_k_plus_4_2)
        H2 = np.amax(q_hat_2)
        q_tilde2 = g21 + self._gamma*g22 + self._gamma**2*g23 + self._gamma**3*g24 + self._gamma**4*H2 # evaulation of x_k u_2
        # print("q_tilde2", q_tilde2)

        ############################# For action3 ################################
        g31 = g_function(current_state, action3, old_action) # g(x_k, u_1)
        x_k_plus_1_3 = f_function(self._arrival_rate, current_state, action3, old_action) # x_{k+1}^1
        u_k_plus_1_3_hat = np.argmax(self._Model.predict_one(x_k_plus_1_3))

        g32 = g_function(x_k_plus_1_3, u_k_plus_1_3_hat, action3) # g(x_k, u_1)
        x_k_plus_2_3 = f_function(self._arrival_rate, x_k_plus_1_3, u_k_plus_1_3_hat, action3) # x_{k+2}^1
        u_k_plus_2_3_hat = np.argmax(self._Model.predict_one(x_k_plus_2_3))

        g33 = g_function(x_k_plus_2_3, u_k_plus_2_3_hat, u_k_plus_1_3_hat)
        x_k_plus_3_3 = f_function(self._arrival_rate, x_k_plus_2_3, u_k_plus_2_3_hat, u_k_plus_1_3_hat) # x_{k+3}^1
        u_k_plus_3_3_hat = np.argmax(self._Model.predict_one(x_k_plus_3_3))

        g34 = g_function(x_k_plus_3_3, u_k_plus_3_3_hat, u_k_plus_2_3_hat)
        x_k_plus_4_3 = f_function(self._arrival_rate, x_k_plus_3_3, u_k_plus_3_3_hat, u_k_plus_2_3_hat) # x_{k+3}^1
        print("If current EW green, four steps later, the state will be:", x_k_plus_4_3)
        q_hat_3 = self._Model.predict_one(x_k_plus_4_3)
        H3 = np.amax(q_hat_3)
        q_tilde3 = g31 + self._gamma*g32 + self._gamma**2*g33 + self._gamma**3*g34 + self._gamma**4*H3 # evaluation of x_k u_3
        # print("q_tilde3", q_tilde3)

        ############################# For action4 ###############################
        g41 = g_function(current_state, action4, old_action) # g(x_k, u_1)
        x_k_plus_1_4 = f_function(self._arrival_rate, current_state, action4, old_action) # x_{k+1}^1
        u_k_plus_1_4_hat = np.argmax(self._Model.predict_one(x_k_plus_1_4))

        g42 = g_function(x_k_plus_1_4, u_k_plus_1_4_hat, action4) # g(x_k, u_1)
        x_k_plus_2_4 = f_function(self._arrival_rate, x_k_plus_1_4, u_k_plus_1_4_hat, action4) # x_{k+2}^1
        u_k_plus_2_4_hat = np.argmax(self._Model.predict_one(x_k_plus_2_4))

        g43 = g_function(x_k_plus_2_4, u_k_plus_2_4_hat, u_k_plus_1_4_hat)
        x_k_plus_3_4 = f_function(self._arrival_rate, x_k_plus_2_4, u_k_plus_2_4_hat, u_k_plus_1_4_hat) # x_{k+3}^1
        u_k_plus_3_4_hat = np.argmax(self._Model.predict_one(x_k_plus_3_4))

        g44 = g_function(x_k_plus_3_4, u_k_plus_3_4_hat, u_k_plus_2_4_hat)
        x_k_plus_4_4 = f_function(self._arrival_rate, x_k_plus_3_4, u_k_plus_3_4_hat, u_k_plus_2_4_hat) # x_{k+3}^1
        print("If current EW Left green, four steps later, the state will be:", x_k_plus_4_4)
        q_hat_4 = self._Model.predict_one(x_k_plus_4_4)
        H4 = np.amax(q_hat_4)
        q_tilde4 = g41 + self._gamma*g42 + self._gamma**2*g43 + self._gamma**3*g44 + self._gamma**4*H4 # Evaluation of x_k u_4
        # print("q_tilde4", q_tilde4)

        #################### Compare ###############################
        a_list.append(q_tilde1)
        a_list.append(q_tilde2)
        a_list.append(q_tilde3)
        a_list.append(q_tilde4)
        # four Q tilde, see which control wins
        max_index = a_list.index(max(a_list))
        return max_index

    def _set_yellow_phase(self, old_action):
        """
        Activate the correct yellow light combination in sumo
        """
        yellow_phase_code = old_action * 2 + 1 # obtain the yellow phase code, based on the old action (ref on environment.net.xml)
        traci.trafficlight.setPhase("TL", yellow_phase_code)


    def _set_green_phase(self, action_number):
        """
        Activate the correct green light combination in sumo
        """
        if action_number == 0:
            traci.trafficlight.setPhase("TL", PHASE_NS_GREEN)
        elif action_number == 1:
            traci.trafficlight.setPhase("TL", PHASE_NSL_GREEN)
        elif action_number == 2:
            traci.trafficlight.setPhase("TL", PHASE_EW_GREEN)
        elif action_number == 3:
            traci.trafficlight.setPhase("TL", PHASE_EWL_GREEN)


    def _get_queue_length(self):
        """
        Retrieve the number of cars with speed = 0 in every incoming lane
        """
        halt_N = traci.edge.getLastStepHaltingNumber("N2TL")
        halt_S = traci.edge.getLastStepHaltingNumber("S2TL")
        halt_E = traci.edge.getLastStepHaltingNumber("E2TL")
        halt_W = traci.edge.getLastStepHaltingNumber("W2TL")
        queue_length = halt_N + halt_S + halt_E + halt_W
        return queue_length


    def _get_state(self):
        """
        Retrieve the state of the intersection from sumo, in the form of cell occupancy
        """
        state = np.zeros(self._num_states) # Occupy matrix - 80; Number of vehicles - 16 or 12
        car_list = traci.vehicle.getIDList()
        if self._step >=100:
            a = self._step % 100
            self._lane_recorder["N0"][int(a/10)%10] = 0
            self._lane_recorder["N1&N2"][int(a/10)%10] = 0
            self._lane_recorder["N3"][int(a/10)%10] = 0
            self._lane_recorder["S0"][int(a/10)%10] = 0
            self._lane_recorder["S1&S2"][int(a/10)%10] = 0
            self._lane_recorder["S3"][int(a/10)%10] = 0
            self._lane_recorder["E0"][int(a/10)%10] = 0
            self._lane_recorder["E1&E2"][int(a/10)%10] = 0
            self._lane_recorder["E3"][int(a/10)%10] = 0
            self._lane_recorder["W0"][int(a/10)%10] = 0
            self._lane_recorder["W1&W2"][int(a/10)%10] = 0
            self._lane_recorder["W3"][int(a/10)%10] = 0

        for car_id in car_list:
            lane_pos = traci.vehicle.getLanePosition(car_id)
            lane_pos = 750 - lane_pos  # inversion of lane pos, so if the car is close to the traffic light -> lane_pos = 0 --- 750 = max len of a road
            # print("car_id and lane_pos pair:", car_id, lane_pos)
            # print(lane_pos)
            lane_id = traci.vehicle.getLaneID(car_id) # Returns the id of the lane the named vehicle was at within the last step.
            if lane_pos <= 200:
                if lane_id == "N2TL_0":
                    state[0] += 1
                elif lane_id == "N2TL_1" or lane_id == "N2TL_2":
                    state[1] += 1
                elif lane_id == "N2TL_3":
                    state[2] += 1
                elif lane_id == "S2TL_0":
                    state[3] += 1
                elif lane_id == "S2TL_1" or lane_id == "S2TL_2":
                    state[4] += 1
                elif lane_id == "S2TL_3":
                    state[5] += 1
                elif lane_id == "E2TL_0":
                    state[6] += 1
                elif lane_id == "E2TL_1" or lane_id == "E2TL_2":
                    state[7] += 1
                elif lane_id == "E2TL_3":
                    state[8] += 1
                elif lane_id == "W2TL_0":
                    state[9] += 1
                elif lane_id == "W2TL_1" or lane_id == "W2TL_2":
                    state[10] += 1
                elif lane_id == "W2TL_3":
                    state[11] += 1
            if lane_pos >= 636: # value tested from manual setting
                # Recorder starts to work
                # Only lane_pos >= 636 will be considered as new vehicle
                # Even start from 749, after 10s, it will reach 621, which is smaller than 636
                if lane_id == "N2TL_0":
                    self._lane_recorder["N0"][int(self._step/10)%10] += 1
                elif lane_id == "N2TL_1" or lane_id == "N2TL_2":
                    self._lane_recorder["N1&N2"][int(self._step/10)%10] += 1
                elif lane_id == "N2TL_3":
                    self._lane_recorder["N3"][int(self._step/10)%10] += 1
                elif lane_id == "S2TL_0":
                    self._lane_recorder["S0"][int(self._step/10)%10] += 1
                elif lane_id == "S2TL_1" or lane_id == "S2TL_2":
                    self._lane_recorder["S1&S2"][int(self._step/10)%10] += 1
                elif lane_id == "S2TL_3":
                    self._lane_recorder["S3"][int(self._step/10)%10] += 1
                elif lane_id == "E2TL_0":
                    self._lane_recorder["E0"][int(self._step/10)%10] += 1
                elif lane_id == "E2TL_1" or lane_id == "E2TL_2":
                    self._lane_recorder["E1&E2"][int(self._step/10)%10] += 1
                elif lane_id == "E2TL_3":
                    self._lane_recorder["E3"][int(self._step/10)%10] += 1
                elif lane_id == "W2TL_0":
                    self._lane_recorder["W0"][int(self._step/10)%10] += 1
                elif lane_id == "W2TL_1" or lane_id == "W2TL_2":
                    self._lane_recorder["W1&W2"][int(self._step/10)%10] += 1
                elif lane_id == "W2TL_3":
                    self._lane_recorder["W3"][int(self._step/10)%10] += 1

        if self._step >=100:
            self._arrival_rate["N0"] = sum(self._lane_recorder["N0"]) / len(self._lane_recorder["N0"])
            self._arrival_rate["N1&N2"] = sum(self._lane_recorder["N1&N2"]) / len(self._lane_recorder["N1&N2"])
            self._arrival_rate["N3"] = sum(self._lane_recorder["N3"]) / len(self._lane_recorder["N3"])
            self._arrival_rate["S0"] = sum(self._lane_recorder["S0"]) / len(self._lane_recorder["S0"])
            self._arrival_rate["S1&S2"] = sum(self._lane_recorder["S1&S2"]) / len(self._lane_recorder["S1&S2"])
            self._arrival_rate["S3"] = sum(self._lane_recorder["S3"]) / len(self._lane_recorder["S3"])
            self._arrival_rate["E0"] = sum(self._lane_recorder["E0"]) / len(self._lane_recorder["E0"])
            self._arrival_rate["E1&E2"] = sum(self._lane_recorder["E1&E2"]) / len(self._lane_recorder["E1&E2"])
            self._arrival_rate["E3"] = sum(self._lane_recorder["E3"]) / len(self._lane_recorder["E3"])
            self._arrival_rate["W0"] = sum(self._lane_recorder["W0"]) / len(self._lane_recorder["W0"])
            self._arrival_rate["W1&W2"] = sum(self._lane_recorder["W1&W2"]) / len(self._lane_recorder["W1&W2"])
            self._arrival_rate["W3"] = sum(self._lane_recorder["W3"]) / len(self._lane_recorder["W3"])

        return state

    


    def _replay(self):
        """
        Retrieve a group of samples from the memory and for each of them update the learning equation, then train
        """
        batch = self._Memory.get_samples(self._Model.batch_size)

        if len(batch) > 0:  # if the memory is full enough
            states = np.array([val[0] for val in batch])  # extract states from the batch
            next_states = np.array([val[3] for val in batch])  # extract next states from the batch

            # prediction
            q_s_a = self._Model.predict_batch(states)  # predict Q(state), for every sample
            q_s_a_d = self._Model.predict_batch(next_states)  # predict Q(next_state), for every sample

            # setup training arrays
            x = np.zeros((len(batch), self._num_states)) # input
            y = np.zeros((len(batch), self._num_actions)) # output

            for i, b in enumerate(batch):
                state, action, reward, _ = b[0], b[1], b[2], b[3]  # extract data from one sample
                current_q = q_s_a[i]  # get the Q(state) predicted before
                print("current_q", current_q)
                current_q[action] = reward + self._gamma * np.amax(q_s_a_d[i])  # update Q(state, action)
                x[i] = state
                y[i] = current_q  # Q(state) that includes the updated action value

            self._Model.train_batch(x, y)  # train the NN


    def _save_episode_stats(self):
        """
        Save the stats of the episode to plot the graphs at the end of the session
        """
        self._reward_store.append(self._sum_neg_reward)  # how much negative reward in this episode
        self._cumulative_wait_store.append(self._sum_waiting_time)  # total number of seconds waited by cars in this episode
        self._avg_queue_length_store.append(self._sum_queue_length / self._max_steps)  # average number of queued cars per step, in this episode


    @property
    def reward_store(self):
        return self._reward_store


    @property
    def cumulative_wait_store(self):
        return self._cumulative_wait_store


    @property
    def avg_queue_length_store(self):
        return self._avg_queue_length_store

