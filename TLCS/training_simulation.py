from numpy.testing._private.utils import HAS_LAPACK64
import traci
import numpy as np
import random
import timeit
import os
from f_function import f_function

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
        old_total_wait = 0
        old_state = -1
        old_action = -1
        
        while self._step < self._max_steps:

            # get current state of the intersection
            current_state = self._get_state()

            # calculate reward of previous action: (change in cumulative waiting time between actions)
            # waiting time = seconds waited by a car since the spawn in the environment, cumulated for every car in incoming lanes
            current_total_wait = self._collect_waiting_times()
            reward = old_total_wait - current_total_wait # Difference of accumulated total waiting time

            # saving the data into the memory
            if self._step != 0:
                self._Memory.add_sample((old_state, old_action, reward, current_state))

            # choose the light phase to activate, based on the current state of the intersection
            action = self._pick_a_control_rollout(current_state, current_total_wait, old_action)
            # action = self._choose_action(current_state, epsilon) # 0 1 2 3
            print("k = ", self._step)
            print("x_k:", current_state)
            print("u_k:", action)
            print("old_action:", old_action)
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
            old_total_wait = current_total_wait
            # saving only the meaningful reward to better see if the agent is behaving correctly
            if reward < 0:
                self._sum_neg_reward += reward

        
        self._save_episode_stats()
        print("Total reward:", self._sum_neg_reward, "- Epsilon:", round(epsilon, 2))
        traci.close()
        simulation_time = round(timeit.default_timer() - start_time, 1)

        print("Training...")
        start_time = timeit.default_timer()
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
            return np.argmax(self._Model.predict_one(state)) # the best action given the current state - exploitation

    def _pick_a_control_rollout(self, current_state, current_total_wait, old_action):
        """
        Get a control with one step look ahead
        """
        if self._step == 0:
            return 0
        # print("old_action", old_action)
        # if old_action == -1:
        #     old_action = 2
        a_list = []
        # x_k, u_1 evaluation
        action1 = 0
        action2 = 1
        action3 = 2
        action4 = 3

        next_state1 = f_function(self._step ,current_state, action1, old_action)
        print("next_state1:", next_state1)
        next_state2 = f_function(self._step ,current_state, action2, old_action)
        print("next_state2:", next_state2)
        next_state3 = f_function(self._step ,current_state, action3, old_action)
        print("next_state3:", next_state3)
        next_state4 = f_function(self._step ,current_state, action4, old_action)
        print("next_state4:", next_state4)
        
        if old_action != action1:
            self._set_yellow_phase(old_action)
            self._simulate(self._yellow_duration)
        
        self._set_green_phase(action1)
        self._simulate(self._green_duration)
        
        future_total_wait1 = self._collect_waiting_times()
        g1 = current_total_wait - future_total_wait1
        print("g1:", g1)

        if old_action != action2:
            self._set_yellow_phase(old_action)
            self._simulate(self._yellow_duration)
        
        self._set_green_phase(action2)
        self._simulate(self._green_duration)
        
        future_total_wait2 = self._collect_waiting_times()
        g2 = current_total_wait - future_total_wait2
        print("g2:", g2)

        if old_action != action3:
            self._set_yellow_phase(old_action)
            self._simulate(self._yellow_duration)
        
        self._set_green_phase(action3)
        self._simulate(self._green_duration)
        
        future_total_wait3 = self._collect_waiting_times()
        g3 = current_total_wait - future_total_wait3
        print("g3:", g3)

        if old_action != action4:
            self._set_yellow_phase(old_action)
            self._simulate(self._yellow_duration)
        
        self._set_green_phase(action4)
        self._simulate(self._green_duration)
        
        future_total_wait4 = self._collect_waiting_times()
        g4 = current_total_wait - future_total_wait4
        print("g4:", g4)
        
        q_s_a_d1 = self._Model.predict_one(next_state1)
        print("q_s_a_d1:", q_s_a_d1)
        H1 = np.amax(q_s_a_d1) # x+
        print(H1)
        q_tilde1 = g1 + self._gamma * H1
        print("q_tilde1:", q_tilde1)
        q_s_a_d2 = self._Model.predict_one(next_state2)
        print("q_s_a_d2:", q_s_a_d2)
        H2 = np.amax(q_s_a_d2)
        print(H2)
        q_tilde2 = g2 + self._gamma * H2
        print("q_tilde2:", q_tilde2)
        q_s_a_d3 = self._Model.predict_one(next_state3)
        H3 = np.amax(q_s_a_d3)
        q_tilde3 = g3 + self._gamma * H3
        print("q_tilde3:", q_tilde3)
        q_s_a_d4 = self._Model.predict_one(next_state4)
        H4 = np.amax(q_s_a_d4)
        q_tilde4 = g4 + self._gamma * H4
        
        print("q_tilde4:", q_tilde4)

        a_list.append(q_tilde1)
        a_list.append(q_tilde2)
        a_list.append(q_tilde3)
        a_list.append(q_tilde4)
        # four Q tilde, see which control wins
        print("a_list:", a_list)
        max_index = a_list.index(max(a_list))
        print("max_index", max_index)
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
        lane_north = 12
        lane_south = 13
        lane_east = 14
        lane_west = 15
        lane_group = None
        lane_north_flag = 0
        lane_south_flag = 0
        lane_east_flag = 0
        lane_west_flag = 0
        
        for car_id in car_list:
            lane_pos = traci.vehicle.getLanePosition(car_id)
            lane_pos = 750 - lane_pos  # inversion of lane pos, so if the car is close to the traffic light -> lane_pos = 0 --- 750 = max len of a road
            lane_id = traci.vehicle.getLaneID(car_id) # Returns the id of the lane the named vehicle was at within the last step.
            if lane_id == "N2TL_0":
                lane_group = 0
                state[lane_group] += 1
                if lane_pos <= 10:
                    lane_north_flag = 1
            elif lane_id == "N2TL_1" or lane_id == "N2TL_2":
                lane_group = 1
                state[lane_group] += 1
                if lane_pos <= 10:
                    lane_north_flag = 1
            elif lane_id == "N2TL_3":
                lane_group = 2
                state[lane_group] += 1
                if lane_pos <= 10:
                    lane_north_flag = 1
            elif lane_id == "S2TL_0":
                lane_group = 3
                state[lane_group] += 1
                if lane_pos <= 10:
                    lane_south_flag = 1
            elif lane_id == "S2TL_1" or lane_id == "S2TL_2":
                lane_group = 4
                state[lane_group] += 1
                if lane_pos <= 10:
                    lane_south_flag = 1
            elif lane_id == "S2TL_3":
                lane_group = 5
                state[lane_group] += 1
                if lane_pos <= 10:
                    lane_south_flag = 1
            elif lane_id == "E2TL_0":
                lane_group = 6
                state[lane_group] += 1
                if lane_pos <= 10:
                    lane_east_flag = 1
            elif lane_id == "E2TL_1" or lane_id == "E2TL_2":
                lane_group = 7
                state[lane_group] += 1
                if lane_pos <= 10:
                    lane_east_flag = 1
            elif lane_id == "E2TL_3":
                lane_group = 8
                state[lane_group] += 1
                if lane_pos <= 10:
                    lane_east_flag = 1
            elif lane_id == "W2TL_0":
                lane_group = 9
                state[lane_group] += 1
                if lane_pos <= 10:
                    lane_west_flag = 1
            elif lane_id == "W2TL_1" or lane_id == "W2TL_2":
                lane_group = 10
                state[lane_group] += 1
                if lane_pos <= 10:
                    lane_west_flag = 1
            elif lane_id == "W2TL_3":
                lane_group = 11
                state[lane_group] += 1
                if lane_pos <= 10:
                    lane_west_flag = 1
            
        state[lane_north] = 0
        state[lane_south] = 0
        state[lane_east] = 0
        state[lane_west] = 0
     
        if lane_north_flag == 1:
            state[lane_north] = 1
        if lane_south_flag == 1:
            state[lane_south] = 1
        if lane_east_flag == 1:
            state[lane_east] = 1
        if lane_west_flag == 1:
            state[lane_west] = 1
            
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

