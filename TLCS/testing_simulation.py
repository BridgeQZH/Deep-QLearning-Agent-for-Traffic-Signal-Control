import traci
import numpy as np
import timeit

from f_function_arrival_rate import f_function
from quadratic_reward_divided import g_function

# phase codes based on environment.net.xml
PHASE_NS_GREEN  = 0   # action 0
PHASE_NS_YELLOW = 1
PHASE_NSL_GREEN  = 2  # action 1
PHASE_NSL_YELLOW = 3
PHASE_EW_GREEN  = 4   # action 2
PHASE_EW_YELLOW = 5
PHASE_EWL_GREEN  = 6  # action 3
PHASE_EWL_YELLOW = 7


class Simulation:
    def __init__(self, Model, TrafficGen, sumo_cmd, gamma, max_steps,
                 green_duration, green_duration_straight, yellow_duration,
                 num_states, num_actions, action_mode):
        self._Model = Model
        self._TrafficGen = TrafficGen
        self._sumo_cmd = sumo_cmd
        self._gamma = gamma
        self._max_steps = max_steps
        self._green_duration = green_duration
        self._green_duration_straight = green_duration_straight
        self._yellow_duration = yellow_duration
        self._num_states = num_states
        self._num_actions = num_actions
        self._action_mode = action_mode
        self._cumulative_wait_store = []
        self._avg_queue_length_store = []


    def run(self, seed):
        """
        Run one test episode with the given seed. No training, no exploration.
        Returns (simulation_time, cumulative_wait, avg_queue_length).
        """
        start_time = timeit.default_timer()
        self._TrafficGen.generate_routefile(seed=seed)
        traci.start(self._sumo_cmd)

        self._step = 0
        self._waiting_times = {}
        self._sum_queue_length = 0
        self._sum_waiting_time = 0
        self._arrival_rate = {
            "N0": 0, "N1&N2": 0, "N3": 0,
            "S0": 0, "S1&S2": 0, "S3": 0,
            "E0": 0, "E1&E2": 0, "E3": 0,
            "W0": 0, "W1&W2": 0, "W3": 0,
        }
        self._lane_recorder = {k: [0] * 10 for k in self._arrival_rate}

        old_action = -1
        phase_index = 0

        while self._step < self._max_steps:
            current_state = self._get_state()
            self._collect_waiting_times()

            if self._action_mode == "traditional":
                action = self._choose_action(current_state)
            elif self._action_mode == "one-step":
                action = self._pick_a_control_rollout(self._last_counts, old_action)
            elif self._action_mode == "greedy":
                action = self._pick_a_control_greedy(self._last_counts, old_action)
            elif self._action_mode == "manual":
                action = phase_index % 4
                phase_index += 1
            else:
                raise ValueError(f"Unknown action_mode: {self._action_mode}")

            if self._step != 0 and old_action != action:
                self._set_yellow_phase(old_action)
                self._simulate(self._yellow_duration)

            self._set_green_phase(action)
            if action in (0, 2):
                self._simulate(self._green_duration_straight)
            else:
                self._simulate(self._green_duration)

            old_action = action

        self._cumulative_wait_store.append(self._sum_waiting_time)
        self._avg_queue_length_store.append(self._sum_queue_length / self._step)

        traci.close()
        sim_time = round(timeit.default_timer() - start_time, 1)
        return sim_time, self._sum_waiting_time, self._sum_queue_length / self._step


    def _simulate(self, steps_todo):
        if (self._step + steps_todo) >= self._max_steps:
            steps_todo = self._max_steps - self._step
        while steps_todo > 0:
            traci.simulationStep()
            self._step += 1
            steps_todo -= 1
            queue_length = self._get_queue_length()
            self._sum_queue_length += queue_length
            self._sum_waiting_time += queue_length


    def _collect_waiting_times(self):
        incoming_roads = ["E2TL", "N2TL", "W2TL", "S2TL"]
        for car_id in traci.vehicle.getIDList():
            wait_time = traci.vehicle.getAccumulatedWaitingTime(car_id)
            road_id = traci.vehicle.getRoadID(car_id)
            if road_id in incoming_roads:
                self._waiting_times[car_id] = wait_time
            else:
                if car_id in self._waiting_times:
                    del self._waiting_times[car_id]


    def _choose_action(self, state):
        """DQN greedy: argmax Q(state)."""
        return np.argmax(self._Model.predict_one(state))


    def _pick_a_control_rollout(self, current_state, old_action):
        """One-step lookahead: Q̃(u) = g(xₖ, u) + γ^T · max_a Q(f(xₖ,u), a).
        f_function returns 12-dim raw counts; DQN expects 24-dim normalized state.
        Normalize the predicted counts and pad zeros for the wait-time half.
        """
        if self._step == 0:
            return 0
        if old_action == -1:
            old_action = 2

        scores = []
        for action in range(4):
            g_val, past_time = g_function(current_state, action, old_action, self._gamma)
            next_counts = f_function(self._arrival_rate, current_state, action, old_action, self._green_duration, self._green_duration_straight, self._yellow_duration)
            next_counts_norm = np.clip(next_counts / 20.0, 0.0, 1.0)
            if self._num_states == 12:
                next_state_for_dqn = next_counts_norm
            else:
                current_waits_norm = np.clip(self._last_waits / self._max_steps, 0.0, 1.0)
                next_state_for_dqn = np.concatenate([next_counts_norm, current_waits_norm])
            q_next = self._Model.predict_one(next_state_for_dqn)
            q_tilde = g_val + self._gamma ** past_time * np.amax(q_next)
            scores.append(q_tilde)
        return int(np.argmax(scores))


    def _pick_a_control_greedy(self, current_state, old_action):
        """Pure greedy: argmax g(xₖ, u) — no DQN tail."""
        if self._step == 0:
            return 0
        if old_action == -1:
            old_action = 2
        scores = [g_function(current_state, u, old_action, self._gamma)[0] for u in range(4)]
        return int(np.argmax(scores))


    def _set_yellow_phase(self, old_action):
        traci.trafficlight.setPhase("TL", old_action * 2 + 1)


    def _set_green_phase(self, action):
        phase_map = {0: PHASE_NS_GREEN, 1: PHASE_NSL_GREEN,
                     2: PHASE_EW_GREEN, 3: PHASE_EWL_GREEN}
        traci.trafficlight.setPhase("TL", phase_map[action])


    def _get_queue_length(self):
        return (traci.edge.getLastStepHaltingNumber("N2TL") +
                traci.edge.getLastStepHaltingNumber("S2TL") +
                traci.edge.getLastStepHaltingNumber("E2TL") +
                traci.edge.getLastStepHaltingNumber("W2TL"))


    def _get_state(self):
        """24-dim state: [normalized counts x12 | normalized wait times x12]."""
        LANE_INDEX = {
            "N2TL_0": 0,  "N2TL_1": 1,  "N2TL_2": 1,  "N2TL_3": 2,
            "S2TL_0": 3,  "S2TL_1": 4,  "S2TL_2": 4,  "S2TL_3": 5,
            "E2TL_0": 6,  "E2TL_1": 7,  "E2TL_2": 7,  "E2TL_3": 8,
            "W2TL_0": 9,  "W2TL_1": 10, "W2TL_2": 10, "W2TL_3": 11,
        }
        RECORDER_KEY = {
            "N2TL_0": "N0",    "N2TL_1": "N1&N2", "N2TL_2": "N1&N2", "N2TL_3": "N3",
            "S2TL_0": "S0",    "S2TL_1": "S1&S2", "S2TL_2": "S1&S2", "S2TL_3": "S3",
            "E2TL_0": "E0",    "E2TL_1": "E1&E2", "E2TL_2": "E1&E2", "E2TL_3": "E3",
            "W2TL_0": "W0",    "W2TL_1": "W1&W2", "W2TL_2": "W1&W2", "W2TL_3": "W3",
        }
        MAX_CARS = 20.0

        counts = np.zeros(12)
        waits  = np.zeros(12)

        if self._step >= 100:
            slot = int((self._step % 100) / 10) % 10
            for key in self._lane_recorder:
                self._lane_recorder[key][slot] = 0

        for car_id in traci.vehicle.getIDList():
            lane_id  = traci.vehicle.getLaneID(car_id)
            if lane_id not in LANE_INDEX:
                continue
            idx      = LANE_INDEX[lane_id]
            lane_pos = 750 - traci.vehicle.getLanePosition(car_id)
            if lane_pos <= 200:
                counts[idx] += 1
                waits[idx]  += traci.vehicle.getAccumulatedWaitingTime(car_id)
            if lane_pos >= 636:
                self._lane_recorder[RECORDER_KEY[lane_id]][int(self._step / 10) % 10] += 1

        if self._step >= 100:
            for key in self._arrival_rate:
                self._arrival_rate[key] = sum(self._lane_recorder[key]) / 10.0

        self._last_counts = counts.copy()
        self._last_waits  = waits.copy()

        normalized_counts = np.clip(counts / MAX_CARS,         0.0, 1.0)
        if self._num_states == 12:
            return normalized_counts
        normalized_waits  = np.clip(waits  / self._max_steps,  0.0, 1.0)
        return np.concatenate([normalized_counts, normalized_waits])


    @property
    def cumulative_wait_store(self):
        return self._cumulative_wait_store

    @property
    def avg_queue_length_store(self):
        return self._avg_queue_length_store
