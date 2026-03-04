from __future__ import absolute_import
from __future__ import print_function

import os
from shutil import copyfile

from testing_simulation import Simulation
from generator import TrafficGenerator
from model import TestModel
from utils import import_test_configuration, set_sumo, set_test_path


if __name__ == "__main__":

    config = import_test_configuration(config_file='testing_settings.ini')
    sumo_cmd = set_sumo(config['gui'], config['sumocfg_file_name'], config['max_steps'])
    model_path, plot_path = set_test_path(
        config['models_path_name'], config['experiment_name'], config['model_to_test']
    )

    Model = TestModel(
        input_dim=config['num_states'],
        model_path=model_path
    )

    TrafficGen = TrafficGenerator(
        config['max_steps'],
        config['n_cars_generated']
    )

    Sim = Simulation(
        Model,
        TrafficGen,
        sumo_cmd,
        config['gamma'],
        config['max_steps'],
        config['green_duration'],
        config['yellow_duration'],
        config['num_states'],
        config['num_actions'],
        config['action_mode']
    )

    mode = config['action_mode']
    n_episodes = config['num_episodes']
    seed_start = config['episode_seed_start']

    print(f"\n===== Testing: mode={mode}  episodes={n_episodes}  seeds={seed_start}..{seed_start+n_episodes-1} =====")
    print(f"Model: {model_path}")
    print(f"{'Ep':>3}  {'Seed':>6}  {'CumWait(s)':>12}  {'AvgQueue':>10}  {'SimTime':>8}")
    print("-" * 50)

    for ep in range(n_episodes):
        seed = seed_start + ep
        sim_time, cum_wait, avg_queue = Sim.run(seed)
        print(f"{ep+1:3d}  {seed:6d}  {cum_wait:12.0f}  {avg_queue:10.2f}  {sim_time:7.1f}s")

    print("-" * 50)
    avg_cum_wait  = sum(Sim.cumulative_wait_store) / n_episodes
    avg_avg_queue = sum(Sim.avg_queue_length_store) / n_episodes
    print(f"{'AVG':>3}  {'':>6}  {avg_cum_wait:12.0f}  {avg_avg_queue:10.2f}")
    print(f"\nResults saved to: {plot_path}")

    copyfile(src='testing_settings.ini', dst=os.path.join(plot_path, 'testing_settings.ini'))
