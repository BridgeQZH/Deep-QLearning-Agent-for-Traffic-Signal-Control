from __future__ import absolute_import
from __future__ import print_function

import os
from shutil import copyfile

# from testing_simulation_fixed import Simulation # To see how the normal way affect the result
from testing_simulation import Simulation
from generator import TrafficGenerator
from model import TestModel
from visualization import Visualization
from utils import import_test_configuration, set_sumo, set_test_path

# TODO: 1. Add testing env stochasity in to heterogeneous
# TODO: Provide NN hyperparameters choosing argument, when it cannot produce good enough results; Even though, having some arguments
# TODO: 2. New state representation ; New NN architecture. Solid step to push Rollout; System equation rather than transition probablity.
# Vector -> New Vector may be high dimension
# Compact state  - + 
# System equation easier, sampling time & duration time




if __name__ == "__main__":

    config = import_test_configuration(config_file='testing_settings.ini')
    sumo_cmd = set_sumo(config['gui'], config['sumocfg_file_name'], config['max_steps'])
    model_path, plot_path = set_test_path(config['models_path_name'], config['model_to_test'])

    Model = TestModel(
        input_dim=config['num_states'],
        model_path=model_path
    )
    
    TrafficGen = TrafficGenerator(
        config['max_steps'], 
        config['n_cars_generated']
    )

    Visualization = Visualization(
        plot_path, 
        dpi=96
    )
        
    Simulation = Simulation(
        Model,
        TrafficGen,
        sumo_cmd,
        config['n_cars_generated'],
        config['max_steps'],
        config['green_duration'],
        config['yellow_duration'],
        config['num_states'],
        config['num_actions']
    )

    print('\n----- Test episode -----')
    simulation_time = Simulation.run(config['episode_seed'])  # run the simulation
    print('Simulation time:', simulation_time, 's')
    

    print("----- Testing info saved at:", plot_path)

    copyfile(src='testing_settings.ini', dst=os.path.join(plot_path, 'testing_settings.ini')) # source, destination

    Visualization.save_data_and_plot(data=Simulation.reward_episode, filename='reward', xlabel='Action step', ylabel='Reward')
    Visualization.save_data_and_plot(data=Simulation.queue_length_episode, filename='queue', xlabel='Step', ylabel='Queue length (vehicles)')
    print("average queue length", sum(Simulation.queue_length_episode)/len(Simulation.queue_length_episode)) 
    
