from scripts.config import Config
from scripts.input_output import *
from scripts.simulated_annealing import simulated_annealing
from scripts.filters import *
import random
import importlib

config = Config()

"""
algorithms_list = ['simulated_annealing', 'cellular_automata', 'self_organizing_maps', 'genetic_algorithm']
algorithms_lever_list = [_config.SA_on, _config.CA_on, _config.SOM_on, _config.GA_on]

algorithm_to_call = [algorithms_list[i] for i in range(len(algorithms_list)) if algorithms_lever_list[i] == True]

if len(algorithm_to_call) != 1:
    print('Please, select ONE algorithm at the config.json file')
    exit()

script_to_use = importlib.import_module(f'scripts.{algorithm_to_call}')
algorithm_function = getattr(script_to_use, f'{algorithm_to_call}')
"""

random.seed(config.seed)
inputs = read_lidar(config)
start_grid, num_stands = create_grid(config, inputs)
dict_grid = create_dict_grid(config, start_grid)

if config.SA_iter_stats:
    final_grid, final_dict_grid, objective_function_evolution, stand_area_evolution, r_squareds_evolution \
        = simulated_annealing(config, start_grid, dict_grid)
    statistics_treatment(config, objective_function_evolution, stand_area_evolution, r_squareds_evolution)
else:
    final_grid, final_dict_grid = simulated_annealing(config, start_grid, dict_grid)
    # TODO: a function that creates a .txt or .csv file with the R^2, area and variances of each stand

filtered_grid = mode_filter(config, final_grid)
heatmap_grid(config, filtered_grid, 'stand')

# dict_grid = create_dict_grid_thorough(filtered_grid)
#
# split_grid, dict_grid = stand_splitter(filtered_grid, dict_grid)
#
# heatmap_grid(config, split_grid, 'stand')

#maps_to_gif(config)
