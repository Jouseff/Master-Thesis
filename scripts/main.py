import numpy as np
import pandas as pd

from scripts.config import Config
from scripts.input_output import *
from scripts.simulated_annealing import simulated_annealing, statistics_sa
from scripts.filters import *
import random
from itertools import combinations

config = Config()

random.seed(config.seed)
inputs = read_lidar(config)
row_names = ['Elev.P20', 'Elev.stdde'] #, 'Num_eco1', 'Elev.MAD.m', 'PEco1_Mean', 'PEco1_Mode'
if config.SA_on:
    # statistics_df = pd.DataFrame(index=[f'{i}' for i in config.SA_term_weights], columns=config.SA_df_names)
    # statistics_df = pd.DataFrame(columns=[f'{i}' for i in config.SA_term_weights], index=row_names)
    statistics_df = pd.DataFrame(index=row_names)
    for i, weights_combination in enumerate([config.SA_term_weights[0]]):
        start_grid, num_stands = create_grid(config, inputs)
        dict_grid = create_dict_grid(config, start_grid)

        grid_sa, dict_grid_sa, decision_over_cells, objective_function_evolution = (
            simulated_annealing(config, start_grid, dict_grid, weights_combination))
        store_grid('raw', config, grid_sa, weights_combination)
        heatmap_grid(config, grid_stands(grid_sa), weights_combination, 'raw')
        statistics_df[f'{weights_combination}'] = statistics_sa(config, grid_sa, dict_grid_sa, weights_combination)
        # statistics_df.iloc[i, 0], statistics_df.iloc[i, 4], statistics_df.iloc[i, 8] = (
        #     statistics_sa(config, grid_sa, dict_grid_sa, weights_combination))
        plot_decision_over_cells(config, decision_over_cells)
        plot_objective_functions(config, objective_function_evolution, dict_grid_sa, weights_combination)

        cleaned_grid, cleaned_dict = mode_cleaner(config, grid_sa)
        store_grid('cleaned', config, cleaned_grid, weights_combination)
        heatmap_grid(config, grid_stands(cleaned_grid), weights_combination, 'cleaned')
        # statistics_df.iloc[i, 1], statistics_df.iloc[i, 5], statistics_df.iloc[i, 9] = (
        #     statistics_sa(config, cleaned_grid, cleaned_dict, weights_combination))

        split_grid, split_dict = relabel_connected_components(cleaned_grid)
        store_grid('split', config, split_grid, weights_combination)
        heatmap_grid(config, grid_stands(split_grid), weights_combination, 'split')
        # statistics_df.iloc[i, 2], statistics_df.iloc[i, 6], statistics_df.iloc[i, 10] = (
        #     statistics_sa(config, split_grid, split_dict, weights_combination))

        filtered_grid, filtered_dict = minimum_area_filter(config, split_grid, split_dict, weights_combination)
        store_grid('filtered', config, filtered_grid, weights_combination)
        heatmap_grid(config, grid_stands(filtered_grid), weights_combination, 'filtered')
        # statistics_df.iloc[i, 3], statistics_df.iloc[i, 7], statistics_df.iloc[i, 11] = (
        #     statistics_sa(config, filtered_grid, filtered_dict, weights_combination))

        heatmap_grid_data(config, filtered_grid, 0, weights_combination, row_names[0])
        heatmap_grid_data(config, filtered_grid, 1, weights_combination, row_names[1])
        # heatmap_grid_data(config, filtered_grid, 2, weights_combination, row_names[2])
        # heatmap_grid_data(config, filtered_grid, 3, weights_combination, row_names[3])
        # heatmap_grid_data(config, filtered_grid, 4, weights_combination, row_names[4])
        # heatmap_grid_data(config, filtered_grid, 5, weights_combination, row_names[5])


    statistics_df.to_csv(f'{config.path_dfs}/statistics_pondered_area_forest_metrics.csv', index=False)
