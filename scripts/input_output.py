import os

from scripts.config import Config
from scripts.structures import Cell
from pathlib import Path
import matplotlib.pyplot as plt
import math
import pandas as pd
import numpy as np


def read_lidar(config: Config) -> pd.DataFrame:
    """
    Function that reads all the specified csv files and returns a single pandas DataFrame

    :param config: instance of the Config class defined at config.py
    :return: pandas DataFrame with all the information contained in the .csv files
    """

    path_repo = str(Path(__file__).parent.parent.absolute())
    csv_file_paths = [config.path_LiDAR + f'/filtered_lidar_{i}.csv' for i in config.blocks]
    concat_df = pd.read_csv(csv_file_paths[0])
    for file in csv_file_paths[1:]:
        csv_df = pd.read_csv(file)
        concat_df = pd.concat([concat_df, csv_df], ignore_index = True)

    return concat_df


def create_grid(config: Config, df: pd.DataFrame) -> tuple:
    """
    Function that initializes the grid as a two-index array for the raster cells. Each raster cell of the LiDAR
    blocks correspond to a single element of the array. There may be more grid elements than raster cells, those
    grid elements that don't correspond to a cell will have all their values set to NA. This function fills the Cell
    data structure associated to each grid element with the values of the LiDAR metrics of its corresponding raster cell

    :param config: instance of the class Config
    :param df: pandas DataFrame that contains all the LiDAR metrics of all the considered blocks
    :return: grid as a two-index numpy array and the inital number of stands
    """
    unique_x = sorted(list(set(df['center.X'].tolist())))
    unique_y = sorted(list(set(df['center.Y'].tolist())))
    num_blocks_x = math.ceil(len(unique_x) / 100)
    num_blocks_y = math.ceil(len(unique_y) / 100)
    lowest_x = min(unique_x)
    lowest_y = min(unique_y)

    grid = np.empty((len(unique_x), len(unique_y)), dtype = object)

    for index, row in df.iterrows():
        cell_index_x = int((row['center.X'] - lowest_x) / 20)
        cell_index_y = int((row['center.Y'] - lowest_y) / 20)
        cell_data_list = list(row)[2:]
        if np.isnan(cell_data_list).any():
            cell_data_list = []
        grid[cell_index_x, cell_index_y] = Cell(0, 0, cell_data_list, row['center.X'], row['center.Y'],
                                                cell_index_x, cell_index_y)

    num_stands = config.stands_per_block_side ** 2 * num_blocks_x * num_blocks_y

    stand_side_x_list = []
    stand_side_y_list = []
    for i in range(int(config.stands_per_block_side * num_blocks_x)):
        stand_side = math.floor(grid.shape[0] / int(config.stands_per_block_side * num_blocks_x))
        if i < grid.shape[0] % int(config.stands_per_block_side * num_blocks_x):
            stand_side = stand_side + 1
        stand_side_x_list.append(stand_side)
    for j in range(int(config.stands_per_block_side * num_blocks_y)):
        stand_side = math.floor(grid.shape[1] / int(config.stands_per_block_side * num_blocks_y))
        if j < grid.shape[1] % int(config.stands_per_block_side * num_blocks_x):
            stand_side = stand_side + 1
        stand_side_y_list.append(stand_side)

    stand_count = 0
    for n in range(len(stand_side_x_list)):
        for m in range(len(stand_side_y_list)):
            for i in range(sum(stand_side_x_list[:n]), sum(stand_side_x_list[:(n+1)])):
                for j in range(sum(stand_side_y_list[:m]), sum(stand_side_y_list[:(m+1)])):
                    if len(grid[i, j].data):
                        grid[i, j].stand = stand_count
                        grid[i, j].stand_aux = stand_count
                    else:
                        grid[i, j].stand = float('nan')
                        grid[i, j].stand_aux = float('nan')
            stand_count += 1

    return grid, num_stands


def create_dict_grid(config, grid: np.ndarray) -> dict:
    num_blocks_x = math.ceil(grid.shape[0] / 100)
    num_blocks_y = math.ceil(grid.shape[1] / 100)
    num_stands = config.stands_per_block_side ** 2 * num_blocks_x * num_blocks_y

    dict_grid = {}
    for stand in range(num_stands):
        dict_stand = {}
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                if grid[i, j].stand == stand:
                    dict_stand[(i, j)] = [i, j] + grid[i, j].data
        dict_grid[stand] = dict_stand

    return dict_grid


def create_dict_grid_thorough(grid: np.ndarray) -> dict:
    stand_set = set()
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            stand_set.add(grid[i, j].stand)

    dict_grid = {}
    for stand in stand_set:
        dict_stand = {}
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                if grid[i, j].stand == stand:
                    dict_stand[(i, j)] = [i, j] + grid[i, j].data
        dict_grid[stand] = dict_stand

    return dict_grid


    num_blocks_x = math.ceil(grid.shape[0] / 100)
    num_blocks_y = math.ceil(grid.shape[1] / 100)
    num_stands = config.stands_per_block_side ** 2 * num_blocks_x * num_blocks_y

    dict_grid = {}
    for stand in range(num_stands):
        dict_stand = {}
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                if grid[i, j].stand == stand:
                    dict_stand[(i, j)] = [i, j] + grid[i, j].data
        dict_grid[stand] = dict_stand

    return dict_grid


def print_grid_stands(grid: np.ndarray) -> None:
    for row in grid:
        for obj in row:
            print("{:.2f}".format(getattr(obj, 'stand')), end=" ")
        print()


def print_grid_data(config: Config, grid: np.ndarray, metric: str) -> None:
    metric_list = config.used_metrics
    if metric in metric_list:
        metric_index = [index for index, value in enumerate(metric_list) if value == metric][0]
        for row in grid:
            for obj in row:
                cell_data_list = getattr(obj, 'data')
                print(type(cell_data_list[metric_index]), end=" ")
            print()
    else:
        print('Specify just one of the LiDAR metrics below:')
        print(f'{config.used_metrics}')
    return None


def heatmap_grid(config: Config, grid: np.ndarray, attribute_name: str) -> None:
    attribute_array = np.array([[getattr(obj, attribute_name) for obj in row] for row in grid])
    nan_mask = np.isnan(attribute_array)
    attribute_array[nan_mask] = np.nanmax(attribute_array) * 1.2
    plt.figure()
    plt.imshow(attribute_array, cmap='hot', interpolation='nearest')
    plt.colorbar(label=attribute_name)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Heatmap of ' + attribute_name)  # Add a title
    plt.savefig(f'{config.path_maps}/map.png')
    plt.close()

    return None


def heatmap_grid_gif(config: Config, grid: np.ndarray, attribute_name: str, iteration: int) -> None:
    attribute_array = np.array([[getattr(obj, attribute_name) for obj in row] for row in grid])
    nan_mask = np.isnan(attribute_array)
    attribute_array[nan_mask] = np.nanmax(attribute_array) * 1.2
    plt.figure()
    plt.imshow(attribute_array, cmap='hot', interpolation='nearest')
    plt.colorbar(label=attribute_name)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Heatmap of ' + attribute_name)  # Add a title
    plt.savefig(f'{config.path_maps_for_gif}/map_iter_{iteration}.png')
    plt.close()

    return None


def statistics_treatment(config: Config,
                         objective_function_evolution: list,
                         stand_area_evolution: list,
                         r_squareds_evolution: list) -> None:
    """
    Function that creates plots of the evolution of the objective function, stand area and r_squareds over the
    temperature.

    :param config: config class from the config.json object
    :param objective_function_evolution: there's one value per stand. The mean of the heuristic is computed as a
    pondered mean where the weights are the relative stand areas, doing so the value of the heuristic of the whole grid
    is obtained.
    :param stand_area_evolution: there's one value per stand
    :param r_squareds_evolution:
    :return:
    """
    total_iterations = int(np.ceil(np.log(config.SA_Tf / config.SA_Ti) / np.log(config.SA_C)))
    heuristic_vs_t_array = np.array(objective_function_evolution)
    area_vs_t_array = np.array(stand_area_evolution)
    r_squared_vs_t_list_arrays = []
    for v in range(len(config.SA_metric_weights)):
        r_squared_vs_t_list_arrays.append(np.array([[list_s[v] for list_s in list_t]
                                                    for list_t in r_squareds_evolution]))

    # heuristic evolution plot:
    means_heuristic = np.average(heuristic_vs_t_array, axis=1, weights=area_vs_t_array)  # ojo amb lo del axis...
    plt.figure()
    plt.plot(np.linspace(config.SA_Ti, config.SA_Tf, total_iterations),
             means_heuristic)
    plt.xlabel('Simulated Annealing Temperature')
    plt.ylabel('Average Objective Function Value of the Stands')
    plt.title('Average Objective Function Value over Temperature')
    plt.grid(True)
    plt.savefig(f'{config.path_plots}/average_heuristic_evolution.png')
    plt.close()

    # stand area evolution plot:
    means_area = np.mean(area_vs_t_array, axis=1)
    variances_area = np.var(area_vs_t_array, axis=1)
    plt.figure()
    plt.errorbar(x=np.linspace(config.SA_Ti, config.SA_Tf, total_iterations),
                 y=means_area,
                 yerr=variances_area,
                 fmt='o')
    plt.xlabel('Simulated Annealing Temperature')
    plt.ylabel('Mean Stand Area')
    plt.title('Mean Stand Area over Temperature')
    plt.grid(True)
    plt.savefig(f'{config.path_plots}/mean_area_evolution.png')
    plt.close()

    # r squared evolution plot:
    plt.figure()
    for i in range(len(config.SA_metric_weights)):
        means_r_squared = np.average(r_squared_vs_t_list_arrays[i], axis=1, weights=area_vs_t_array) # ojo amb lo del axis...
        plt.plot(np.linspace(config.SA_Ti, config.SA_Tf, total_iterations),
                 means_r_squared)
    plt.xlabel('Simulated Annealing Temperature')
    plt.ylabel('Average R$^2$')
    plt.title('Average R$^2$ of all variables over Temperature')
    plt.grid(True)
    plt.legend(config.used_metrics)
    plt.savefig(f'{config.path_plots}/average_r_squared_evolution.png')
    plt.plot()

    return None


def maps_to_gif(config: Config) -> None:
    filenames = os.listdir(config.path_maps_for_gif)
    images = [imageio.imread(filename) for filename in filenames]
    imageio.mimsave(f'{config.path_gifs}/grid_evolution.gif', images, duration=config.gif_duration)

    # TODO: also write the creation time at the title

    return None
