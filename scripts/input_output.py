import copy
import os
import datetime

from scripts.config import Config
from scripts.structures import Cell
import pandas as pd
import matplotlib.pyplot as plt
import math
import numpy as np
import glob


def read_lidar(config: Config) -> pd.DataFrame:
    """
    Function that reads all the specified csv files and returns a single pandas DataFrame

    :param config: instance of the Config class defined at config.py
    :return: pandas DataFrame with all the information contained in the .csv files
    """
    if config.use_pca:
        csv_file_paths = [config.path_LiDAR + f'/pca_block_{i}.csv' for i in config.blocks]
    else:
        csv_file_paths = [config.path_LiDAR + f'/filtered_lidar_{i}.csv' for i in config.blocks]

    concat_df = pd.read_csv(csv_file_paths[0])
    for file in csv_file_paths[1:]:
        csv_df = pd.read_csv(file)
        concat_df = pd.concat([concat_df, csv_df], ignore_index = True)
        concat_df = concat_df.drop(columns=['Return.3.c', 'Elev.IQ', 'Elev.P40', 'Int.P40', 'Total.firs', 'elevation'])

    return concat_df


def create_grid(config: Config, df: pd.DataFrame) -> tuple:
    """
    Function that initializes the grid as a two-index array for the raster cells. Each raster cell of the LiDAR
    blocks correspond to a single element of the array. There may be more grid elements than raster cells, those
    grid elements that don't correspond to a cell will have all their values set to NA. This function fills the Cell
    data structure associated to each grid element with the values of the LiDAR metrics of its corresponding raster cell

    :param config: instance of the class Config
    :param df: pandas DataFrame that contains all the LiDAR metrics of all the considered blocks
    :return: grid as a two-index numpy array and the initial number of stands
    """
    unique_x = sorted(list(set(df['center.X'].tolist())))[0:config.cells_per_side]
    unique_y = sorted(list(set(df['center.Y'].tolist())))[0:config.cells_per_side]
    num_blocks_x = math.ceil(len(unique_x) / config.cells_per_side)
    num_blocks_y = math.ceil(len(unique_y) / config.cells_per_side)
    lowest_x = min(unique_x)
    lowest_y = min(unique_y)

    big_grid = np.empty((len(unique_x), len(unique_y)), dtype=object)

    for index, row in df.iterrows():
        cell_index_x = int((row['center.X'] - lowest_x) / 20)
        cell_index_y = int((row['center.Y'] - lowest_y) / 20)
        cell_data_list = list(row)[:-8]
        if np.isnan(cell_data_list).any():
            cell_data_list = []
        big_grid[cell_index_x, cell_index_y] = Cell(0, cell_data_list, row['center.X'], row['center.Y'],
                                                    cell_index_x, cell_index_y)

    grid = big_grid[50:, :50]
    num_stands = config.stands_per_side ** 2 * num_blocks_x * num_blocks_y

    stand_side_x_list = []
    stand_side_y_list = []
    for i in range(int(config.stands_per_side * num_blocks_x)):
        stand_side = math.floor(grid.shape[0] / int(config.stands_per_side * num_blocks_x))
        if i < grid.shape[0] % int(config.stands_per_side * num_blocks_x):
            stand_side = stand_side + 1
        stand_side_x_list.append(stand_side)
    for j in range(int(config.stands_per_side * num_blocks_y)):
        stand_side = math.floor(grid.shape[1] / int(config.stands_per_side * num_blocks_y))
        if j < grid.shape[1] % int(config.stands_per_side * num_blocks_x):
            stand_side = stand_side + 1
        stand_side_y_list.append(stand_side)

    stand_count = 0
    for n in range(len(stand_side_x_list)):
        for m in range(len(stand_side_y_list)):
            for i in range(sum(stand_side_x_list[:n]), sum(stand_side_x_list[:(n+1)])):
                for j in range(sum(stand_side_y_list[:m]), sum(stand_side_y_list[:(m+1)])):
                    if len(grid[i, j].data):
                        grid[i, j].stand = stand_count
                    else:
                        grid[i, j].stand = float('nan')
            stand_count += 1

    return grid, num_stands


def create_dict_grid(config, grid: np.ndarray) -> dict:
    num_blocks_x = math.ceil(grid.shape[0] / config.cells_per_side)
    num_blocks_y = math.ceil(grid.shape[1] / config.cells_per_side)
    num_stands = config.stands_per_side ** 2 * num_blocks_x * num_blocks_y

    dict_grid = {}
    for stand in range(num_stands):
        dict_stand = {}
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                if grid[i, j].stand == stand:
                    dict_stand[(i, j)] = [i, j] + grid[i, j].data
        dict_grid[stand] = dict_stand

    return dict_grid


def grid_stands(grid: np.ndarray) -> np.ndarray:
    bare_grid = np.empty(grid.shape)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            bare_grid[i, j] = grid[i, j].stand
    return bare_grid


def create_dict_to_segment(grid: np.ndarray) -> dict:
    stands = set()
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            stands.add(grid[i, j])

    dict_grid = {}
    for stand in range(len(stands)):
        list_stand = []
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                if grid[i, j] == stand:
                    list_stand.append((i, j))
        dict_grid[stand] = list_stand

    return dict_grid


def fill_empty_cells(config: Config, data_grid: np.ndarray) -> np.ndarray:
    for i in range(data_grid.shape[0]):
        for j in range(data_grid.shape[1]):
            while len(data_grid[i, j].data) != len(config.SA_metric_weights) + 2:
                data_grid[i, j].data.append(0)
    return data_grid


def update_stands_grid(stand_grid: np.ndarray, data_grid: np.ndarray) -> np.ndarray:
    for i in range(stand_grid.shape[0]):
        for j in range(stand_grid.shape[1]):
            data_grid[i, j].stand = stand_grid[i, j]
    return data_grid


def del_dict(dictionary: dict, key_to_remove: tuple) -> dict:
    copy_dictionary = copy.deepcopy(dictionary)
    del copy_dictionary[key_to_remove]
    return copy_dictionary


def add_dict(dictionary: dict, key_to_add: tuple, value_to_add: list) -> dict:
    copy_dictionary = copy.deepcopy(dictionary)
    copy_dictionary[key_to_add] = value_to_add
    return copy_dictionary


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


def print_grid_stands(grid: np.ndarray) -> None:
    for row in grid:
        for obj in row:
            print("{:.2f}".format(getattr(obj, 'stand')), end=" ")
        print()


def store_grid(status: str, config: Config, data_grid: np.ndarray, weights: list) -> None:
    grid_to_print = grid_stands(data_grid)
    np.savetxt(f'{config.path_arrays}/grid_{status}_t0_{config.SA_Ti}_tf_{config.SA_Tf}_C_{config.SA_C}'
               f'_I_{config.SA_I}_w_{weights}.txt',
               grid_to_print, delimiter=',')
    return None


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


def heatmap_grid(config: Config, attribute_array: np.ndarray, weights: list, status: str) -> None:
    nan_mask = np.isnan(attribute_array)
    attribute_array[nan_mask] = np.nanmax(attribute_array) * 1.2
    plt.figure()
    plt.imshow(attribute_array, cmap='nipy_spectral', interpolation='nearest')
    plt.colorbar()
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Final segmentation')  # Add a title
    plt.savefig(f'{config.path_maps}/map_{status}_Ti_{config.SA_Ti}_Tf_{config.SA_Tf}_C_{config.SA_C}_'
                f'I_{config.SA_I}_w_{weights}.png')
    plt.close()

    return None


def heatmap_grid_gif(config: Config, grid: np.ndarray, attribute_name: str, iteration: int, weights: list) -> None:
    attribute_array = np.array([[getattr(obj, attribute_name) for obj in row] for row in grid])
    nan_mask = np.isnan(attribute_array)
    attribute_array[nan_mask] = np.nanmax(attribute_array) * 1.2
    plt.figure()
    plt.imshow(attribute_array, cmap='hot', interpolation='nearest')
    plt.colorbar(label=attribute_name)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Heatmap of ' + attribute_name)  # Add a title
    plt.savefig(f'{config.path_maps_for_gif}/map_iter_{iteration}_{weights}.png')
    plt.close()

    return None


def heatmap_grid_data(config: Config, grid: np.ndarray, index: int, weights_of: list, metric_name: str) -> None:
    attribute_array = np.empty(grid.shape)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if math.isnan(grid[i, j].stand) is False:
                attribute_array[i, j] = grid[i, j].data[index]
            else:
                attribute_array[i, j] = float('nan')

    stand_array = grid_stands(grid)

    horizontal_boundaries = (stand_array[:-1, :] != stand_array[1:, :])
    vertical_boundaries = (stand_array[:, :-1] != stand_array[:, 1:])

    nan_mask = np.isnan(attribute_array)
    attribute_array[nan_mask] = np.nanmax(attribute_array) * 1.2
    plt.figure()
    plt.pcolor(attribute_array, cmap='Greys')

    x_indices, y_indices = np.where(horizontal_boundaries)
    for x, y in zip(x_indices, y_indices):
        plt.plot([y, y + 1], [x + 1, x + 1], color='red', linewidth=1.5)

    x_indices, y_indices = np.where(vertical_boundaries)
    for x, y in zip(x_indices, y_indices):
        plt.plot([y + 1, y + 1], [x, x + 1], color='red', linewidth=1.5)

    plt.colorbar(label=f'{index}')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(f'Heatmap of metric {metric_name}')
    plt.savefig(f'{config.path_plots}/map_metric_{metric_name}_block_{config.blocks[0]}_w_{weights_of}.png')
    plt.close()

    return None


def plot_decision_over_cells(config: Config, decision_over_cells: dict) -> None:
    total_decisions = sum(list(decision_over_cells.values())[0])
    labels = ['NaN', 'Inner', 'Good accepted', 'Bad spared', 'Bad rejected']
    transposed_dict = {f'{labels[i]}': [decision_over_cells[key][i] for key in decision_over_cells]
                             for i in range(len(next(iter(decision_over_cells.values()))))}
    sparing_freq = [total_decisions * a / (a + b) for a, b in zip(transposed_dict['Bad spared'],
                                                                  transposed_dict['Bad rejected'])]
    acceptance_freq = [total_decisions * (a + b) / (a + b + c) for a, b, c in zip(transposed_dict['Good accepted'],
                                                                                  transposed_dict['Bad spared'],
                                                                                  transposed_dict['Bad rejected'])]
    bottom = np.zeros(len(transposed_dict['NaN']))
    fig, ax = plt.subplots()
    for decision, counts in transposed_dict.items():
        p = ax.bar(decision_over_cells.keys(), counts, width=0.6, label=decision, bottom=bottom)
        bottom += counts
        # ax.bar_label(p, label_type='center')
    x_values = np.linspace(1, len(sparing_freq), len(sparing_freq))
    ax.plot(x_values, sparing_freq)
    ax.plot(x_values, acceptance_freq)
    # ax.plot()
    ax.set_title('Number of each decision made at each T')
    ax.legend()
    plt.savefig(f'{config.path_plots}/decisions_C_{config.SA_C}_t0_{config.SA_Ti}'
                f'_tf_{config.SA_Tf}_I_{config.SA_I}.png')
    plt.close()

    return None


def plot_objective_functions(config: Config, list_of_dicts: list, dict_grid: dict, weights_of: list) -> None:
    transposed = []
    for i in range(len(list_of_dicts[0])):
        of_stand = []
        for dict_t in list_of_dicts:
            of_stand.append(dict_t[i])
        transposed.append(of_stand)

    areas = []
    for i in dict_grid.keys():
        areas.append(len(dict_grid[i]))

    average_of = round(np.dot(np.array(list(list_of_dicts[-1].values())), np.array(areas)) / sum(areas), 2)

    plt.figure()
    for i, nested_list in enumerate(transposed):
        plt.plot(np.linspace(1, len(nested_list), len(nested_list)), nested_list, label=f'{areas[i]}')
    plt.xlabel('Iterations')
    plt.ylabel('Value of the objective function')
    plt.title(f"Evolution of the stands' objective function. Average = {average_of}")
    plt.grid(True)
    plt.legend()
    plt.savefig(f'{config.path_plots}/OF_evolution_C_{config.SA_C}_t0_{config.SA_Ti}'
                f'_tf_{config.SA_Tf}_I_{config.SA_I}_w_{weights_of}.png')
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
    current_datetime = datetime.datetime.now()
    formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M:%S")

    filenames = os.listdir(config.path_maps_for_gif)
    print(filenames)
    print(config.path_maps_for_gif)
    images = [imageio.imread(filename) for filename in filenames]
    imageio.mimsave(f'{config.path_gifs}/grid_evolution_({formatted_datetime}).gif', images,
                    duration=config.gif_duration)

    return None


def final_statistics_csv(config: Config) -> None:
    filename = f'{config.path_dfs}/final_R2.csv'
    # TODO: end this function!!!
    return None

# we have to run this function before and after the filtering!
