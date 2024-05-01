import numpy as np
import random
import math

from scripts.config import Config
from scripts.input_output import heatmap_grid_gif


def centroid(xs: list, ys: list) -> [float, float]:
    """
    Function that calculates the center of a stand given the x-y coordinates of all its member cells

    :param xs: list of x coordinates of the member cells of the stand
    :param ys: list of y coordinates of the member cells of the stand
    :return: center of the considered stand
    """
    return [np.mean(xs), np.mean(ys)]


def stand_area(config: Config, num_cells: int) -> float:
    """
    Function that computes the area term of the SA metaheuristic

    :param config: config class from the config.json object
    :param num_cells: the number of cells that belong to the stand
    :return: the numeric value of the area term of the SA metaheuristic
    """
    weight_area = config.SA_term_weights[0]
    result = weight_area / (1+np.exp(-2 * (num_cells * config.area_cell - 1)))
    return result


def stand_variance(config: Config, data_list: list) -> float:
    """
    Function that computes the variance term of the SA metaheuristic

    :param config: config class from the config.json object
    :param data_list: list of lists, where each element list contains all the metrics for a given cell of the stand
    :return: the numeric value of the variance term of the SA metaheuristic
    """
    weight_variance = config.SA_term_weights[1]
    relative_variance = sum(config.SA_metric_weights[i] * np.var(data_list[i]) /
                            np.mean(data_list[i]) for i in range(len(config.SA_metric_weights)))
    result = weight_variance / (1 + np.exp(3 * (relative_variance - 0.3)))
    return result


def stand_shape(config: Config, centroid_coords: list, num_cells: int, x_coords: list, y_coords: list) -> float:
    """
    Function that computes the shape term of the SA metaheuristic

    :param config: config class from the config.json object
    :param centroid_coords: coordinates of the center of the stand
    :param num_cells: number of cells that the stand has
    :param x_coords: list of the x coordinates of all the cells of the stands
    :param y_coords: list of the y coordinates of all the cells of the stands
    :return: the numeric value of the shape term of the SA metaheuristic
    """
    weight_shape = config.SA_term_weights[2]
    result = weight_shape / num_cells * sum(1 / (1 + np.exp(8 * (np.linalg.norm(np.array([x_coords[i], y_coords[i]])
                                                                           - np.array(centroid_coords)) /
                                                            np.sqrt(num_cells * config.area_cell / np.pi) - 1)))
                                            for i in range(num_cells))
    return result


def stand_objective_function(config: Config, dict_stand: dict) -> float:
    """
    Function that computes the objective function (a.k.a. the metaheuristic) for a given stand

    :param config: config class from the config.json object
    :param dict_stand: dictionary of a stand that collects all the information of its cells
    :return: the numeric value of the SA metaheuristic for the given stand
    """
    if len(dict_stand):
        transposed_dict_stand = {i: [dict_stand[key][i] for key in dict_stand]
                                 for i in range(len(next(iter(dict_stand.values()))))}

        x_coords_stand = transposed_dict_stand[0]
        y_coords_stand = transposed_dict_stand[1]
        stand_data = [transposed_dict_stand[i+2] for i in range(len(config.SA_metric_weights))]

        stand_centroid = centroid(x_coords_stand, y_coords_stand)

        objective_function = stand_area(config, len(dict_stand)) + \
                             stand_variance(config, stand_data) + \
                             stand_shape(config, stand_centroid, len(dict_stand), x_coords_stand, y_coords_stand)
        return objective_function
    else:
        return 0


def statistics_for_temperature(config: Config,
                               grid: np.ndarray,
                               dict_grid: dict,
                               num_stands: int) -> tuple:
    """
    Function that returns the value of R^2, area, and metaheuristic for each stand. This function is crafted to be used
    at the end of all the iterations for each temperature vale.

    :param config: config class from the config.json object
    :param grid: numpy array representing the terrain that we want to segment
    :param dict_grid: dictionary of dictionaries of the grid, one per stand, that collects the info of all the cells
    :param num_stands:
    :return: the numeric value of the SA metaheuristic for the given stand
    """
    all_objective_functions = []
    all_stand_areas = []
    total_variances = []
    all_r_squareds = []

    for stand_index in range(num_stands):
        if len(dict_grid[stand_index]):
            all_stand_areas.append(len(dict_grid[stand_index]))
            all_objective_functions.append(stand_objective_function(config, dict_grid[stand_index]))
        else:
            all_stand_areas.append(0)
            all_objective_functions.append(0)

    for m in range(len(config.SA_metric_weights)):
        total_variances.append(np.var([grid[i, j].data[m] for i in range(grid.shape[0]) for j in range(grid.shape[1])
                                       if math.isnan(grid[i, j].stand) is False]))

    for stand_index in range(num_stands):
        stand_r_squareds_list = []
        len_dict_stand = len(dict_grid[stand_index])
        for v in range(len(config.SA_metric_weights)):
            if len_dict_stand > 1:
                stand_r_squareds_list.append(1 - np.var([cell_list[v + 2] for cell_list
                                                        in dict_grid[stand_index].values()]) / total_variances[v])
            else:
                stand_r_squareds_list.append(0)  # I know that this is blasphemy if the stand has 1 cell
        all_r_squareds.append(stand_r_squareds_list)

    return all_objective_functions, all_stand_areas, all_r_squareds


def simulated_annealing(config: Config,
                        grid: np.ndarray,
                        dict_grid: dict) -> tuple:
    """
    Function that performs the Simulated Annealing algorithm

    :param config: config class from the config.json object
    :param grid: numpy array representing the terrain that we want to segment
    :param dict_grid: dictionary that contains all the information about the grid
    :return:
    """
    num_stands = len(dict_grid)
    objective_function_evolution = []
    stand_area_evolution = []
    r_squareds_evolution = []

    t_iter = 1
    total_iterations = int(np.ceil(np.log(config.SA_Tf / config.SA_Ti) / np.log(config.SA_C)))
    t = config.SA_Ti
    while t > config.SA_Tf:
        print(f'Iteration {t_iter} out of {total_iterations}. Current temperature is {"{:.4f}".format(t)}. '
              f'Final temperature will be {config.SA_Tf}')
        for iteration in range(config.SA_I):
            # it should be 1e4 according to the paper, but I'm positive that this has to do with the number of data
            # points on the grid! i.e. iterations is prop to grid.shape[0] * grid.shape[1]
            random_x = random.randint(0, grid.shape[0] - 1)
            random_y = random.randint(0, grid.shape[1] - 1)

            if math.isnan(grid[random_x, random_y].stand) is False:
                different_stand = []
                current_objective_function = stand_objective_function(config, dict_grid[grid[random_x, random_y].stand])
                for p in [-1, 0, 1]:
                    for q in [-1, 0, 1]:
                        if (random_x + p) < 0 or (random_x + p) >= grid.shape[0] \
                                              or (random_y + q) < 0 \
                                              or (random_y + q) >= grid.shape[1]:
                            continue
                        else:
                            other_stand = grid[random_x + p, random_y + q].stand
                            if other_stand != grid[random_x, random_y].stand and math.isnan(other_stand) is False:
                                different_stand.append(grid[random_x + p, random_y + q].stand)
                if len(different_stand) == 0:
                    continue

                grid[random_x, random_y].stand = random.choice(different_stand)
                candidate_objective_function = (stand_objective_function(config, dict_grid[grid[random_x, random_y].stand]) +
                                                stand_objective_function(config, dict_grid[grid[random_x, random_y].stand_aux])) / 2

                if candidate_objective_function > current_objective_function:
                    dict_grid[grid[random_x, random_y].stand][(random_x, random_y)] = \
                        dict_grid[grid[random_x, random_y].stand_aux][(random_x, random_y)]
                    del dict_grid[grid[random_x, random_y].stand_aux][(random_x, random_y)]
                    grid[random_x, random_y].stand_aux = grid[random_x, random_y].stand
                else:
                    probability = random.uniform(0, 1)
                    if probability < np.exp((candidate_objective_function - current_objective_function) / t):
                        dict_grid[grid[random_x, random_y].stand][(random_x, random_y)] = \
                            dict_grid[grid[random_x, random_y].stand_aux][(random_x, random_y)]
                        del dict_grid[grid[random_x, random_y].stand_aux][(random_x, random_y)]
                        grid[random_x, random_y].stand_aux = grid[random_x, random_y].stand
                    else:
                        grid[random_x, random_y].stand = grid[random_x, random_y].stand_aux

        if config.SA_iter_stats:
            statistics_temperature = statistics_for_temperature(config, grid, dict_grid, num_stands)
            objective_function_evolution.append(statistics_temperature[0])
            stand_area_evolution.append(statistics_temperature[1])
            r_squareds_evolution.append(statistics_temperature[2])

        heatmap_grid_gif(config, grid, 'stand', t_iter)
        t = config.SA_C * t
        t_iter += 1

    if config.SA_iter_stats:
        return grid, dict_grid, objective_function_evolution, stand_area_evolution, r_squareds_evolution
    else:
        return grid, dict_grid
