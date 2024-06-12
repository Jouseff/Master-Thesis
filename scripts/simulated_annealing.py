import numpy as np
import random
import math

from scripts.config import Config
from scripts.input_output import heatmap_grid_gif, store_grid, del_dict, add_dict


def centroid(xs: list, ys: list) -> [float, float]:
    """
    Function that calculates the center of a stand given the x-y coordinates of all its member cells

    :param xs: list of x coordinates of the member cells of the stand
    :param ys: list of y coordinates of the member cells of the stand
    :return: center of the considered stand
    """
    return [np.mean(xs), np.mean(ys)]


def stand_area(config: Config, num_cells: int, weights_of: list) -> float:
    """
    Function that computes the area term of the SA metaheuristic

    :param weights_of:
    :param config: config class from the config.json object
    :param num_cells: the number of cells that belong to the stand
    :return: the numeric value of the area term of the SA metaheuristic
    """
    initial_area = math.ceil((config.cells_per_side / config.stands_per_side) ** 2) * config.cell_area
    area_of_stand = num_cells * config.cell_area
    weight_area = weights_of[0]
    result = weight_area / (1 + np.exp(-2.634 * (area_of_stand / initial_area - 1)))
    return result


def stand_variance(config: Config, data_list: list, weights_of: list) -> float:
    """
    Function that computes the variance term of the SA metaheuristic

    :param weights_of:
    :param config: config class from the config.json object
    :param data_list: list of lists, where each element list contains all the metrics for a given cell of the stand
    :return: the numeric value of the variance term of the SA metaheuristic
    """
    normalization = sum(config.SA_metric_weights) * 100
    relative_variance = sum(config.SA_metric_weights[i] * np.var(data_list[i]) /
                            (np.mean(data_list[i]) * normalization) for i in range(len(config.SA_metric_weights)))
    result = weights_of[1] / (1 + np.exp(3 * (abs(relative_variance) - 0.3)))
    # print(relative_variance, result)
    return result


def stand_shape(config: Config, centroid_coords: list, num_cells: int, x_coords: list, y_coords: list, weights_of: list) -> float:
    """
    Function that computes the shape term of the SA metaheuristic

    :param config: config class from the config.json object
    :param centroid_coords: coordinates of the center of the stand
    :param num_cells: number of cells that the stand has
    :param x_coords: list of the x coordinates of all the cells of the stands
    :param y_coords: list of the y coordinates of all the cells of the stands
    :return: the numeric value of the shape term of the SA metaheuristic
    """
    weight_shape = weights_of[2]
    result = weight_shape / num_cells * sum(1 / (1 + np.exp(8 * (np.linalg.norm(np.array([x_coords[i], y_coords[i]])
                                                                                - np.array(centroid_coords)) /
                                                                 np.sqrt(num_cells * config.cell_area / np.pi) - 1)))
                                            for i in range(num_cells))
    return result


def stand_fract(dict_stand: dict, weights_of: list):
    weight_fract = weights_of[2]
    cell_coords = dict_stand.keys()
    area = len(cell_coords)
    perimeter = 0
    for cell in cell_coords:
        perimeter += sum([(cell[0] - 1, cell[1]) not in cell_coords,
                          (cell[0] + 1, cell[1]) not in cell_coords,
                          (cell[0], cell[1] - 1) not in cell_coords,
                          (cell[0], cell[1] + 1) not in cell_coords])
    fractal_dimension = 2 * np.log(perimeter / 4) / np.log(area)
    result = weight_fract / (1 + np.exp(10 * (fractal_dimension - 1.5)))
    return result


def stand_objective_function(config: Config, dict_stand: dict, weights_of: list) -> float:
    """
    Function that computes the objective function (a.k.a. the metaheuristic) for a given stand

    :param weights_of:
    :param config: config class from the config.json object
    :param dict_stand: dictionary of a stand that collects all the information of its cells
    :return: the numeric value of the SA metaheuristic for the given stand
    """
    if len(dict_stand) > 1:
        length = len(next(iter(dict_stand.values())))
        transposed_dict_stand = {i: [value[i] for value in dict_stand.values()] for i in range(length)}

        # x_coords_stand = transposed_dict_stand[0]
        # y_coords_stand = transposed_dict_stand[1]
        stand_data = [transposed_dict_stand[i+2] for i in range(len(config.SA_metric_weights))]

        # stand_centroid = centroid(x_coords_stand, y_coords_stand)

        objective_function = stand_area(config, len(dict_stand), weights_of) + \
                             stand_variance(config, stand_data, weights_of) + \
                             stand_fract(dict_stand, weights_of)
        return objective_function
    elif len(dict_stand) == 1:
        initial_area = math.ceil((config.cells_per_side / config.stands_per_side) ** 2) * config.cell_area
        term_area = weights_of[0] / (1 + np.exp(-2.634 * (config.cell_area / initial_area - 1)))
        term_fractal = weights_of[2] / (1 + np.exp(10 * (1 - 1.5)))
        return term_area + term_fractal
    else:
        return 0


def statistics_sa(config: Config,
                  grid: np.ndarray,
                  dict_grid: dict,
                  weights_of: list) -> tuple:
    """
    Function that returns the value of R^2, area, and metaheuristic for each stand. This function is crafted to be used
    at the end of all the iterations for each temperature vale.

    :param weights_of:
    :param config: config class from the config.json object
    :param grid: numpy array representing the terrain that we want to segment
    :param dict_grid: dictionary of dictionaries of the grid, one per stand, that collects the info of all the cells
    :return: the numeric value of the SA metaheuristic for the given stand
    """
    all_objective_functions = []
    all_stand_areas = []
    total_variances = []
    all_r_squareds = []

    for stand_index, dict_stand in dict_grid.items():
        if len(dict_stand):
            all_stand_areas.append(len(dict_stand))
            all_objective_functions.append(stand_objective_function(config, dict_stand, weights_of))
        else:
            all_stand_areas.append(0)
            all_objective_functions.append(0)

    for m in range(len(config.SA_metric_weights)):
        total_variances.append(np.var([grid[i, j].data[m] for i in range(grid.shape[0]) for j in range(grid.shape[1])
                                       if math.isnan(grid[i, j].stand) is False]))

    for stand_index in dict_grid.keys():
        stand_r_squareds_list = []
        len_dict_stand = len(dict_grid[stand_index])
        for v in range(len(config.SA_metric_weights)):
            if len_dict_stand > 1:
                stand_r_squareds_list.append(1 - np.var([cell_list[v + 2] for cell_list
                                                        in dict_grid[stand_index].values()]) / total_variances[v])
            else:
                stand_r_squareds_list.append(0)  # I know that this is blasphemy... sorry :_)
        all_r_squareds.append(stand_r_squareds_list)

    area_mean = round(np.mean(all_stand_areas), 3)
    of_average = sum(np.array(all_objective_functions) * np.array(all_stand_areas)) / sum(all_stand_areas)
    r_sq_list = np.apply_along_axis(lambda col: sum(col * np.array(all_stand_areas)) / sum(all_stand_areas), axis=0,
                                    arr=np.array(all_r_squareds)).tolist()
    return r_sq_list # area_mean, of_average,


def simulated_annealing(config: Config,
                        grid: np.ndarray,
                        dict_grid: dict,
                        weights_of: list) -> tuple:
    """
    Function that performs the Simulated Annealing algorithm

    :param weights_of: weights for each of the terms of the objective function (in short, o.f.)
    :param config: config class from the config.json object
    :param grid: numpy array representing the terrain that we want to segment
    :param dict_grid: dictionary that contains all the information about the grid
    :return:
    """
    objective_function_evolution = []
    decision_over_cells = {}

    t_iter = 1
    total_iterations = int(np.ceil(np.log(config.SA_Tf / config.SA_Ti) / np.log(config.SA_C)))
    t = config.SA_Ti
    while t > config.SA_Tf:
        print(f'Iteration {t_iter} out of {total_iterations}. Current temperature is {"{:.4f}".format(t)}. '
              f'Final temperature will be {config.SA_Tf}. The weights of the OF are: {weights_of}')
        num_spares = 0
        num_bingos = 0
        num_nonnan = 0
        num_inners = 0
        num_wrongs = 0
        for iteration in range(config.SA_I):
            random_x = random.randint(0, grid.shape[0] - 1)
            random_y = random.randint(0, grid.shape[1] - 1)

            if math.isnan(grid[random_x, random_y].stand) is False:
                num_nonnan += 1
                different_stand = []
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
                    num_inners += 1
                    continue

                current_stand = grid[random_x, random_y].stand
                candidate_stand = random.choice(different_stand)

                if config.SA_ponder_of_by_area:
                    current_area = len(dict_grid[current_stand])
                    candidate_area = len(dict_grid[candidate_stand])
                    current_objective_function = ((current_area * stand_objective_function(
                                                  config, dict_grid[current_stand], weights_of) +
                                                  candidate_area * stand_objective_function(
                                                  config, dict_grid[candidate_stand], weights_of)) /
                                                  (current_area + candidate_area))
                else:
                    current_objective_function = (stand_objective_function(
                                                  config, dict_grid[current_stand], weights_of) +
                                                  stand_objective_function(
                                                  config, dict_grid[candidate_stand], weights_of)) / 2

                candidate_dict_deleted = del_dict(dict_grid[current_stand], (random_x, random_y))
                candidate_dict_updated = add_dict(dict_grid[candidate_stand], (random_x, random_y),
                                                  dict_grid[current_stand][(random_x, random_y)])

                if config.SA_ponder_of_by_area:
                    candidate_objective_function = (((current_area - 1) * stand_objective_function(
                                                    config, candidate_dict_deleted, weights_of) +
                                                    (candidate_area + 1) * stand_objective_function(
                                                    config, candidate_dict_updated, weights_of)) /
                                                    (current_area + candidate_area))
                else:
                    candidate_objective_function = (stand_objective_function(
                                                    config, candidate_dict_deleted, weights_of) +
                                                    stand_objective_function(
                                                    config, candidate_dict_updated, weights_of)) / 2

                del candidate_dict_deleted
                del candidate_dict_updated

                if candidate_objective_function > current_objective_function:
                    num_bingos += 1
                    grid[random_x, random_y].stand = candidate_stand
                    dict_grid[candidate_stand][(random_x, random_y)] = dict_grid[current_stand][(random_x, random_y)]
                    del dict_grid[current_stand][(random_x, random_y)]
                else:
                    probability = random.uniform(0, 1)
                    if probability < np.exp((candidate_objective_function - current_objective_function) / t):
                        num_spares += 1
                        grid[random_x, random_y].stand = candidate_stand
                        dict_grid[candidate_stand][(random_x, random_y)] = \
                            dict_grid[current_stand][(random_x, random_y)]
                        del dict_grid[current_stand][(random_x, random_y)]
                    else:
                        num_wrongs += 1

        objective_function_evolution_t = {}
        for stand_num, dict_stand in dict_grid.items():
            objective_function_evolution_t[stand_num] = stand_objective_function(config, dict_stand, weights_of)
        objective_function_evolution.append(objective_function_evolution_t)

        decision_over_cells.update({f'{t_iter}':
                                        [config.SA_I - num_nonnan, num_inners, num_bingos, num_spares, num_wrongs]})
        # store_grid(t_iter, config, grid)
        heatmap_grid_gif(config, grid, 'stand', t_iter, weights_of)
        t = config.SA_C * t
        t_iter += 1

    return grid, dict_grid, decision_over_cells, objective_function_evolution
