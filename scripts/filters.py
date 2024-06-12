"""
Here we have to create:

a function that goes through all the grid with a filter grid of 5x5 cells that redefines the center cell stand as the
most common one at the filter grid. Be careful with the borders!

a function that splits a discontinuous stand into different continuous stands
"""
import numpy as np
import math
from scripts.config import Config
from scripts.input_output import *
from scripts.simulated_annealing import stand_objective_function
from collections import Counter
from scipy.ndimage import label


def mode_cleaner(config: Config, grid_sa: np.ndarray) -> tuple:
    grid = grid_sa.copy()
    side = config.grid_filter_side
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            stands_in_filter = []
            for p in np.linspace(-side / 2 + 0.5, side / 2 - 0.5, side).astype(int):  # if side = 3, equiv to [-1, 0, 1]
                for q in np.linspace(-side / 2 + 0.5, side / 2 - 0.5, side).astype(int):
                    if (i + p) < 0 or (i + p) >= grid.shape[0] or (j + q) < 0 or (j + q) >= grid.shape[1]:
                        continue
                    else:
                        other_stand = grid[i + p, j + q].stand
                        if math.isnan(other_stand) is False:
                            stands_in_filter.append(grid[i + p, j + q].stand)
            grid[i, j].stand = Counter(stands_in_filter).most_common(1)[0][0]

    cleaned_grid = fill_empty_cells(config, grid)
    cleaned_dict = create_dict_grid_thorough(cleaned_grid)

    return cleaned_grid, cleaned_dict


def relabel_connected_components(cleaned_grid: np.ndarray) -> tuple:
    """
    Relabel connected components in a 2D numpy array.

    Parameters:
    arr (numpy.ndarray): A 2D numpy array with integer labels.

    Returns:
    numpy.ndarray: A 2D numpy array with each connected component given a unique label.
    """
    grid = grid_stands(cleaned_grid)

    unique_labels = np.unique(grid)
    new_arr = np.zeros_like(grid)
    current_label = 1

    for label_value in unique_labels:
        # Create a binary array for the current label
        binary_array = (grid == label_value).astype(int)

        # Label the connected components in the binary array
        labeled_array, num_features = label(binary_array)

        # Assign new labels to the connected components
        for i in range(1, num_features + 1):
            new_arr[labeled_array == i] = current_label
            current_label += 1

    split_grid = update_stands_grid(new_arr, cleaned_grid)
    split_dict = create_dict_grid_thorough(split_grid)

    return split_grid, split_dict


def minimum_area_filter(config: Config, split_grid: np.ndarray, split_dict: dict, weights: list) -> tuple:
    filtered_grid = split_grid.copy()
    filtered_dict = split_dict.copy()
    for stand, dict_stand in split_dict.copy().items():
        if len(dict_stand) < config.minimum_area:
            surrounding_stands_set = set()
            for cell in dict_stand:
                for p in [-1, 0, 1]:
                    for q in [-1, 0, 1]:
                        if 0 <= cell[0] + p < filtered_grid.shape[0] and 0 <= cell[1] + q < filtered_grid.shape[1]:
                            if filtered_grid[cell[0] + p, cell[1] + q].stand != stand:
                                surrounding_stands_set.add(filtered_grid[cell[0] + p, cell[1] + q].stand)
            surrounding_stands = list(surrounding_stands_set)
            of_candidates = []
            for other_stand in surrounding_stands:
                dict_other_stand = filtered_dict[other_stand].copy()
                of_now = stand_objective_function(config, dict_other_stand, weights)
                dict_other_stand_new = dict_other_stand | dict_stand
                of_annex = stand_objective_function(config, dict_other_stand_new, weights)
                of_candidates.append(of_annex - of_now)
                del dict_other_stand
            best_stand = surrounding_stands[of_candidates.index(max(of_candidates))]
            filtered_dict[best_stand] = filtered_dict[best_stand].copy() | dict_stand
            for cell in filtered_dict[stand].keys():
                filtered_grid[cell[0], cell[1]].stand = best_stand
            del filtered_dict[stand]
    return filtered_grid, filtered_dict
