"""
Here we have to create:

a function that goes through all the grid with a filter grid of 5x5 cells that redefines the center cell stand as the
most common one at the filter grid. Be careful with the borders!

a function that splits a discontinuous stand into different continuous stands
"""
import numpy as np
import math
from scripts.config import Config
from collections import Counter
from itertools import product


def mode_filter(config: Config, grid: np.ndarray) -> np.ndarray:
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

    return grid


def dfs(adjacency_list: dict, start_node, visited: set, dict_connected: dict, dict_stand: dict):
    visited.add(start_node)
    dict_connected[start_node] = dict_stand[start_node]
    for next_node in adjacency_list[start_node] - visited:
        dfs(adjacency_list, next_node, visited, dict_connected, dict_stand)
    return visited, dict_connected


def stand_splitter(grid: np.ndarray, dict_grid: dict) -> tuple:
    # 0. before using this function, clean empty nested dictionaries!
    # 1. renumber stands into connected components and rewrite stand dictionaries accordingly
    new_dict_grid = {}
    max_stand_index = len(dict_grid) - 1
    for stand_num, dict_stand in dict_grid.items():
        # 1.1. creation of the "neighboring cells" graph of the stand
        cell_coords = [tuple(cell_data[:2]) for cell_data in dict_stand.values()]
        # 1.1.1. compute all possible edges between cells, now considered vertices of a graph
        stand_edges = product(cell_coords, cell_coords)
        # 1.1.2. filter out all edges that connect a cell to itself or to non-contiguous cells
        stand_nearby_edges = []
        for edge in stand_edges:
            length = np.sqrt((edge[0][0] - edge[1][0])**2 + (edge[0][1] - edge[1][1])**2)
            if np.sqrt(2) >= length > 0:
                stand_nearby_edges.append(edge)

        # 1.2. create the adjacency list of the "neighboring cells" graph
        adjacency_list = {}
        for u, v in stand_nearby_edges:
            adjacency_list.setdefault(u, []).append(v)
        adjacency_list = {key: set(value) for key, value in adjacency_list.items()}

        # 1.3. go through all cells of the stand and record the connected components of its corresponding graph
        number_of_partitions = 0
        yet_to_visit = cell_coords
        while True:
            number_of_partitions += 1
            visited = set()
            dict_connected = {}
            # 1.3.1. apply a "depth first search" algorithm to every connected part of the graph
            visited, dict_connected = dfs(adjacency_list, yet_to_visit[0], visited, dict_connected, dict_stand)
            # 1.3.2. update the list of cells that are yet to be visited by the dfs algorithm
            yet_to_visit = [cell for cell in yet_to_visit if cell not in list(visited)]
            # 1.3.3. append to the new dictionary of the grid the dictionary of the found connected component by the dfs
            if len(yet_to_visit) == 0:
                break
            elif len(yet_to_visit) != 0 and number_of_partitions == 1:
                new_dict_grid[stand_num] = dict_connected
            else:
                max_stand_index += 1
                new_dict_grid[max_stand_index] = dict_connected
    del dict_grid

    # 2. update the numbering of the stands of the np.ndarray grid
    for stand_num, dict_stand in new_dict_grid.items():
        for cell_data in dict_stand.values():
            grid[cell_data[0], cell_data[1]].stand = stand_num

    return grid, new_dict_grid
