import json
import csv
from pathlib import Path


class Config:
    def __init__(self) -> None:
        path_project = str(Path(__file__).parent.parent.absolute())
        with open(path_project + "/config.json") as file:
            config = json.load(file)

        self.lidar_transformation = config['lidar_transformation']
        self.blocks = self.lidar_transformation['blocks']

        self.seed = config['seed']

        self.paths = config['paths']
        self.path_LiDAR = f"{path_project}/{self.paths['LiDAR']}"
        self.path_forest = f"{path_project}/{self.paths['forest']}"
        self.path_others = f"{path_project}/{self.paths['others']}"
        self.path_dfs = f"{path_project}/{self.paths['dfs']}"
        self.path_maps = f"{path_project}/{self.paths['maps']}"
        self.path_plots = f"{path_project}/{self.paths['plots']}"
        self.path_gifs = f"{path_project}/{self.paths['gifs']}"
        self.path_maps_for_gif = f"{path_project}/{self.paths['maps_for_gif']}"

        with open(self.path_LiDAR + f'/filtered_lidar_{self.blocks[0]}.csv') as csv_file:
            reader = csv.reader(csv_file)
            self.used_metrics = next(reader)[:-2]

        self.map = config['map']
        self.area_cell = self.map['area_cell']
        self.stands_per_block_side = self.map['stands_per_block_side']

        self.SA = config['simulated_annealing']
        self.SA_on = self.SA['activate']
        self.SA_Ti = self.SA['T_initial']
        self.SA_Tf = self.SA['T_final']
        self.SA_C = self.SA['Cooling']
        self.SA_I = self.SA['Iter_per_T']
        self.SA_term_weights = self.SA['term_weights']
        self.SA_metric_weights = self.SA['metric_weights']
        self.SA_iter_stats = self.SA['iter_stats']

        self.GA = config['genetic_algorithm']
        self.GA_on = self.GA['activate']

        self.SOM = config['self_organizing_map']
        self.SOM_on = self.SOM['activate']

        self.CA = config['cellular_automata']
        self.CA_on = self.CA['activate']

        self.fine_tuning = config['fine_tuning']
        self.grid_filter_side = self.fine_tuning['grid_filter_side']
        self.hybrid_CA = self.fine_tuning['cellular_automata_hybrid']

        self.cleaning = config['cleaning']
        self.minimum_area = self.cleaning['minimum_area']

        self.evaluation = config['evaluation']
        self.evaluate_variance = self.evaluation['statistics']['variance']
        self.evaluate_R2 = self.evaluation['statistics']['R2']
        self.evaluate_area_distribution = self.evaluation['morphology']['area_distribution']
        self.evaluate_roundness = self.evaluation['morphology']['roundness']

        self.graphic_outputs = config['graphic_outputs']
        self.gif_duration = self.graphic_outputs['gif_duration']
