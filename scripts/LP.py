import gurobipy as gp
from gurobipy import GRB
import numpy as np
from config import Config


class ModelConstructor:
    def __init__(self, grid: np.ndarray, dict_metrics: dict, config: Config, model: gp.model):
        self.cell_area = 0.04
        self.cell_perimeter = 80
        self.dev_max_metric1 = 1.5
        self.big_constant = 1000
        self.metric1 = dict_metrics[1]
        self.stands = []
        self.coords = []

        self.m = model

        self.variables = {}

        self.stand_area = None
        self.stand_perimeter = None
        self.cell_in_stand = None
        self.adjacent_in_stand = None

        self.mean_metric1 = None
        self.dev_metric1 = None
        self.mean_metric1_in_stand = None

    def optimization(self):
        self.define_variables()
        self.define_objective_function()
        self.define_constraints()

        return self.m

    def define_variables(self):
        self.variables['cell_in_stand'] = self.m.addVars(self.coords, self.stands, vtype=GRB.BINARY, name='x')
        self.variables['adjacent_in_stand'] = self.m.addVars(self.coords, self.coords, self.stands, vtype=GRB.BINARY,
                                                             name='epsilon')
        self.variables['stand_area'] = self.m.addVars(self.stands, lb=0.0, vtype=GRB.CONTINUOUS, name='a')
        self.variables['stand_perimeter'] = self.m.addVars(self.stands, lb=0.0, vtype=GRB.CONTINUOUS, name='p')

        # variables for the metrics
        self.variables['mean_metric1'] = self.m.addVars(self.stands, lb=0.0, vtype=GRB.CONTINUOUS, name='h_bar')
        self.variables['dev_metric1'] = self.m.addVars(self.coords, self.stands, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS,
                                                       name='d')
        self.variables['mean_metric1_in_stand'] = self.m.addVars(self.coords, self.stands, lb=0.0, vtype=GRB.CONTINUOUS,
                                                                 name='omega')

    def define_objective_function(self):
        self.m.setObjective(sum(self.stand_perimeter[s] for s in self.stands), sense=GRB.MINIMIZE)

    def define_constraints(self):
        self.m.addConstrs((sum(self.variables['cell_in_stand'][c, s] * self.cell_area for c in self.coords) ==
                           self.variables['stand_area'][s] for s in self.stands), name='1')

        self.m.addConstrs((sum(self.variables['cell_in_stand'][c, s] for s in self.stands) <= 1 for c in self.coords),
                          name='2')

        self.m.addConstrs((self.variables['stand_area'][s1] <= 1.2 * self.variables['stand_area'][s2]
                           for s1 in self.stands for s2 in self.stands), name='3.1')
        self.m.addConstrs((self.variables['stand_area'][s1] >= 0.8 * self.variables['stand_area'][s2]
                           for s1 in self.stands for s2 in self.stands), name='3.2')

        self.m.addConstrs((self.variables['stand_perimeter'][s] ==
                           4 * sum(self.cell_in_stand[c, s] for c in self.coords) - 2 *
                           sum(self.adjacent_in_stand[c1, c2, s] for c1 in self.coords for c2 in self.coords)
                           for s in self.stands), name='4')
        self.m.addConstrs((self.variables['adjacent_in_stand'][c1, c2, s] <= self.variables['cell_in_stand'][c1, s]
                           for c1 in self.coords for c2 in self.coords for s in self.stands), name='4.aux1')
        self.m.addConstrs((self.variables['adjacent_in_stand'][c1, c2, s] <= self.variables['cell_in_stand'][c2, s]
                           for c1 in self.coords for c2 in self.coords for s in self.stands), name='4.aux2')

        # constraints for the metrics
        self.m.addConstrs((sum(self.variables['mean_metric1'][c] * self.variables['cell_in_stand'][c, s]
                               for s in self.stands) == sum(self.variables['mean_metric1_in_stand'][c, s]
                                                            for s in self.stands) for c in self.coords), name='5')
        self.m.addConstrs((self.variables['mean_metric1_in_stand'][c, s] <= self.big_constant *
                           self.variables['cell_in_stand'][c, s] for c in self.coords for s in self.stands),
                          name='5.aux1')
        self.m.addConstrs((self.variables['mean_metric1_in_stand'][c, s] <= self.variables['mean_metric1'][s]
                           for c in self.coords for s in self.stands), name='5.aux2')
        self.m.addConstrs((self.variables['mean_metric1'][s] - self.variables['mean_metric1_in_stand'][c, s] +
                           self.big_constant * self.variables['cell_in_stand'][c, s] <= self.big_constant
                           for c in self.coords for s in self.stands), name='5.aux3')

        self.m.addConstrs((self.metric1[c] * self.variables['cell_in_stand'] - self.variables['mean_metric1_in_stand']
                           <= self.variables['dev_metric1'][c, s] for c in self.coords for s in self.stands),
                          name='6.1')
        self.m.addConstrs((self.variables['mean_metric1_in_stand'] - self.metric1[c] * self.variables['cell_in_stand']
                           <= self.variables['dev_metric1'][c, s] for c in self.coords for s in self.stands),
                          name='6.2')

        self.m.addConstrs((self.variables['dev_metric1'][c, s] <= self.dev_max_metric1 for c in self.coords
                           for s in self.stands), name='7')

