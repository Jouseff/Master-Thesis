from dataclasses import dataclass


@dataclass
class Cell:
    stand: int
    stand_aux: int
    data: list
    x: int
    y: int
    x_index: int
    y_index: int