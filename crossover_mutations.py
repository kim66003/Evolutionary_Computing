######################################
# cross_mutations.py
# Evolutionary Computing Task I
# VU 2020
# Authors: Stefan Klut (2686873)
#          Kimberley Boersma (2572145)
#          Timo van Milligen (2684444)
#          Selma Muhammad (2578081)
######################################

import numpy as np
from copy import deepcopy


def discrete_uniform(parent1, parent2):
    child = deepcopy(parent1)
    temp = deepcopy(parent2)
    mask = np.random.randint(2, size=len(child), dtype=bool)
    child[mask] = temp[mask]
    return child


if __name__ == "__main__":
    parent1 = np.array(range(10))
    parent2 = np.array(range(10, 20))
    child = discrete_uniform(parent1, parent2)
    print(parent1)
