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
    # Uniformly select a gene from one of the parents
    child = deepcopy(parent1)
    temp = deepcopy(parent2)
    mask = np.random.randint(2, size=len(child), dtype=bool)
    child[mask] = temp[mask]
    return child


def discrete_n_point(parent1, parent2, n):
    # Cross over between n distinct locations
    child1 = deepcopy(parent1)
    child2 = deepcopy(parent2)
    random_indices = sorted(np.random.choice(range(1, len(child1)), n,
                                             replace=False))
    split_child1 = np.split(child1, random_indices)
    split_child2 = np.split(child2, random_indices)

    # Swap the genes from the split parents
    for i in range(n + 1):
        if i % 2 == 0:
            split_child1[i], split_child2[i] = split_child2[i], split_child1[i]

    # Choose which child to keep
    if np.random.randint(2):
        print('child2: ', np.concatenate(split_child2))
        return np.concatenate(split_child1)
    else:
        print('child1: ', np.concatenate(split_child1))
        return np.concatenate(split_child2)


def uniform_mutation(individual, prob=0.01):
    # Randomly reset a gene to a uniformly sampled value
    mask = np.random.uniform(0, 1, len(individual))
    lower_bound, upper_bound = -1, 1
    individual = [np.random.uniform(lower_bound, upper_bound)
                  if mask[i] < prob else ind for i, ind in enumerate(individual)]
    return np.array(individual)


def normal_mutation(individual, sigma=0.1):
    # Add Gaussian noise to all genes
    return individual + np.random.normal(0, sigma, size=len(individual))


if __name__ == "__main__":
    parent1 = np.array(range(10))
    parent2 = np.array(range(10, 20))
    child = discrete_n_point(parent1, parent2, 3)
    print(child)

    individual = np.full((1, 100), 100)[0]
    # print(normal_mutation(individual))
