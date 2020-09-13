######################################
# selection.py
# Evolutionary Computing Task I
# VU 2020
# Authors: Stefan Klut (2686873)
#          Kimberley Boersma (2572145)
#          Timo van Milligen (2684444)
#          Selma Muhammad (2578081)
######################################

import numpy as np


def tournament_selection(population, k):
    # Sample k individuals and select the most fit individual
    indices = np.random.randint(0, len(population.pop), k)
    best_individual = np.argmax(population.fitness[indices])
    return population.pop[indices[best_individual]]


def fps(population):
    # Return an individual with their probability based on their fitness
    return np.random.choice(population.pop,
                            p=population.fitness/np.sum(population.fitness))


def linear_ranking(population, sp):
    # Return an individual with their probability based on their rank
    rank_pops = [x for _, x in sorted(zip(population.fitness, population.pop))]
    size = len(rank_pops)
    prob = [((2-sp)/size) + (2*i*(sp-1))/(size*(size-1))
            for i in range(size)]
    return np.random.choice(rank_pops, p=prob)


if __name__ == "__main__":
    class P():
        def __init__(self):
            self.pop = list(range(10))
            self.fitness = np.random.randint(0, 10, 10)
            print('population: ', self.pop)
            print('fitness: ', self.fitness)
    a = P()
    print(fps(a))
