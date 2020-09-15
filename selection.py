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


def fps(population, *_):
    # Return an individual with their probability based on their fitness
    return np.random.choice(population.pop,
                            p=population.fitness/np.sum(population.fitness))


def linear_ranking(population, sp):
    # Return an individual with their probability based on their rank
    fitness_reshape = np.reshape(population.fitness, (-1, 1))
    rank_pops = np.array([x for _, x in sorted(zip(fitness_reshape, 
                                                   population.pop), 
                                               key=lambda x:x[0])])
    size = len(rank_pops)
    prob = [((2-sp)/size) + (2*i*(sp-1))/(size*(size-1))
            for i in range(size)]
    index_individual = np.random.choice(range(len(rank_pops)), p=prob)
    return rank_pops[index_individual]


def survival_selection_fitness(population):
    children_fitness_reshape = np.reshape(population.children_fitness, (-1, 1))
    rank_children = [x for _, x in sorted(zip(children_fitness_reshape, 
                                              population.children), 
                                          key=lambda x:x[0],
                                          reverse=True)]
    return rank_children[:population.size]


if __name__ == "__main__":
    class P():
        def __init__(self):
            self.children = np.random.rand(10,20)
            self.children_fitness = np.random.randint(0, 10, 10)
            self.children_size = 10
            self.size = 10
            print('population: ', self.children)
            print('fitness: ', self.children_fitness)
    a = P()
    print(survival_selection_fitness(a))
