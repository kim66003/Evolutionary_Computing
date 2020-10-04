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
import scipy.spatial


def tournament_selection(population, k=5):
    # Sample k individuals and select the most fit individual
    indices = np.random.randint(0, len(population.pop), k)
    best_individual = np.argmax(population.fitness[indices])
    return population.pop[indices[best_individual]]


def fps(population, *_):
    norm_fitness = norm(population.fitness)
    # Return an individual with their probability based on their fitness
    index_individual = np.random.choice(range(len(population.pop)),
                                        p=norm_fitness/np.sum(norm_fitness))
    return population.pop[index_individual]


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

# Survival selection methods
def survival_selection_fitness(population):
    children_fitness_reshape = np.reshape(population.children_fitness, (-1, 1))
    _, rank_index = zip(*sorted(zip(children_fitness_reshape, 
                                    range(len(population.children)), 
                                key=lambda x:x[0],
                                reverse=True)))
    return np.array(rank_index[:population.size])

def survival_selection_prob(population):
    norm_fitness = norm(population.children_fitness)
    probs = norm_fitness/np.sum(norm_fitness)
    rank_index = np.random.choice(range(len(population.children)), 
                                  size=population.size,
                                  p=probs, 
                                  replace=False)
    return rank_index


def norm(fitness):
    # helpers function for survival selection with probabilities
    max_fitness = max(fitness)
    min_fitness = min(fitness)
    if (max_fitness - min_fitness) > 0:
        fit_norm = (fitness - min_fitness)/(max_fitness - min_fitness)
    else:
        fit_norm = np.zeros_like(fitness)    
    fit_norm = [0.00000001 if val <= 0 else val for val in fit_norm]
    return fit_norm

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
    print(distance(a.children))
