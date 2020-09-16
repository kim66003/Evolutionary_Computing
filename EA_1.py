######################################
# EA_1.py
# Evolutionary Computing Task I
# VU 2020
# Authors: Stefan Klut (2686873)
#          Kimberley Boersma (2572145)
#          Timo van Milligen (2684444)
#          Selma Muhammad (2578081)
######################################

from selection import tournament_selection
import sys
sys.path.insert(0, 'evoman')
from environment import Environment
from demo_controller import player_controller
from selection import *
from crossover_mutations import *
import os
import sys

import numpy as np

sys.path.insert(0, 'evoman')


# initialize hidden neurons
n_hidden_neurons = 10

experiment_name = "test123"
os.makedirs(experiment_name, exist_ok=True)

# Enviroment
env = Environment(experiment_name=experiment_name,
                  enemies=[2],
                  player_controller=player_controller(n_hidden_neurons),
                  enemymode="static",
                  level=2,
                  speed="fastest")

# initialize parameters
n_pop, n_weights = 10, (env.get_num_sensors()+1) * \
    n_hidden_neurons + (n_hidden_neurons+1)*5
n_generations = 3


class Population():
    # initialize population class
    def __init__(self, size, n_weights, lower_bound=-1, upper_bound=1):
        self.size = size
        self.n_weights = n_weights
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.mutation_fraction = 0.2

        # create population
        self.pop = np.random.uniform(self.lower_bound, self.upper_bound,
                                     (self.size, self.n_weights))

        # calculate fitness for initial population
        self.fitness = self.calc_fitness(self.pop)

    def calc_fitness(self, pop):
        return np.array([env.play(pcont=x)[0] for x in pop])
    
    def create_children(self, n_children, 
                        select_method=tournament_selection, select_var=None,
                        cross_method=intermediate_whole, cross_var=None,
                        mutation_method=normal_mutation, mutation_var=None
                        ):
        # function that runs selection, crossover, mutation and returns n children

        assert mutation_var is not None, "Please add variable for the mutation."
        assert cross_var is not None, "Please add variable for the cross."
        assert select_var is not None, "Please add variable for the mutation."

        children = [] # list with children
        
        for i in range(n_children):
            # select parents
            parent1 = select_method(self, select_var)
            parent2 = select_method(self, select_var)
            # create child with crossover
            child = cross_method(parent1, parent2, cross_var)
            # mutate fraction of children
            if np.random.binomial(n=1, p=self.mutation_fraction):
                child = mutation_method(child, mutation_var)
            children.append(child)

        self.children = np.array(children)
        self.children_size = n_children
        self.children_fitness = self.calc_fitness(self.children)

    
    def replace_new_gen(self, new_population, new_fitness):
        # make sure new population and fitness have same shape as old gen
        assert new_population.shape == self.pop.shape
        assert new_fitness.shape == self.fitness.shape
        # replace old generation with new generation
        self.pop = new_population
        self.fitness = new_fitness


    def __str__(self):
        print_class = ''
        for i in range(len(self.pop)):
            print_class += 'Population: ' + \
                str(i) + ' Fitness: ' + str(self.fitness[i]) + '\n'
        return print_class

# initialize population
population = Population(n_pop, n_weights)

# TODO loop dit

for i in range(n_generations):
    print('Generation: ', i)
    population.create_children(n_children=10, 
                               select_method=tournament_selection, select_var=4,
                               cross_method=intermediate_whole, cross_var=0.5, 
                               mutation_method=normal_mutation, mutation_var=0.1)
    # new_fitness, new_pop = survival_selection_fitness(population)
    new_fitness, new_pop = survival_selection_prob(population)
    population.replace_new_gen(new_pop, new_fitness)


# Save the initial solution
env.update_solutions([population.pop, population.fitness])
