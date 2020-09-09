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
from environment import Environment
from demo_controller import player_controller
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


class Population():
    # initialize population class
    def __init__(self, size, n_weights, lower_bound=-1, upper_bound=1):
        self.size = size
        self.n_weights = n_weights
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

        # create population
        self.pop = np.random.uniform(self.lower_bound, self.upper_bound,
                                     (self.size, self.n_weights))

        # calculate fitness for initial population
        self.fitness = np.array([env.play(pcont=x)[0] for x in self.pop])

    def __str__(self):
        print_class = ''
        for i in range(len(self.pop)):
            print_class += 'Population: ' + \
                str(i) + ' Fitness: ' + str(self.fitness[i]) + '\n'
        return print_class


population = Population(n_pop, n_weights)

print(tournament_selection(population, 4))


# Save the initial solution
env.update_solutions([population.pop, population.fitness])
