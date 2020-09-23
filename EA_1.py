######################################
# EA_1.py
# Evolutionary Computing Task I
# VU 2020
# Authors: Stefan Klut (2686873)
#          Kimberley Boersma (2572145)
#          Timo van Milligen (2684444)
#          Selma Muhammad (2578081)
######################################

import sys
sys.path.insert(0, 'evoman')
from environment import Environment
from selection import tournament_selection
from demo_controller import player_controller
from selection import *
from crossover_mutations import *
import os
import numpy as np


# environment settings
if len(sys.argv) > 3:
    enemy_no = sys.argv[1]
    if sys.argv[2] == 'normal':
        mutation_method = normal_mutation
        mutation_var = 0.1
    elif sys.argv[2] == 'uniform':
        mutation_method = uniform_mutation
        mutation_var = 0.01
    elif sys.argv[2] == 'none':
        mutation_method = sys.argv[2]
        mutation_var = 0
    if sys.argv[3] in ["on", "off"]:
        logs = sys.argv[3]
    if len(sys.argv) > 4:
        if sys.argv[4] == 'ssh':
            os.environ["SDL_VIDEODRIVER"] = "dummy"
    print("Parameter SETTINGS:\nenemy: {}\nmutation: {}\nmutation_var={}".format(enemy_no, mutation_method, mutation_var))
else:
    print("arg1: 1/2/3 (enemy_no), arg2: normal/uniform/none (mutation), arg3: on/off (prints) arg4: ssh (optional if running in terminal)")
    print("so like this: python EA_1.py 1 normal off ssh\n or: python EA_1.py 2 uniform off\n or: python EA_1.py 3 uniform on")
    sys.exit(1)


experiment_name = "results/task1"
os.makedirs(experiment_name, exist_ok=True)

# initialize hidden neurons
n_hidden_neurons = 10

# Enviroment
env = Environment(experiment_name=experiment_name,
                  enemies=[enemy_no],
                  player_controller=player_controller(n_hidden_neurons),
                  enemymode="static",
                  level=2,
                  speed="fastest",
                  logs=logs)


class Population():
    # initialize population class
    def __init__(self, size, n_weights, lower_bound=-1, upper_bound=1):
        self.size = size
        self.n_weights = n_weights
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.mutation_fraction = 0.2
        self.generation = 0

        # create population
        self.pop = np.random.uniform(self.lower_bound, self.upper_bound,
                                     (self.size, self.n_weights))

        # calculate fitness for initial population
        self.fitness = self.calc_fitness(self.pop)

        # best solution
        self.best_solution = max(self.fitness)
        # stagnation counter
        self.stagnation_count = 0

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
            if not mutation_method == "none":
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
        # update best solution
        if max(self.fitness) > self.best_solution:
            self.best_solution = max(self.fitness)
        else:
            self.stagnation_count += 1
        self.generation += 1


    def save_results(self, training_i, mutation, first_run=False):
        best = np.argmax(self.fitness)
        std  =  np.std(self.fitness)
        mean = np.mean(self.fitness)

        # saves results of this generation
        file_results  = open(experiment_name+f'/results_enemy{env.enemyn}_train{training_i}_mut{mutation}.txt','a')
        if first_run:
            file_results.write('gen best mean std')
        print( '\n GENERATION '+str(self.generation)+' '+str(round(self.fitness[best],6))+' '+str(round(mean,6))+' '+str(round(std,6)))
        file_results.write('\n'+str(self.generation)+' '+str(round(self.fitness[best],6))+' '+str(round(mean,6))+' '+str(round(std,6)))
        file_results.close()

        # save weights
        np.savetxt(experiment_name+f'/best_enemy{env.enemyn}_train{training_i}_mut{mutation}.txt',self.pop[best])


    def __str__(self):
        print_class = ''
        for i in range(len(self.pop)):
            print_class += 'Population: ' + \
                str(i) + ' Fitness: ' + str(self.fitness[i]) + '\n'
        return print_class


def simulate(training_i, n_pop, n_weights, n_children, n_generations, mutation_type, mut_method, mut_var,
             stagnation_point=5):
    # initialize population
    population = Population(n_pop, n_weights)

    # saves results for first pop
    population.save_results(training_i, mutation_type, first_run=True)

    for i in range(n_generations):
        print('Generation: ', i+1)

        # if population fitness stagnates, increase mutation probability
        mutation_multiple = 1
        if population.stagnation_count > stagnation_point:
            print('population has stagnated')
            mutation_multiple = 10
            population.stagnation_count = 0
            # change mutation method and var if population stagnated
            # if mutation_type == "none":
            #     mut_method = uniform_mutation
            #     mut_var = 0.01

        population.create_children(n_children=n_children, 
                                select_method=tournament_selection, select_var=5,
                                cross_method=intermediate_whole, cross_var=0.5, 
                                mutation_method=mut_method, mutation_var=mutation_multiple*mut_var)
        
        # if mutation_type == "none":
        #     # change mutation method and var back to none
        #     mut_method = mutation_type
        #     mut_var = 0
                                
        # new_fitness, new_pop = survival_selection_fitness(population)
        new_fitness, new_pop = survival_selection_prob(population)
        
        #Always let the best of the previous population advance to the next generation
        best = np.argmax(population.fitness)
        new_pop[-1] = population.pop[best]
        new_fitness[-1] = population.fitness[best]
        
        #Replace the population by its children
        population.replace_new_gen(new_pop, new_fitness)
        
        # save results for every generation
        population.save_results(training_i, mutation_type)

if __name__ == "__main__":
    # initialize number of trainings
    n_training = 10
    # initialize parameters
    n_pop, n_weights = 100, (env.get_num_sensors()+1) * \
        n_hidden_neurons + (n_hidden_neurons+1)*5
    n_generations = 30

    n_children = 200

    for i in [9]:#range(n_training):
        print('Training iteration: ', i)
        simulate(i, n_pop=n_pop, n_weights=n_weights, n_children=n_children, n_generations=n_generations, 
        mutation_type=sys.argv[2], mut_method=mutation_method, mut_var=mutation_var)
