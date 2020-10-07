######################################
# EA_1.py
# Evolutionary Computing Task II
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
import scipy


# environment settings
if len(sys.argv) > 6:
    enemies = [int(sys.argv[1][0]),int(sys.argv[1][1]),int(sys.argv[1][2])]
    # crossover type
    if sys.argv[2] == 'discrete_uniform':
        crossover_method = discrete_uniform
        crossover_var = 0.5
    elif sys.argv[2] == 'discrete_n_point':
        crossover_method = discrete_n_point
        crossover_var = 20  # random value for n point
    elif sys.argv[2] == 'intermediate_single':
        crossover_method = intermediate_single
        crossover_var = 0.5
    elif sys.argv[2] == 'intermediate_whole':
        crossover_method = intermediate_whole
        crossover_var = 0.5
    elif sys.argv[2] == 'intermediate_simple':
        crossover_method = intermediate_simple
        crossover_var = 0.5
    elif sys.argv[2] == 'intermediate_blend':
        crossover_method = intermediate_blend
        crossover_var = 0.5    
    # selection type
    if sys.argv[3] == 'tournament_selection':
        selection_method = tournament_selection
    elif sys.argv[3] == 'fps':
        selection_method = fps
    elif sys.argv[3] == 'linear_ranking':
        selection_method = linear_ranking
    # survival type
    if sys.argv[4] == 'survival_selection_fitness':
        survival_method = survival_selection_fitness 
    elif sys.argv[4] == 'survival_selection_prob':
        survival_method = survival_selection_prob
    # mutation type
    if sys.argv[5] == 'normal':
        mutation_method = normal_mutation
        mutation_var = 0.1
    elif sys.argv[5] == 'uniform':
        mutation_method = uniform_mutation
        mutation_var = 0.01
    if sys.argv[6] in ["on", "off"]:
        logs = sys.argv[6]
    # set fitness sharing parameter
    sharing = False
    if len(sys.argv) > 7:
        if sys.argv[7] == 'ssh':
            os.environ["SDL_VIDEODRIVER"] = "dummy"
    if len(sys.argv) > 8:
        if sys.argv[8] == 'fitness_sharing':
            sharing = True
    print("Parameter SETTINGS: enemies: {}\ncrossover: {}\nselection: {}\nselection_survival: {}\nmutation: {}\nmutation_var={}".format(enemies, sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], mutation_var))
else:
    print("arg1: enemy_no1enemy_no2enemy_no3, arg2: crossover type, arg3: selection type, arg4: survival selection, arg5: normal/uniform (mutation), arg6: on/off (prints) arg7: ssh (optional if running in terminal)")
    print("so like this: python EA_task2.py 123 intermediate_blend tournament_selection survival_selection_fitness normal off ssh")
    sys.exit(1)


experiment_name = "results/task2/parameter_tuning/sigma_tuning/"
os.makedirs(experiment_name, exist_ok=True)

# initialize hidden neurons
n_hidden_neurons = 10

# Enviroment
env = Environment(experiment_name=experiment_name,
                  enemies=enemies,
                  player_controller=player_controller(n_hidden_neurons),
                  enemymode="static",
                  level=2,
                  speed="fastest",
                  multiplemode="yes",
                  logs=logs)


class Population():
    # initialize population class
    def __init__(self, size, n_weights, lower_bound=-1, upper_bound=1, sharing=False, sigma=None):
        self.size = size
        self.n_weights = n_weights
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.mutation_fraction = 0.2
        self.generation = 0
        self.sharing = sharing
        self.sigma = sigma
        self.original_fitness = []
        self.shared_fitnesses = []
        # create population
        self.pop = np.random.uniform(self.lower_bound, self.upper_bound,
                                     (self.size, self.n_weights))

        # calculate fitness for initial population
        self.fitness = self.calc_fitness(self.pop, sigma=sigma)
        # best solution
        self.best_solution = max(self.fitness)
        # stagnation counter
        self.stagnation_count = 0
    
    def distance(self, pop, metric='euclidean'):
        distance = scipy.spatial.distance.pdist(pop, metric=metric)
        distance_matrix = scipy.spatial.distance.squareform(distance)
        return distance_matrix

    def fitness_sharing(self, distance_matrix, fitness, sigma=None):
        denom = np.array([sum(1 - (i / sigma) if i <= sigma else 0 for i in l) 
                          for l in distance_matrix])
        #With how many other agents an agent shares its fitness
        self.shared_fitnesses = np.array([sum(1 if i <= sigma else 0 for i in l) 
                          for l in distance_matrix])
        return (fitness / denom)
        
    def calc_fitness(self, pop, sigma=None):
        results = np.array([env.play(pcont=x) for x in pop])
        fitness = results[:, 0]
        self.original_fitness = fitness
        #player_life = results[:, 1]
        #enemy_life = results[:, 2]
        #time = results[:, 3]
        
        if self.sharing:
            distance_matrix = self.distance(pop)
            return self.fitness_sharing(distance_matrix, fitness, sigma=sigma)
        else:
            return fitness
    
    def create_children(self, n_children, sigma,
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
        self.children_fitness = self.calc_fitness(self.children, sigma=sigma)

    
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
            self.stagnation_count = 0
        else:
            self.stagnation_count += 1
        self.generation += 1


    def save_results(self, training_i, crossover, selection, survival, mutation, sigma, first_run=False):
        best = np.argmax(self.fitness)
        std  =  np.std(self.fitness)
        mean = np.mean(self.fitness)
        best_performance = np.argmax(self.original_fitness)
        # saves results of this generation
        file_results  = open(experiment_name+f'/results_enemy{env.enemies}_train{training_i}_{crossover}_{selection}_{survival}_mut{mutation}_sigma{sigma}.txt','a')
        if first_run:
            file_results.write('gen best mean std')
        print( '\n GENERATION '+str(self.generation)+' '+str(round(self.fitness[best],6))+' '+str(round(mean,6))+' '+str(round(std,6)))
        file_results.write('\n'+str(self.generation)+' '+str(round(self.fitness[best],6))+' '+str(round(mean,6))+' '+str(round(std,6)))
        file_results.close()
        
        if sharing:
            file_results  = open(experiment_name+f'/shared_fitness_results_sigma{sigma}_enemy{env.enemies}_train{training_i}.txt','a')
            print('shared fitnesses ', self.shared_fitnesses)
            file_results.write('\n'+ str(self.shared_fitnesses))
            file_results.close()

        # save weights
        np.savetxt(experiment_name+f'/best_enemy{env.enemies}_train{training_i}_{crossover}_{selection}_{survival}_mut{mutation}_sigma{sigma}.txt',self.pop[best_performance])


    def __str__(self):
        print_class = ''
        for i in range(len(self.pop)):
            print_class += 'Population: ' + \
                str(i) + ' Fitness: ' + str(self.fitness[i]) + '\n'
        return print_class


def simulate(training_i, n_pop, n_weights, n_children, n_generations, 
            cross_type, cross_method, select_type, select_method,
            surv_type, surv_method, mut_type, mut_method, mut_var, sigma,
            cross_var=0.5, select_var=5, stagnation_point=5, fitness_sharing=False):
    # initialize population
    population = Population(n_pop, n_weights, sharing=fitness_sharing, sigma=sigma)

    # saves results for first pop
    population.save_results(training_i, cross_type, select_type, surv_type, mut_type, sigma, first_run=True)

    for i in range(n_generations):
        print('Generation: ', i+1)

        # if population fitness stagnates, increase mutation probability
        mutation_multiple = 1
        if population.stagnation_count > stagnation_point:
            print('population has stagnated')
            mutation_multiple = 10
            population.stagnation_count = 0

        best_index = np.argmax(population.original_fitness)
        best_pop = population.pop[best_index]
        best_fitness = population.original_fitness[best_index]
        best_temp = population.fitness[best_index]

        population.create_children(n_children=n_children, 
                                select_method=select_method, select_var=select_var,
                                cross_method=cross_method, cross_var=cross_var, 
                                mutation_method=mut_method, mutation_var=mutation_multiple*mut_var, sigma=sigma)
                                
        # new_fitness, new_pop = survival_selection_fitness(population)
        indices = surv_method(population)
        population.original_fitness = population.original_fitness[indices]
        
        #Save how much fitness the survivors shared
        if(sharing):
            population.shared_fitnesses = np.array(population.shared_fitnesses)[indices]

        new_fitness = population.children_fitness[indices]
        new_pop = population.children[indices]
        #Always let the best of the previous population advance to the next generation
        # best = np.argmax(population.original_fitness)
        
        new_pop[-1] = best_pop
        new_fitness[-1] = best_temp
        population.original_fitness[-1] = best_fitness
        #Replace the population by its children
        population.replace_new_gen(new_pop, new_fitness)
        
        # save results for every generation
        population.save_results(training_i, cross_type, select_type, surv_type, mut_type, sigma)

if __name__ == "__main__":
    # initialize number of trainings
    n_training = 10
    # initialize parameters
    n_pop, n_weights = 30, (env.get_num_sensors()+1) * \
        n_hidden_neurons + (n_hidden_neurons+1)*5
    n_generations = 10
    n_children = 90
    sigmas_fitness = [0.5, 1, 1.5, 2]
    # sigma = None

    for i in range(n_training):
        print('Training iteration: ', i)
        for sigma_ in sigmas_fitness:
            simulate(i, n_pop=n_pop, n_weights=n_weights, n_children=n_children, n_generations=n_generations, 
            cross_type=sys.argv[2], cross_method=crossover_method, cross_var=crossover_var, select_type=sys.argv[3], 
            select_method=selection_method, surv_type=sys.argv[4], surv_method=survival_method, 
            mut_type=sys.argv[5], mut_method=mutation_method, mut_var=mutation_var, fitness_sharing=sharing, sigma=sigma_)
