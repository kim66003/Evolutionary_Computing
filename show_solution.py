######################################
# show_solution.py
# Evolutionary Computing Task I
# VU 2020
# Authors: Stefan Klut (2686873)
#          Kimberley Boersma (2572145)
#          Timo van Milligen (2684444)
#          Selma Muhammad (2578081)
######################################

# imports framework
import sys, os
sys.path.insert(0, 'evoman') 
from environment import Environment
from demo_controller import player_controller
import numpy as np

os.makedirs("results/", exist_ok=True)
experiment_name = "results/EA_1/"
os.makedirs(experiment_name, exist_ok=True)

# Update the number of neurons for this specific example
n_hidden_neurons = 10

# initializes environment with ai player using random controller, playing against static enemy
env = Environment(experiment_name=experiment_name,
                  player_controller=player_controller(n_hidden_neurons),
                  enemymode="static",
                  enemies = [1],
                  level=2,
                  speed='normal'
                  )

# TODO Load solution
fitness_sharing = ['fitness_sharing','no_fitness_sharing']
groups = ['[4,5,6]', '[4,7,8]']

for sharing in fitness_sharing:
	# tests saved demo solutions for each enemy
    for group in groups:
        gains = []
        for train in range(0, 10):
            mean_indv_gain = []
            
            print('\n LOADING SAVED SOLUTION FOR GROUP: '+str(group) + 'FITNESS: '+ sharing + 'TRAINING: ' + str(train))
            # Load generalist controller
            solution = np.loadtxt('results/task2/training/'+str(sharing)+'best_enemy'+str(group)+'_train'+str(train)+
                '_discrete_uniform_fps_survival_selection_prob_mutuniform_sigma4.txt')
            for enemy in range(1,9):
                # Update the enemy  
                env.update_parameter('enemies',[enemy])

                for run in range(5):
                    env.play(solution)
                    individual_gain = env.get_playerlife() - env.get_enemylife()
                    mean_indv_gain.append(individual_gain)
                    print('individual gain', env.get_playerlife() - env.get_enemylife())
            gains.append(np.mean(mean_indv_gain))

        # write individual gain to text file
        np.savetxt(f'results/best_solutions/task2/individual_gain_{sharing}_{group}', gains)