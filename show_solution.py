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
                  level=2,
                  speed='normal'
                  )

# TODO Load solution
mutations = ['uniform','normal']

for mutation in mutations:
	# tests saved demo solutions for each enemy
    for enemy in range (1,2):
        gains = []
        for train in range(0, 10):
            mean_indv_gain = []
            for run in range(5):
                print('\n LOADING SAVED SOLUTION FOR ENEMY: '+str(enemy)+' mutation: '+str(mutation) + ' training: '+str(train))
                # Update the enemy  
                env.update_parameter('enemies',[enemy])
                # Load specialist controller
                solution = np.loadtxt('results/task2/best_enemy'+str(enemy)+'_train'+str(train)+'_mut'+mutation+'.txt')
                env.play(solution)
                individual_gain = env.get_playerlife() - env.get_enemylife()
                mean_indv_gain.append(individual_gain)
                print('individual gain', env.get_playerlife() - env.get_enemylife())
            gains.append(np.mean(mean_indv_gain))

        # write individual gain to text file
        np.savetxt('results/best_solutions/individual_gain_enemy{}_mut{}'.format(enemy, mutation), gains)