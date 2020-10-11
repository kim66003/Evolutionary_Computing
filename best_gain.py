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

# Update the number of neurons for this specific example
n_hidden_neurons = 10

os.makedirs("results/", exist_ok=True)
experiment_name = "results/EA_2/"
os.makedirs(experiment_name, exist_ok=True)

# initializes environment with ai player using random controller, playing against static enemy
env = Environment(experiment_name=experiment_name,
                  player_controller=player_controller(n_hidden_neurons),
                  enemymode="static",
                  enemies = [1],
                  level=2,
                  speed='fastest'
                  )

group = "[4, 7, 8]" # TODO
sharing = "fitness_sharing" #TODO
if sharing == 'no_fitness_sharing':
    sigma = None
else:
    sigma = 4
train = 0 #TODO

player_life = []
enemy_life = []

print('\n LOADING SAVED SOLUTION FOR GROUP: '+str(group) + ' FITNESS: '+ sharing + ' TRAINING: ' + str(train))
# Load generalist controller
solution = np.loadtxt('./results/task2/training/'+str(sharing)+'/best_enemy'+str(group)+'_train'+str(train)+
    '_discrete_uniform_fps_survival_selection_prob_mutuniform_sigma'+str(sigma)+'.txt')
for enemy in range(1,9):
    # Update the enemy  
    env.update_parameter('enemies',[enemy])

    player_life_list, enemy_life_list = [], []
    for run in range(5):
        env.play(solution)
        individual_gain = env.get_playerlife() - env.get_enemylife()
        player_life_list.append(env.get_playerlife())
        enemy_life_list.append(env.get_enemylife())
    player_life.append(np.mean(player_life_list))
    enemy_life.append(np.mean(enemy_life_list))

path = 'results/best_solutions/task2/'
with open(path+'best_solution_player_enemy_life.txt', 'w') as f:
    f.write("player life\n")
    for item in player_life:
        f.write("%s\n" % item)
    f.write("enemy life\n")
    for item in enemy_life:
        f.write("%s\n" % item)