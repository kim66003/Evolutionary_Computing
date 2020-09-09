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

os.makedirs("results/", exist_ok=True)
experiment_name = "results/EA_1/"
os.makedirs(experiment_name, exist_ok=True)

# Update the number of neurons for this specific example
n_hidden_neurons = 10

# initializes environment with ai player using random controller, playing against static enemy
env = Environment(experiment_name=experiment_name,
                  player_controller=player_controller(n_hidden_neurons),
                  enemy_mode="static",
                  level=2
                  )

# TODO Load solution
# np.loadtxt?
env.play()
