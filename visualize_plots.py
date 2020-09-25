######################################
# visualize_plots.py
# Evolutionary Computing Task I
# VU 2020
# Authors: Stefan Klut (2686873)
#          Kimberley Boersma (2572145)
#          Timo van Milligen (2684444)
#          Selma Muhammad (2578081)
######################################

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from os import listdir
from os.path import isfile, join

def load_files(path, enemy, mutation, weights=False):
    if weights:
        start_string = 'best'
    else:
        start_string = 'results'
    # load results filenames
    results_files = [path + f for f in listdir(path) if isfile(join(path, f)) and f.startswith(start_string)
                     and 'enemy{}'.format(enemy) in f and mutation in f]
    return results_files


def preprocess_results(results):
    results_array = np.array([np.loadtxt(x, skiprows=1) for x in results])
    # average and std for best solution
    best_columns = results_array[:,:,1]
    mean_best = np.mean(best_columns, axis=0)
    std_best = np.std(best_columns, axis=0)
    # average and std for average solution
    average_columns = results_array[:,:,2]
    mean_average = np.mean(average_columns, axis=0)
    std_average = np.std(average_columns, axis=0)
    return mean_best, std_best, mean_average, std_average


def line_plot(results, enemy, extra_print=None, uniform=False, normal=False, none=False):
    colors = ['cyan', 'blue', 'magenta', 'purple']
    for i, result in enumerate(results):
        mean_best, std_best, mean_average, std_average = result
        lower_bound_best = mean_best - std_best
        upper_bound_best = mean_best + std_best
        lower_bound_avg = mean_average - std_average
        upper_bound_avg = mean_average + std_average

        plt.plot(mean_best, color=colors[i*2], linestyle='dashed')
        plt.plot(mean_average, color=colors[i*2+1])
        plt.xlabel('Generations')
        plt.ylabel('Fitness')
        plt.fill_between(range(len(mean_best)), lower_bound_best, upper_bound_best, alpha=.3, color=colors[i*2])    
        plt.fill_between(range(len(mean_average)), lower_bound_avg, upper_bound_avg, alpha=.3, color=colors[i*2+1])

    if len(results) == 2:
        if uniform == True and normal == True:
            plt.legend(['EA_uniform: mean best solution', 'EA_uniform: mean average solution', 'EA_normal: mean best solution', 'EA_normal: mean average solution'], fontsize='x-large')
        elif uniform == True and none == True:
            plt.legend(['EA_uniform: mean best solution', 'EA_uniform: mean average solution', 'EA_nomut: mean best solution', 'EA_nomut: mean average solution'], fontsize='x-large')
        elif normal == True and none == True:
            plt.legend(['EA_normal: mean best solution', 'EA_normal: mean average solution', 'EA_nomut: mean best solution', 'EA_nomut: mean average solution'], fontsize='x-large')

        plt.savefig('results/plots/lineplot_enemy{}{}'.format(enemy, extra_print))
    elif len(results) == 1:
        plt.legend(['EA3: mean best solution', 'EA3: mean average solution'], fontsize='x-large')
        plt.savefig('results/plots/lineplot_enemy{}_doomsday'.format(enemy))
    plt.show()


if __name__ == "__main__":
    enemy = 3
    # results mutation uniform
    results_files_1 = load_files('results/task1/', enemy, 'uniform')
    results_1 = preprocess_results(results_files_1)
    # results mutation normal
    results_files_2 = load_files('results/task1/', enemy, 'normal')
    results_2 = preprocess_results(results_files_2)
    # results doomsday
    # results_files_3 = load_files('results/task1/', enemy, 'doomsday')
    # results_3 = preprocess_results(results_files_3)
    # results mutnone
    # results_files_4 = load_files('results/task1/', enemy, 'none')
    # results_4 = preprocess_results(results_files_4)

    line_plot([results_1, results_2], enemy=enemy, extra_print='_uniform_vs_normal', uniform=True, normal=True)
    
    # line_plot([results_1, results_4], enemy=enemy, extra_print='_uniform_vs_none', uniform=True, none=True)
    # line_plot([results_2, results_4], enemy=enemy, extra_print='_normal_vs_none', normal=True, none=True)
