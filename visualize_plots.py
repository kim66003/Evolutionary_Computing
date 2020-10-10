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


def line_plot(results, methods, colors, enemy, extra_print=None):
    for i, result in enumerate(results):
        mean_best, std_best, mean_average, std_average = result
        lower_bound_best = mean_best - std_best
        upper_bound_best = mean_best + std_best
        lower_bound_avg = mean_average - std_average
        upper_bound_avg = mean_average + std_average

        # first plot mean of best solutions
        plt.plot(mean_best, color=colors[i*2], linestyle='dashed')
        # then plot mean of average solutions
        plt.plot(mean_average, color=colors[i*2+1])
        plt.xlabel('Generations')
        plt.ylabel('Fitness')
        # plot std of mean best
        plt.fill_between(range(len(mean_best)), lower_bound_best, upper_bound_best, alpha=.3, color=colors[i*2])   
        # plot std of mean average 
        plt.fill_between(range(len(mean_average)), lower_bound_avg, upper_bound_avg, alpha=.3, color=colors[i*2+1])

    legend_titles = []
    for i in range(len(methods)):
        legend_titles.append(f'{methods[i]}: mean best solution')
        legend_titles.append(f'{methods[i]}: mean average solution')

    plt.legend(legend_titles, fontsize='large')
    plt.savefig('results/plots/lineplot_enemy{}{}'.format(enemy, extra_print))
    plt.show()


if __name__ == "__main__":
    enemy = '[4, 7, 8]'
    paths = ['results/task2/training/fitness_sharing/', 'results/task2/training/no_fitness_sharing/']
    methods = ['sigma4', 'sigmaNone']
    results = []

    for path, method in zip(paths, methods):
        results_files = load_files(path, enemy, method)
        result = preprocess_results(results_files)
        results.append(result)

    colors = ['red', 'orange', 'cyan', 'green', 'purple', 'yellow', 'magenta', 'blue', 'sienna', 'darkviolet', 'teal', 'pink']
    line_plot(results, methods, colors, enemy=enemy, extra_print='_sigma_tuning')
