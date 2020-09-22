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


def line_plot(results, enemy):
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


    plt.legend(['EA1: mean best solution', 'EA1: mean average solution', 'EA2: mean best solution', 'EA2: mean average solution'], fontsize='x-large')
    plt.savefig('results/plots/lineplot_enemy{}'.format(enemy))
    plt.show()


# function for setting the colors of the box plots pairs
def setBoxColors(bp):
    print(bp['boxes'], bp['caps'], bp['whiskers'], bp('fliers'), bp['medians'])
    plt.setp(bp['boxes'][0], color='blue')
    plt.setp(bp['caps'][0], color='blue')
    plt.setp(bp['caps'][1], color='blue')
    plt.setp(bp['whiskers'][0], color='blue')
    plt.setp(bp['whiskers'][1], color='blue')
    plt.setp(bp['fliers'][0], color='blue')
    plt.setp(bp['fliers'][1], color='blue')
    plt.setp(bp['medians'][0], color='blue')

    plt.setp(bp['boxes'][1], color='red')
    plt.setp(bp['caps'][2], color='red')
    plt.setp(bp['caps'][3], color='red')
    plt.setp(bp['whiskers'][2], color='red')
    plt.setp(bp['whiskers'][3], color='red')
    plt.setp(bp['fliers'][2], color='red')
    plt.setp(bp['fliers'][3], color='red')
    plt.setp(bp['medians'][1], color='red')

def boxplot(results_en1, results_en2):
    A = [[1, 2, 5,],  [7, 2]]
    B = [[5, 7, 2, 2, 5], [7, 2, 5]]
    C = [[3,2,5,7], [6, 7, 3]]

    fig = plt.figure()
    ax = plt.axes()

    # first boxplot pair
    bp = plt.boxplot(A, positions = [1, 2], widths = 0.6)
    # setBoxColors(bp)

    # second boxplot pair
    bp = plt.boxplot(B, positions = [4, 5], widths = 0.6)
    # setBoxColors(bp)

    # thrid boxplot pair
    bp = plt.boxplot(C, positions = [7, 8], widths = 0.6)
    # setBoxColors(bp)

    # set axes limits and labels
    plt.xlim(0,9)
    plt.ylim(0,9)
    ax.set_xticklabels(['Enemy 1', 'Enemy 2', 'Enemy 3'])
    ax.set_xticks([1.5, 4.5, 7.5])

    # draw temporary red and blue lines and use them to create a legend
    hB, = plt.plot([1,1],'b-')
    hR, = plt.plot([1,1],'r-')
    plt.legend((hB, hR),('EA 1', 'EA 2'))
    hB.set_visible(False)
    hR.set_visible(False)

    plt.show()


if __name__ == "__main__":
    # enemy = 3
    # # results enemy 1 mutation normal
    # results_files_1 = load_files('results/task1/', enemy, 'uniform')
    # results_1 = preprocess_results(results_files_1)
    # # results enemy 2 mutation normal
    # results_files_2 = load_files('results/task1/', enemy, 'normal')
    # results_2 = preprocess_results(results_files_2)
    
    # line_plot([results_1, results_2], enemy=enemy)
    best_sol_path = 'results/best_solutions/'
    results_gain_uniform = np.loadtxt(best_sol_path+'individual_gain_enemy3_mutuniform')
    results_gain_normal = np.loadtxt(best_sol_path+'individual_gain_enemy3_mutnormal')
    boxplot(results_gain_uniform, results_gain_normal)
