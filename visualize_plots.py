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


def line_plot(results):
    for result in results:
        mean_best, std_best, mean_average, std_average = result
        lower_bound_best = mean_best - std_best
        upper_bound_best = mean_best + std_best
        lower_bound_avg = mean_average - std_average
        upper_bound_avg = mean_average + std_average

        plt.plot(mean_best, color='blue')
        plt.plot(mean_average, color='purple')
        plt.xlabel('Generations')
        plt.ylabel('Fitness')
        plt.fill_between(range(len(mean_best)), lower_bound_best, upper_bound_best, alpha=.3, color='blue')    
        plt.fill_between(range(len(mean_average)), lower_bound_avg, upper_bound_avg, alpha=.3, color='purple')


    plt.legend(['EA1: mean best solution', 'EA1: mean average solution', 'EA2: mean best solution', 'EA2: mean average solution'])
    plt.show()



if __name__ == "__main__":
    # results enemy 1 mutation normal
    results_files_1 = load_files('results/task1/', 1, 'normal')
    results_1 = preprocess_results(results_files_1)
    # results enemy 2 mutation normal
    results_files_2 = load_files('results/task1/', 2, 'normal')
    results_2 = preprocess_results(results_files_2)
    
    line_plot([results_1, results_2])
