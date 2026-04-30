from typing import List

import numpy as np
# you need to install this package `ioh`. Please see documentations here: 
# https://iohprofiler.github.io/IOHexp/ and
# https://pypi.org/project/ioh/
from ioh import get_problem, logger, ProblemClass
from s4196864_s4213211_GA import s4196864_s4213211_GA, create_problem, tournament_selection

budget = 5000000

# To make your results reproducible (not required by the assignment), you could set the random seed by
# `np.random.seed(some integer, e.g., 42)`
np.random.seed(42)

# Hyperparameters to tune, e.g.
hyperparameter_space = {
    "population_size": [5, 8, 10, 20, 25, 30],
    "mutation_rate": [0.01, 0.015, 0.02, 0.03, 0.05, 0.1],
    "crossover_rate": [0.5, 0.6, 0.7, 0.75, 0.8]
}


# Hyperparameter tuning function
def tune_hyperparameters() -> List:
    # You should decide/engineer the `score` youself, which is the tuning objective

    # create the LABS problem and the data logger
    F18, _logger1 = create_problem(dimension=50, fid=18)
    # create the N-Queens problem and the data logger
    F23, _logger2 = create_problem(dimension=49, fid=23)

    rest_budget = budget
    results = []
    # Track min/max for normalization
    exit_loop = False
    for pop_size in hyperparameter_space['population_size']:
        for mutation_rate in hyperparameter_space['mutation_rate']:
            for crossover_rate in hyperparameter_space['crossover_rate']:
                # You should initialize you GA implementation with a hyperparameter setting
                # and execute it on both problems F18, and F23
                # please decide how many function evaluations you wish to use for running the GA
                # on each problem per each hyperparameter setting
                # ......
                for mutation_type in ['standard', 'multi_bit']:
                    for crossover_type in ['single', 'uniform']:
                        for selection_choice in ['proportional', 2, 5, 8]:
                            if rest_budget <= 0:
                                exit_loop = True
                                break
                            num_evaluations_F18 = 5000
                            num_evaluations_F23 = 5000
                            result_F18 = 0
                            result_F23 = 0
                            if selection_choice != 'proportional':
                                selection_type = 'tournament'
                                tournament_k = selection_choice
                            else:
                                selection_type = 'proportional'
                                tournament_k = 0
                            for _ in range(20):
                                result_F18 += s4196864_s4213211_GA(F18, pop_size, mutation_rate, crossover_rate,
                                                                   min(num_evaluations_F18, rest_budget), mutation_type,
                                                                   crossover_type, selection_type, tournament_k)
                                rest_budget -= num_evaluations_F18
                                F18.reset()
                                result_F23 += s4196864_s4213211_GA(F23, pop_size, mutation_rate, crossover_rate,
                                                                   min(num_evaluations_F23, rest_budget), mutation_type,
                                                                   crossover_type, selection_type, tournament_k)
                                rest_budget -= num_evaluations_F23
                                F23.reset()

                            result_F18 /= 20
                            result_F23 /= 20

                            results.append({
                                'params': [pop_size, mutation_rate, crossover_rate, mutation_type, crossover_type,
                                           selection_type, tournament_k],
                                'F18': result_F18,  # Maximize
                                'F23': result_F23  # Minimize
                            })
                        if exit_loop:
                            break
                    if exit_loop:
                        break
                if exit_loop:
                    break
            if exit_loop:
                break
        if exit_loop:
            break

    _logger1.close()
    _logger2.close()
    pareto_front = []
    for i, result1 in enumerate(results):
        is_dominated = False
        for j, result2 in enumerate(results):
            if i != j:
                # Check if result2 dominates result1 (both objectives to maximize)
                if (result2['F18'] >= result1['F18'] and result2['F23'] >= result1['F23']) and \
                        (result2['F18'] > result1['F18'] or result2['F23'] > result1['F23']):
                    is_dominated = True
                    break
        if not is_dominated:
            pareto_front.append(result1)

    # Step 2: Select best compromise solution
    # For maximization, ideal point is max of both objectives
    ideal_F18 = max(r['F18'] for r in results)
    ideal_F23 = max(r['F23'] for r in results)

    best_distance = float('inf')
    best_solution = None

    for solution in pareto_front:
        # Normalize distances (changed for maximization)
        norm_F18 = (ideal_F18 - solution['F18']) / (ideal_F18 - min(r['F18'] for r in results))
        norm_F23 = (ideal_F23 - solution['F23']) / (ideal_F23 - min(r['F23'] for r in results))
        distance = (norm_F18 ** 2 + norm_F23 ** 2) ** 0.5

        if distance < best_distance:
            best_distance = distance
            best_solution = solution

    best_params = best_solution['params']
    print(best_solution)

    return best_params


if __name__ == "__main__":
    # Hyperparameter tuning to determine the best parameters for both problems
    population_size, mutation_rate, crossover_rate, mutation_type, crossover_type, selection_type, tournament_k = tune_hyperparameters()
    print(population_size)
    print(mutation_rate)
    print(crossover_rate)
    print(mutation_type)
    print(crossover_type)
    print(selection_type)
    print(tournament_k)
