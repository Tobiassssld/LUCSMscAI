import sys
from typing import Tuple
import numpy as np
import bisect
# you need to install this package `ioh`. Please see documentations here: 
# https://iohprofiler.github.io/IOHexp/ and
# https://pypi.org/project/ioh/
import ioh
from ioh import get_problem, logger, ProblemClass

budget = 5000

np.random.seed(42)
def single_crossover(p1, p2, crossover_probability):
    if np.random.uniform(0, 1) < crossover_probability:
        point = np.random.randint(1, len(p1))
        p1[point:], p2[point:] = p2[point:], p1[point:]


# Uniform Crossover
def crossover(p1, p2, crossover_probability):
    if np.random.uniform(0, 1) < crossover_probability:
        for i in range(len(p1)):
            if np.random.uniform(0, 1) < 0.5:
                t = p1[i]
                p1[i] = p2[i]
                p2[i] = t


def get_crossover_dict():
    return {
        "single": single_crossover,
        "uniform": crossover
    }


def get_mutation_dict():
    return {
        "standard": mutation,
        "multi_bit": multi_bit_mutation
    }


# Standard bit mutation using mutation rate p
def mutation(p, mutation_rate):
    for i in range(len(p)):
        if np.random.uniform(0, 1) < mutation_rate:
            p[i] = 1 - p[i]


# Random reseting mutation
def multi_bit_mutation(p, mutation_rate):
    num_bits = int(len(p) * mutation_rate)
    positions = np.random.choice(len(p), num_bits, replace=False)
    for pos in positions:
        p[pos] = 1 - p[pos]


def get_selection_dict():
    return {
        "tournament": tournament_selection,
        "proportional": mating_seletion
    }


def tournament_selection(parent, parent_f, tournament_k):
    # Using the tournament selection
    select_parent = []
    for i in range(len(parent)):
        tournament_k = min(tournament_k, len(parent))
        pre_select = np.random.choice(len(parent_f), tournament_k, replace=False)
        max_f = sys.float_info.min
        index = 0
        for p in pre_select:
            if parent_f[p] > max_f:
                index = p
                max_f = parent_f[p]
        select_parent.append(parent[index].copy())
    return select_parent


def mating_seletion(parent, parent_f, tournament_k):
    # Using the proportional selection

    # Plusing 0.001 to avoid dividing 0
    f_min = min(parent_f)
    f_sum = sum(parent_f) - (f_min - 0.001) * len(parent_f)

    rw = [(parent_f[0] - f_min + 0.001) / f_sum]
    for i in range(1, len(parent_f)):
        rw.append(rw[i - 1] + (parent_f[i] - f_min + 0.001) / f_sum)

    select_parent = []
    for i in range(len(parent)):
        r = np.random.uniform(0, 1)
        index = 0
        # print(rw,r)
        while (r > rw[index]):
            index = index + 1

        select_parent.append(parent[index].copy())
    return select_parent


def s4196864_s4213211_GA(problem: ioh.problem.PBO, pop_size=5, mutation_rate=0.01,
                         crossover_probability=0.5, num_evaluations=budget, mutation_type="standard",
                         crossover_type="uniform", selection_type="proportional", tournament_k=2) -> None:
    # initial_pop = ... make sure you randomly create the first population
    # `problem.state.evaluations` counts the number of function evaluation automatically,
    # which is incremented by 1 whenever you call `problem(x)`.
    # You could also maintain a counter of function evaluations if you prefer.
    f_opt = sys.float_info.min
    x_opt = None

    parent = []
    parent_f = []
    for i in range(pop_size):
        # Initialization
        parent.append(np.random.randint(2, size=problem.meta_data.n_variables))
        parent_f.append(problem(parent[i]))
    while problem.state.evaluations < num_evaluations:
        # please implement the mutation, crossover, selection here
        # .....
        # this is how you evaluate one solution `x`
        # f = problem(x)
        # no return value needed
        offspring = get_selection_dict()[selection_type](parent, parent_f, tournament_k)

        for i in range(0, pop_size - (pop_size % 2), 2):
            get_crossover_dict()[crossover_type](offspring[i], offspring[i + 1], crossover_probability)

        for i in range(pop_size):
            get_mutation_dict()[mutation_type](offspring[i], mutation_rate)

        parent = offspring.copy()
        for i in range(pop_size):
            parent_f[i] = problem(parent[i])
            if parent_f[i] > f_opt:
                f_opt = parent_f[i]
                x_opt = parent[i].copy()

    print(f_opt, x_opt)
    return f_opt


def create_problem(dimension: int, fid: int) -> Tuple[ioh.problem.PBO, ioh.logger.Analyzer]:
    # Declaration of problems to be tested.
    problem = get_problem(fid, dimension=dimension, instance=1, problem_class=ProblemClass.PBO)

    # Create default logger compatible with IOHanalyzer
    # `root` indicates where the output files are stored.
    # `folder_name` is the name of the folder containing all output. You should compress the folder 'run' and upload it to IOHanalyzer.
    l = logger.Analyzer(
        root="data",
        # the working directory in which a folder named `folder_name` (the next argument) will be created to store data
        folder_name="run",  # the folder name to which the raw performance data will be stored
        algorithm_name="genetic_algorithm",  # name of your algorithm
        algorithm_info="Practical assignment of the EA course",
    )
    # attach the logger to the problem
    problem.attach_logger(l)
    return problem, l


if __name__ == "__main__":
    # this how you run your algorithm with 20 repetitions/independent run
    # create the LABS problem and the data logger
    F18, _logger = create_problem(dimension=50, fid=18)
    for run in range(20):
        s4196864_s4213211_GA(F18)
        F18.reset()  # it is necessary to reset the problem after each independent run
    _logger.close()  # after all runs, it is necessary to close the logger to make sure all data are written to the folder

    print("time for 23")

    # create the N-Queens problem and the data logger
    F23, _logger = create_problem(dimension=49, fid=23)
    for run in range(20):
        s4196864_s4213211_GA(F23)
        F23.reset()
    _logger.close()
