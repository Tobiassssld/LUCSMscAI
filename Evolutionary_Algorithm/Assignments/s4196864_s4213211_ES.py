import numpy as np
# you need to install this package `ioh`. Please see documentations here: 
# https://iohprofiler.github.io/IOHexp/ and
# https://pypi.org/project/ioh/
from ioh import get_problem, logger, ProblemClass

budget = 50000
dimension = 10
np.random.seed(42)


# To make your results reproducible (not required by the assignment), you could set the random seed by
# `np.random.seed(some integer, e.g., 42)`
def global_intermediary_recombination(population, step_sizes, rotation_angles, lambda_es, random_type=2):
    mu_es = population.shape[0]
    angles_dimension = rotation_angles.shape[1]  # number of rotation angles

    recombined_population = np.zeros((lambda_es, dimension))
    recombined_step_sizes = np.zeros((lambda_es, dimension))
    recombined_angles = np.zeros((lambda_es, angles_dimension))

    for i in range(lambda_es):
        weights = np.random.rand(mu_es) if random_type == 1 else np.ones(mu_es)
        weights /= weights.sum()  # Normalize weights to sum to 1

        offspring = np.dot(weights, population)
        recombined_population[i] = offspring

        offspring_step_sizes = np.dot(weights, step_sizes)
        recombined_step_sizes[i] = offspring_step_sizes

        offspring_angles = np.dot(weights, rotation_angles)
        recombined_angles[i] = offspring_angles

    return recombined_population, recombined_step_sizes, recombined_angles


def discrete_recombination(population, step_sizes, rotation_angles, lambda_es, random_type=3):
    mu_es = population.shape[0]
    angles_dimension = rotation_angles.shape[1]

    recombined_population = np.zeros((lambda_es, dimension))
    recombined_step_sizes = np.zeros((lambda_es, dimension))
    recombined_angles = np.zeros((lambda_es, angles_dimension))

    iteration_size = (lambda_es + 1) // 2

    for i in range(iteration_size - 1):
        parent_indices = np.random.choice(mu_es, size=2, replace=False)
        parent1_idx, parent2_idx = parent_indices

        # Create random masks
        mask = np.random.random(dimension) < 0.5
        angle_mask = np.random.random(angles_dimension) < 0.5

        recombined_population[2 * i] = np.where(mask, population[parent1_idx], population[parent2_idx])
        recombined_population[(2 * i) + 1] = np.where(mask, population[parent2_idx], population[parent1_idx])
        recombined_step_sizes[2 * i] = np.where(mask, step_sizes[parent1_idx], step_sizes[parent2_idx])
        recombined_step_sizes[(2 * i) + 1] = np.where(mask, step_sizes[parent2_idx], step_sizes[parent1_idx])
        recombined_angles[2 * i] = np.where(angle_mask, rotation_angles[parent1_idx], rotation_angles[parent2_idx])
        recombined_angles[(2 * i) + 1] = np.where(angle_mask, rotation_angles[parent2_idx],
                                                  rotation_angles[parent1_idx])

    parent_indices = np.random.choice(mu_es, size=2, replace=False)
    parent1_idx, parent2_idx = parent_indices

    mask = np.random.random(dimension) < 0.5
    angle_mask = np.random.random(angles_dimension) < 0.5

    recombined_population[2 * (iteration_size - 1)] = np.where(mask, population[parent1_idx], population[parent2_idx])
    recombined_step_sizes[2 * (iteration_size - 1)] = np.where(mask, step_sizes[parent1_idx], step_sizes[parent2_idx])
    recombined_angles[2 * (iteration_size - 1)] = np.where(angle_mask, rotation_angles[parent1_idx],
                                                           rotation_angles[parent2_idx])
    # Odd number judgement.
    if (lambda_es % 2 == 0):
        recombined_population[(2 * iteration_size) - 1] = np.where(mask, population[parent2_idx],
                                                                   population[parent1_idx])
        recombined_step_sizes[(2 * iteration_size) - 1] = np.where(mask, step_sizes[parent2_idx],
                                                                   step_sizes[parent1_idx])
        recombined_angles[(2 * iteration_size) - 1] = np.where(angle_mask, rotation_angles[parent2_idx],
                                                               rotation_angles[parent1_idx])

    return recombined_population, recombined_step_sizes, recombined_angles


def correlated_mutation(population, step_sizes, rotation_angles):
    lambda_es = population.shape[0]

    tau = 1.0 / np.sqrt(2 * dimension)
    tau_prime = 1.0 / np.sqrt(2 * np.sqrt(dimension))  # global step size
    beta = 0.0873  # ≈ 5 degrees, for rotation angles

    # Mutate rotation angles
    random_angles = np.random.normal(0, beta, rotation_angles.shape)
    mutated_angles = rotation_angles + random_angles

    # Mutate step sizes
    global_step = np.random.normal(0, 1, (lambda_es, 1))
    local_steps = np.random.normal(0, 1, step_sizes.shape)
    mutated_steps = step_sizes * np.exp(tau_prime * global_step + tau * local_steps)

    # Create covariance matrices and perform correlated mutation
    mutated_population = np.zeros_like(population)
    for i in range(lambda_es):
        # Create rotation matrix from angles
        C = create_covariance_matrix(mutated_angles[i])

        # Generate correlated random vector
        z = np.random.multivariate_normal(np.zeros(dimension), C)

        mutated_population[i] = population[i] + mutated_steps[i] * z

    return mutated_population, mutated_steps, mutated_angles


def get_recombination_dict():
    return {
        1: global_intermediary_recombination,
        2: global_intermediary_recombination,
        3: discrete_recombination
    }


def create_covariance_matrix(angles):
    """Create covariance matrix from rotation angles."""
    C = np.eye(dimension)
    idx = 0

    for i in range(dimension - 1):
        for j in range(i + 1, dimension):
            # Create rotation matrix
            angle = angles[idx]
            R = np.eye(dimension)
            R[i, i] = np.cos(angle)
            R[i, j] = -np.sin(angle)
            R[j, i] = np.sin(angle)
            R[j, j] = np.cos(angle)

            # Update covariance matrix
            C = R @ C
            idx += 1

    return C @ C.T


def s4196864_s4213211_ES(problem, pattern='+', recombination_type=3, mu_es=2, lambda_es=4):
    # hint: F18 and F19 are Boolean problems. Consider how to present bitstrings as real-valued vectors in ES
    # initial_pop = ... make sure you randomly create the first population

    # `problem.state.evaluations` counts the number of function evaluation automatically,
    # which is incremented by 1 whenever you call `problem(x)`.
    # You could also maintain a counter of function evaluations if you prefer.
    dimension_angles = (dimension * (dimension - 1)) // 2
    population = np.random.uniform(low=-5.0, high=5.0, size=(mu_es, dimension))  # individuals
    step_sizes = np.ones((mu_es, dimension)) * 0.1  # mutation step sizes
    rotation_angles = np.zeros((mu_es, dimension_angles))  # rotation angles
    fitness = np.array([problem(x) for x in population])

    while problem.state.evaluations < budget:
        # please implement the mutation, crossover, selection here
        # .....
        # this is how you evaluate one solution `x`
        # f = problem(x)
        # 1. Recombination
        offspring, offspring_steps, offspring_angles = get_recombination_dict()[recombination_type](
            population, step_sizes, rotation_angles, lambda_es, recombination_type
        )

        # 2. Correlated Mutation
        mutated_offspring, mutated_steps, mutated_angles = correlated_mutation(
            offspring, offspring_steps, offspring_angles
        )

        # 3. Evaluation
        offspring_fitness = np.array([problem(x) for x in mutated_offspring])

        # 4. Selection (μ,λ) or (μ + λ)

        # Select best μ individuals
        if pattern == ',':
            selected_indices = np.argsort(offspring_fitness)[:mu_es]  # minimize
            population = mutated_offspring[selected_indices]
            step_sizes = mutated_steps[selected_indices]
            rotation_angles = mutated_angles[selected_indices]
            fitness = offspring_fitness[selected_indices]
        else:
            combined_pop = np.vstack((population, mutated_offspring))
            combined_steps = np.vstack((step_sizes, mutated_steps))
            combined_angles = np.vstack((rotation_angles, mutated_angles))
            combined_fitness = np.concatenate((fitness, offspring_fitness))

            selected_indices = np.argsort(combined_fitness)[:mu_es]  # minimize
            population = combined_pop[selected_indices]
            step_sizes = combined_steps[selected_indices]
            rotation_angles = combined_angles[selected_indices]
            fitness = combined_fitness[selected_indices]

    # Return best solution found
    best_idx = np.argmin(fitness)
    print(f"best_fitness: {fitness[best_idx]}")

    # no return value needed 


def create_problem(fid: int):
    # Declaration of problems to be tested.
    problem = get_problem(fid, dimension=dimension, instance=1, problem_class=ProblemClass.BBOB)

    # Create default logger compatible with IOHanalyzer
    # `root` indicates where the output files are stored.
    # `folder_name` is the name of the folder containing all output. You should compress the folder 'run' and upload it to IOHanalyzer.
    l = logger.Analyzer(
        root="data",
        # the working directory in which a folder named `folder_name` (the next argument) will be created to store data
        folder_name="run",  # the folder name to which the raw performance data will be stored
        algorithm_name="evolution strategy",  # name of your algorithm
        algorithm_info="Practical assignment part2 of the EA course",
    )
    # attach the logger to the problem
    problem.attach_logger(l)
    return problem, l


if __name__ == "__main__":
    # this how you run your algorithm with 20 repetitions/independent run
    F23, _logger = create_problem(23)
    for run in range(20):
        s4196864_s4213211_ES(F23, pattern='+', recombination_type=3, mu_es=15, lambda_es=100)
        F23.reset()  # it is necessary to reset the problem after each independent run
    _logger.close()  # after all runs, it is necessary to close the logger to make sure all data are written to the folder
