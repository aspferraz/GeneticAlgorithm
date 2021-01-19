'''
    File name: main.py
    Author: Antonio Ferraz
    Date created: 11/20/2020
    Date last modified: 11/25/2020
    Python Version: 3.8
'''

import struct
import math
import numpy as np
import random
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

DEFAULT_PRECISION = 6


def float_to_bin(num):
    return format(struct.unpack('!I', struct.pack('!f', num))[0], '032b')


def bin_to_float(binary):
    return struct.unpack('!f', struct.pack('!I', int(binary, 2)))[0]


def rand_decision(probability):
    return random.random() < probability


# changes according to the problem
def fitness(x):
    if -1 <= x <= 2:
        f = x * math.sin(10 * math.pi * x) + 1
    else:
        f = float('-inf')

    return round(f, DEFAULT_PRECISION)


def calc_population_fitness(population):
    f = np.empty(len(population))
    for c_idx in range(len(population)):
        c = population[c_idx]
        f[c_idx] = fitness(bin_to_float(c))
    return f


def select_best_individuals(population, population_fitness, rate=0.5):
    fitness_copy = np.copy(population_fitness)
    selected_qtt = np.uint16(len(population) * rate)
    best_individuals = np.empty(selected_qtt, dtype="<U32")

    for parent_idx in range(selected_qtt):
        max_fitness_idx = np.where(fitness_copy == np.max(fitness_copy))
        max_fitness_idx = max_fitness_idx[0][0]
        best_individuals[parent_idx] = population[max_fitness_idx]
        fitness_copy[max_fitness_idx] = float('-inf')

    return best_individuals


def select_individuals(population, population_fitness, rate=0.5):
    fitness_copy = np.copy(population_fitness)
    selected_qtt = np.uint16(len(population) * rate)
    parents = np.empty(selected_qtt, dtype="<U32")
    if rate < 1:
        max_fitness = np.max(fitness_copy)
        population_idx = 0
        while '' in parents:
            if population_idx == len(population):
                population_idx = 0
            current_fitness = fitness_copy[population_idx]
            if rand_decision(current_fitness / max_fitness):
                parent_idx = np.where(parents == '')[0][0]
                parents[parent_idx] = population[population_idx]
                fitness_copy[population_idx] = float('-inf')
            population_idx += 1
    else:
        parents = np.copy(population)
    return parents


def crossover(parents, rate=0.7):
    offspring = np.empty(parents.shape[0], dtype="<U32")
    sample = random.sample(range(parents.shape[0]), np.uint16(parents.shape[0] * rate))

    point = np.uint8(len(parents[0]) / 2 + 1)

    for k in range(offspring.shape[0]):
        if k in sample:
            parent1_idx = k % parents.shape[0]
            parent2_idx = (k + 1) % parents.shape[0]
            signal = str(np.uint8(parents[parent1_idx][0]) * np.uint8(parents[parent2_idx][0]))
            child = signal + parents[parent1_idx][1:point] + parents[parent2_idx][point:]
            offspring[k] = child
        else:
            offspring[k] = parents[k]

    return offspring


def mutation(population, rate=0.01):
    for idx in range(population.shape[0]):
        chromosome = population[idx]
        for gene_idx in range(32):
            if rand_decision(rate):
                new_gene = '0' if chromosome[gene_idx] == '1' else '1'
                population[idx] = '%s%s%s' % (chromosome[:gene_idx], new_gene, chromosome[gene_idx + 1:])

    return population


def plot_results(data_frame):
    # defines font size and line width
    sns.set(font_scale=1, rc={"lines.linewidth": 2})

    # defines plot size
    plt.rcParams["figure.figsize"] = [20, 10]

    grid = sns.lineplot(data=data_frame, x="time", y="best value in population", hue="mutation rate",
                        style="mutation rate")

    grid.set(yscale="log")

    plt.show()


def run(population_size, domain_limits, generations=5, model='generational', selection_rate=1.0, elitism_rate=0.1,
        crossover_rate=0.7, mutation_rate=0.01):
    if model == 'steady-state':
        if selection_rate >= 1:
            raise ValueError('selection_rate must be < 1 if model is steady-state')
    else:  # generational
        if selection_rate != 1:
            raise ValueError('selection_rate must be 1 if model is generational')

    phenotypes = np.random.uniform(domain_limits[0], domain_limits[1], [population_size])
    population = list(map(lambda x: float_to_bin(round(x, DEFAULT_PRECISION)), phenotypes))

    results = {}
    for generation in range(1, generations + 1):
        if not generation % 10:
            print('Generation:', generation)

        population_fitness = calc_population_fitness(population)

        if not generation % 10:
            results[f"{mutation_rate}:{generation}"] = np.max(population_fitness)

        parents = select_individuals(population, population_fitness, rate=selection_rate)

        elite = select_best_individuals(population, population_fitness, elitism_rate)

        offspring = crossover(parents, rate=crossover_rate)

        offspring = mutation(offspring, rate=mutation_rate)

        # Creating the new population based on the parents and offspring.
        if elite.shape[0]:
            population[0:elite.shape[0]] = elite

        remaining_population_size = (len(population) - parents.shape[0]) - elite.shape[0]

        if remaining_population_size > 0:
            population[elite.shape[0]: elite.shape[0] + remaining_population_size] = np.random.choice(
                parents[elite.shape[0]:],
                remaining_population_size)

            population[elite.shape[0] + remaining_population_size:] = offspring
        else:
            population[elite.shape[0]:] = \
                offspring[: offspring.shape[0] + remaining_population_size]

    return results


if __name__ == '__main__':
    models = ('generational', 'steady-state')
    data = run(100, (-1, 2), generations=200, model=models[1], selection_rate=0.85, elitism_rate=0.01,
               crossover_rate=0.9, mutation_rate=0.01)
    data.update(run(100, (-1, 2), generations=200, model=models[1], selection_rate=0.85, elitism_rate=0.01,
                    crossover_rate=0.9, mutation_rate=0.05))
    data.update(run(100, (-1, 2), generations=200, model=models[1], selection_rate=0.85, elitism_rate=0.01,
                    crossover_rate=0.9, mutation_rate=0.1))

    df = pd.DataFrame.from_dict(data, orient="index", columns=["best value in population"])
    df["mutation rate"] = [i.split(":")[0] for i in df.index]
    df["time"] = [int(i.split(":")[1]) for i in df.index]
    print(df.head(100))
    plot_results(df)
