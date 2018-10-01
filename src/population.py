from snake import AutonomousSnake
import copy
import random
import numpy as np
import os


class TrainingConfiguration:

    def __init__(self, mutation_rate, mutation_magnitude, max_moves=100, shuffle=False):
        self.max_moves = max_moves
        self.mutation_rate = mutation_rate
        self.mutation_magnitude = mutation_magnitude
        self.shuffle = shuffle


class Population:

    def __init__(self, population_size, pid, training_config, champion=None):
        self.population_size = population_size
        self.pid = pid

        self.training_config = training_config

        if not champion:
            self.individuals = [AutonomousSnake(base_max_moves=self.training_config.max_moves) for i in
                                range(self.population_size)]
            self.all_time_champion = AutonomousSnake()
        else:
            self.individuals = [copy.copy(champion) for i in range(self.population_size - 1)]
            for individual in self.individuals:
                individual.mutate(self.training_config.mutation_rate, self.training_config.mutation_magnitude)
                individual.max_moves = self.training_config.max_moves
            self.individuals.append(champion)
            self.all_time_champion = champion

        self.top_generation = [self.all_time_champion]
        self.fitness_sum = 0
        self.generation = 0
        try:
            os.mkdir("populations/pop{}".format(self.pid))
        except Exception as e:
            print(e)
        f = open("populations/pop{}/log.log".format(self.pid), "w")
        f.write("Population {} history".format(self.pid))
        f.close()

    def evolution(self):

        steps = 0
        live_individuals = np.arange(len(self.individuals))
        while len(live_individuals) > 0:
            for i in live_individuals:
                self.individuals[i].think()
                self.individuals[i].move()
                if not self.individuals[i].alive:
                    live_individuals = np.delete(live_individuals, np.argwhere(live_individuals == i))
            steps += 1

        self.fitness_sum = 0
        for individual in self.individuals:
            individual.calc_fitness()
            self.fitness_sum += individual.fitness

        self.individuals.sort(key=lambda c: c.fitness)
        self.top_generation = self.individuals[-100:]
        # print("worst fitness: {}, best fitness: {}".format(self.individuals[0].fitness, self.individuals[-1].fitness))

        self.reproduction()

        self.all_time_champion.save(self.pid, self.generation)
        log = "\n\nPopulation {}: Generation {} done in {} steps. champion fitness: {}, total fitness: {}".format(
            self.pid,
            self.generation,
            steps,
            self.all_time_champion.fitness,
            self.fitness_sum)
        print(log)
        with open('populations/pop{}/log.log'.format(self.pid), 'a') as f:
            f.write(log)
        self.generation += 1

    def reproduction(self):
        next_gen = []

        if self.individuals[-1].fitness > self.all_time_champion.fitness:
            self.all_time_champion = self.individuals[-1]

        individuals_pool = self.individuals[:]

        for i in range(self.population_size - 1):
            parent1 = self.select_random_individual(individuals_pool)
            parent2 = self.select_random_individual(individuals_pool)

            child = parent1.crossover_brain(parent2)
            child.mutate(self.training_config.mutation_rate, self.training_config.mutation_magnitude)

            next_gen.append(child)

        next_gen.append(self.all_time_champion.reincarnate())
        self.individuals = next_gen

    def select_random_individual(self, individuals_pool):
        random_point = np.random.randint(0, self.fitness_sum)
        if self.training_config.shuffle:
            random.shuffle(individuals_pool)

        pointer = 0
        for individual in individuals_pool:
            pointer += individual.fitness
            if pointer > random_point:
                return individual
        return self.all_time_champion

    def introduce_from_outside(self, others):
        merged_population = []
        while len(merged_population) < self.population_size:
            merged_population.extend(copy.deepcopy(self.top_generation))
            for other in others:
                merged_population.extend(copy.deepcopy(other.top_generation))
        merged_population = merged_population[:self.population_size]
        for individual in merged_population:
            individual.mutate(self.training_config.mutation_rate, self.training_config.mutation_magnitude)
        merged_population[0] = copy.copy(self.all_time_champion)
        self.individuals = merged_population

    def reset_fitness(self):
        for individual in self.individuals:
            individual.fitness = 0

    def update_config(self, config):
        self.training_config = config
        for individual in self.individuals:
            individual.max_moves = self.training_config.max_moves
