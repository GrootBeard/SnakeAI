from snake import AutonomousSnake
import copy
import random
import numpy as np


class Population:

    def __init__(self, population_size, pid, champion=None, mutation_rate=0, crossover_rate=0):
        self.population_size = population_size
        self.pid = pid

        if not champion:
            self.individuals = [AutonomousSnake() for i in range(self.population_size)]
            self.all_time_champion = AutonomousSnake()
        else:
            self.individuals = [copy.copy(champion) for i in range(self.population_size - 1)]
            for individual in self.individuals:
                individual.mutate(mutation_rate)
            self.individuals.append(champion)
            self.all_time_champion = champion

        self.top_generation = [self.all_time_champion]
        self.fitness_sum = 0
        self.generation = 0
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate

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
        print("worst fitness: {}, best fitness: {}".format(self.individuals[0].fitness, self.individuals[-1].fitness))

        self.reproduction()

        self.all_time_champion.save(self.pid, self.generation)
        print("Generation {} done in {} steps. champion fitness: {}, total fitness: {}".format(self.generation, steps,
                                                                                               self.all_time_champion.fitness,
                                                                                               self.fitness_sum))
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
            child.mutate(self.mutation_rate)

            next_gen.append(child)

        next_gen.append(self.all_time_champion.reincarnate())
        self.individuals = next_gen

    def select_random_individual(self, individuals_pool):
        random_point = np.random.randint(0, self.fitness_sum)
        # random.shuffle(individuals_pool)

        pointer = 0
        for individual in individuals_pool:
            pointer += individual.fitness
            if pointer > random_point:
                return individual
        return self.all_time_champion


    def introduce_from_outside(self, others):
        merged_population = []
        while len(merged_population) < self.population_size:
            merged_population.extend(self.top_generation)
            for other in others:
                merged_population.extend(other.top_generation)
        merged_population = merged_population[:self.population_size]
        for individual in merged_population:
            individual.mutate(self.mutation_rate)
        self.individuals = merged_population