from snake import AutonomousSnake
import math
import copy
import numpy as np


class Generation:

    def __init__(self, gid, population_size, seed_brains=None, mutation_rate=0, n_crossover=0):
        self.gid = gid
        self.seed_brains = seed_brains
        self.population_size = population_size
        self.snakes = self.populate(mutation_rate, n_crossover)
        self.ranked_snakes = []  # Ranked by time alive
        self.alive = True

        self.age = 0

    def populate(self, mutation_rate, n_crossover):
        if not self.seed_brains:
            return [AutonomousSnake([13, 8], i) for i in range(self.population_size)]

        seeds = self.seed_brains[:]

        # Generate more seeds by crossing over seeds
        for brain in self.seed_brains:
            for other in self.seed_brains:
                if brain is not other:
                    for i in range(n_crossover):
                        seeds.append(brain.crossover(other))

        clones_per_seed = math.ceil(self.population_size / len(seeds))
        snakes = []
        seed_snakes = []
        counter = 0
        for brain in seeds:
            new_snake = AutonomousSnake([13, 8], counter)
            new_snake.brain = brain
            seed_snakes.append(new_snake)
            counter += 1
            for i in range(clones_per_seed):
                new_snake = AutonomousSnake([13, 8], counter)
                new_snake.brain = brain
                snakes.append(new_snake)
                counter += 1

        snakes = self.mutate(snakes, mutation_rate)
        snakes.extend(seed_snakes)
        return snakes

    @staticmethod
    def mutate(snakes, mutation_rate):
        for snake in snakes:
            snake.brain.weights = [w + np.random.normal(size=np.shape(w)) * mutation_rate
                                   for w in snake.brain.weights]
            snake.brain.biases = [b + np.random.normal(size=np.shape(b)) * mutation_rate
                                  for b in snake.brain.biases]
        return snakes

    def step(self):
        for snake in self.snakes:
            snake.think()
            snake.move()
            if not snake.alive:
                self.snakes.remove(snake)
                self.ranked_snakes.append(snake)

    def live(self):
        while len(self.snakes) > 0:
            # print("step: {}".format(self.age))
            self.step()
            self.age += 1
            if self.age % 50 == 0:
                print("{} steps completed. {} snakes still alive".format(self.age, len(self.snakes)))
        print("Did {} steps".format(self.age))
        self.ranked_snakes.sort(key=lambda c: c.score)

    def save_champions(self, min_rank, m=0):
        for snake in self.ranked_snakes[-min_rank:]:
            print("snake {}: {}".format(snake.sid, snake.score))
            snake.save(self.gid)

    def get_champion_brains(self, min_rank):
        brains = []
        for snake in self.ranked_snakes[-min_rank:]:
            brains.append(snake.brain)
        return brains
