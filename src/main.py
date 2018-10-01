from population import Population, TrainingConfiguration
from snake import AutonomousSnake
import numpy as np

training_config = TrainingConfiguration(0.35, 0.2, 90)

snake1 = AutonomousSnake.load_snake("populations/pop{}/gen{}.pickle".format(1, 2))
snake2 = AutonomousSnake.load_snake("populations/pop{}/gen{}.pickle".format(2, 18))
snake3 = AutonomousSnake.load_snake("populations/pop{}/gen{}.pickle".format(6006, 6))
snake4 = AutonomousSnake.load_snake("populations/pop{}/gen{}.pickle".format(6007, 25))
snake5 = AutonomousSnake.load_snake("populations/pop{}/gen{}.pickle".format(7007, 25))
snake6 = AutonomousSnake.load_snake("populations/pop{}/gen{}.pickle".format(7008, 26))

pop1 = Population(2000, 100, training_config, snake1)
pop2 = Population(2000, 101, training_config, snake2)
pop3 = Population(2000, 102, training_config, snake3)
pop4 = Population(2000, 103, training_config, snake4)
pop5 = Population(2000, 104, training_config, snake5)
pop6 = Population(2000, 105, training_config, snake6)

populations = []
populations.append(pop1)
populations.append(pop2)
populations.append(pop3)
populations.append(pop4)
populations.append(pop5)
populations.append(pop6)

for pop in populations:
    pop.evolution()

for i in range(len(populations)):
    r = np.arange(6)
    np.delete(r, i)
    populations[i].introduce_from_outside([populations[j] for j in r])
    populations[i].reset_fitness()

for i in range(10):
    for pop in populations:
        pop.evolution()

for i in range(len(populations)):
    r = np.arange(6)
    np.delete(r, i)
    populations[i].introduce_from_outside([populations[j] for j in r])
    populations[i].reset_fitness()

for i in range(10):
    populations[i].evolution()
