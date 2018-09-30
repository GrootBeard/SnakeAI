from population import Population
from snake import AutonomousSnake


snake = AutonomousSnake.load_snake("champions/gen7.pickle")
pop = Population(2000, 1, snake, 0.1, 0.05)
for i in range(200):
    pop.evolution()
