from population import Population, TrainingConfiguration
from snake import AutonomousSnake


training_config = TrainingConfiguration(0.3, 0.25, 120)

snake1 = AutonomousSnake.load_snake("population/pop{}/gen{}".format(1, 2))
snake2 = AutonomousSnake.load_snake("population/pop{}/gen{}".format(2, 18))

pop1 = Population(2000, 6006, training_config, snake1)
pop2 = Population(2000, 6007, training_config, snake2)

pop1.introduce_from_outside([pop2])

# snake = AutonomousSnake.load_snake("champions/gen7.pickle")
# pop = Population(2000, 1, snake, 0.1, 0.05)
for i in range(200):
    pop1.evolution()
