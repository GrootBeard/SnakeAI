from population import Population, TrainingConfiguration
from snake import AutonomousSnake


training_config = TrainingConfiguration(0.5, 0.25, 120)

# snake1 = AutonomousSnake.load_snake("populations/pop{}/gen{}.pickle".format(1, 2))
# snake2 = AutonomousSnake.load_snake("populations/pop{}/gen{}.pickle".format(2, 18))

# pop1 = Population(2000, 6006, training_config, snake1)
pop1 = Population(2000, 6008, training_config, None)
pop2 = Population(2000, 6007, training_config, None)


# snake = AutonomousSnake.load_snake("champions/gen7.pickle")
# pop = Population(2000, 1, snake, 0.1, 0.05)
for i in range(10):
    pop1.evolution()
    pop2.evolution()

pop1.introduce_from_outside([pop2])

for i in range(5):
    pop1.evolution()
    pop2.evolution()

pop2.introduce_from_outside([pop1])

for i in range(5):
    pop1.evolution()
    pop2.evolution()

pop2.introduce_from_outside([pop1])
pop1.introduce_from_outside([pop2])

for i in range(5):
    pop1.evolution()
    pop2.evolution()