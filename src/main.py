from population import Population


pop = Population(1000, None, 0.05, 0.05)
for i in range(10):
    pop.evolution()
