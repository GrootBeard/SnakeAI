from population import Population


pop = Population(2000, None, 0.1, 0.05)
for i in range(200):
    pop.evolution()
