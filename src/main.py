from generation import Generation


gen0 = Generation(0, 1000)
gen0.live()
gen0.save_champions(10)

prev_champs = gen0.get_champion_brains(25)

for i in range(1, 50):
    print("\nGeneration #{}".format(i))
    geni = Generation(i, 1000, prev_champs, n_crossover=0, mutation_rate=0.005)
    geni.live()
    geni.save_champions(25)
    prev_champs = geni.get_champion_brains(25)

