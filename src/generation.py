from snake import AutonomousSnake


class Generation:

    def __init__(self, population_size, seed_snakes=None, mutation_rate=0):
        self.seed_snakes = seed_snakes
        self.population_size = population_size
        self.snakes = self.populate()
        self.active_snakes = self.snakes
        self.alive = True

    def populate(self, mutation_rate = 0):
        if not self.seed_snakes:
            return [AutonomousSnake([1, 3]) for i in range(self.population_size)]
        

    def mutate(self, mutation_rate):        
        // snake.neural_net.weights *= np.randint(0,1, size)*np.random(size)
        pass

    def step(self):
        for snake in self.snakes:
            snake.think()
            snake.move()
            if not snake.alive:
                self.active_snakes.remove(snake)
    
    def live(self):
        pass