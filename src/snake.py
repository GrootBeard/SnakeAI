import numpy as np
import pickle
import os
import math
import copy

from neural_network import Network


class AutonomousSnake:
    directions = {0: np.array([0, -1]), 1: np.array([1, -1]), 2: np.array([1, 0]), 3: np.array([1, 1]),
                  4: np.array([0, 1]), 5: np.array([-1, 1]), 6: np.array([-1, 0]), 7: np.array([-1, -1])}

    def __init__(self, pos=[13, 13], length=5, base_max_moves=100, field_width=25, field_height=25):
        self.field_width = field_width
        self.field_height = field_height
        self.position = np.array([pos[0], pos[1]])
        self.velocity = self.directions[np.random.choice([0, 2, 4, 6])]
        self.length = length
        self.score = 0

        self.brain = Network([24, 18, 4])

        self.tail = np.empty(shape=[0, 2])
        for i in range(self.length - 1, 0, -1):
            self.tail = np.append(self.tail, [self.position - i * self.velocity], axis=0)

        self.alive = True
        self.time_alive = 0

        self.max_moves = base_max_moves
        self.moves_left = self.max_moves

        self.grow_count = 0

        self.lastMoveDir = np.array(self.velocity)
        self.food = self.place_food()

        self.vision = np.zeros(24)
        self.fitness = 0

    def think(self):
        self.see()
        decision = self.brain.feed_forward(self.vision.reshape((24, 1)))
        # decision_arg = np.argmax(decision)
        chosen_direction = self.directions[np.argmax(decision) * 2]
        if not np.array_equal(chosen_direction, -self.velocity):
            self.velocity = chosen_direction
        # self.velocity = self.directions[decision_arg * 2]  # if not np.array_equal(self.velocity, -self.directions[
        #     decision_arg * 2]) else self.velocity

    def move(self):

        if self.will_collide():
            self.alive = False

        if not self.alive:
            return

        self.tail = np.append(self.tail, [self.position], axis=0)
        self.position += self.velocity
        if self.grow_count == 0:
            self.tail = np.delete(self.tail, 0, 0)
        else:
            self.grow_count -= 1
            self.length += 1

        self.lastMoveDir = self.velocity

        if np.array_equal(self.position, self.food):
            self.eat()

        self.time_alive += 1
        self.moves_left -= 1

        if self.moves_left <= 0:
            self.alive = False

    def eat(self):
        self.food = self.place_food()
        self.grow_count += 1
        self.score += 1
        self.moves_left = self.max_moves

    def place_food(self):
        pos = np.array([np.random.randint(0, self.field_width), np.random.randint(0, self.field_height)])
        while self.occupied(pos):
            pos = np.array([np.random.randint(0, self.field_width), np.random.randint(0, self.field_height)])

        return pos

    def calc_fitness(self):
        if self.score < 10:
            self.fitness = math.floor(self.time_alive ** 2 * 2 ** self.score)
        else:
            self.fitness = self.time_alive ** 2 * 2 ** 10 * (self.score - 9)

    # Check if the position is occupied by the body of the snake
    def occupied(self, pos):
        for cell in self.tail:
            if np.array_equal(cell, pos):
                return True
        return np.array_equal(self.position, pos)

    def will_collide(self):
        next_position = self.position + self.velocity
        if self.is_on_tail(next_position):
            return True
        return (next_position[0] < 0 or next_position[0] >= self.field_width or next_position[1] < 0 or next_position[
            1] >= self.field_height)

    def is_on_tail(self, pos):
        for cell in self.tail:
            if np.array_equal(pos, cell):
                return True
        return False

    def see(self):
        self.vision = np.array([])

        for d in self.directions:
            d_vision = self.look_in_direction(self.directions[d])
            self.vision = np.append(self.vision, d_vision)

    def look_in_direction(self, direction):
        cur_pos = self.position + direction

        vision = np.zeros(3)
        food_found = False
        tail_found = False

        distance = 1.0

        while not (
                cur_pos[0] < 0 or cur_pos[0] >= self.field_width or cur_pos[1] < 0 or cur_pos[1] >= self.field_height):
            if not food_found and np.array_equal(self.food, cur_pos):
                vision[0] = 1.0
                food_found = True

            if not tail_found and self.is_on_tail(cur_pos):
                vision[1] = 1.0 / distance
                tail_found = True

            cur_pos = cur_pos + direction
            distance += 1.0

        vision[2] = 1.0 / distance

        return vision

    def crossover_brain(self, other):
        new_snake = AutonomousSnake()
        if self.fitness > other.fitness:
            new_snake.brain = self.brain.crossover(other.brain)
        else:
            new_snake.brain = other.brain.crossover(self.brain)
        return new_snake

    def mutate(self, rate, mag):
        self.brain.mutate(rate, mag)

    def reincarnate(self):
        self.position = np.array((13, 13))
        self.velocity = self.directions[np.random.choice([0, 2, 4, 6])]
        self.length = 5
        self.alive = True
        self.time_alive = 0
        self.grow_count = 0
        self.tail = np.empty(shape=[0, 2])
        for i in range(self.length - 1, 0, -1):
            self.tail = np.append(self.tail, [self.position - i * self.velocity], axis=0)
        return self

    def save(self, population, generation):
        brain_data = {"sizes": self.brain.sizes,
                      "weights": self.brain.weights,
                      "biases": self.brain.biases}

        data = {"generation": generation,
                # "id": self.sid,
                "fitness": self.fitness,
                "brain": brain_data}
        try:
            pickle_out = open("populations/pop{}/gen{}.pickle".format(population, generation), "wb")
        except Exception:
            os.mkdir("populations/pop{}".format(population))
            pickle_out = open("populations/pop{}/gen{}.pickle".format(population, generation), "wb")
        pickle_out = open("populations/pop{}/gen{}.pickle".format(population, generation), "wb")
        pickle.dump(data, pickle_out)
        pickle_out.close()

    @classmethod
    def load_snake(cls, filename):
        pickle_in = open(filename, 'rb')
        data = pickle.load(pickle_in)
        pickle_in.close()
        brain = Network(data["brain"]["sizes"])
        brain.weights = [np.array(w) for w in data["brain"]["weights"]]
        brain.biases = [np.array(b) for b in data["brain"]["biases"]]

        snake = AutonomousSnake()
        snake.fitness = data["fitness"]
        snake.brain = brain
        return snake

    def __copy__(self):
        snake_copy = AutonomousSnake()
        snake_copy.brain = copy.copy(self.brain)
        return snake_copy
