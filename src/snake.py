import numpy as np
import pickle

import neural_network


class AutonomousSnake:

    def __init__(self, pos, length=3, base_max_moves=200, field_width=25, field_height=18):
        self.fieldWidth = field_width
        self.fieldHeight = field_height
        self.position = np.array([pos[0], pos[1]])
        self.velocity = np.array((0, 1))
        self.length = length

        self.neural_net = NeuralNetwork.Network([24, 18, 4])

        self.directions = {0: np.array([0, -1]), 1: np.array([1, -1]), 2: np.array([1, 0]), 3: np.array([1, 1]),
                           4: np.array([0, 1]), 5: np.array([-1, 1]), 6: np.array([-1, 0]), 7: np.array([-1, -1])}

        self.tail = np.empty(shape=[0, 2])
        for i in range(self.length - 1, 0, -1):
            self.tail = np.append(self.tail, [[self.position[0], self.position[1] - i]], axis=0)

        self.alive = True
        self.movesLeft = base_max_moves

        self.growCnt = 0

        self.lastMoveDir = np.array(self.velocity)
        self.food = self.place_food()

        self.vision = np.zeros(24)

    def think(self):
        decision = self.neural_net.feed_forward(self.see())
        decision = np.argmax(decision)
        self.velocity = self.directions[2 * decision]

    def move(self):

        if self.will_collide():
            self.alive = False

        if not self.alive:
            return

        self.tail = np.append(self.tail, [self.position], axis=0)
        self.position += self.velocity
        if self.growCnt == 0:
            self.tail = np.delete(self.tail, 0, 0)
        else:
            self.growCnt -= 1
            self.length += 1

        self.lastMoveDir = self.velocity

        if np.array_equal(self.position, self.food):
            self.eat()

        self.see()
        print(self.vision)

    def eat(self):
        self.food = self.place_food()
        self.growCnt += 1

    def place_food(self):
        pos = np.array([np.random.randint(0, self.fieldWidth), np.random.randint(0, self.fieldHeight)])
        while self.occupied(pos):
            pos = np.array([np.random.randint(0, self.fieldWidth), np.random.randint(0, self.fieldHeight)])

        return pos

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
        return (next_position[0] < 0 or next_position[0] >= self.fieldWidth or next_position[1] < 0 or next_position[
            1] >= self.fieldHeight)

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

        while not (cur_pos[0] < 0 or cur_pos[0] >= self.fieldWidth or cur_pos[1] < 0 or cur_pos[1] >= self.fieldHeight):
            if not food_found and np.array_equal(self.food, cur_pos):
                vision[0] = 1
                food_found = True

            if not tail_found and self.is_on_tail(cur_pos):
                vision[1] = 1.0 / distance
                tail_found = True

            cur_pos = cur_pos + direction
            distance += 1.0

        vision[2] = 1.0 / distance

        return vision

    def save_snake(self, generation, snake_id, score):
        brain_data = {"sizes": self.neural_net.sizes,
                      "weights": self.neural_net.weights,
                      "biases": self.neural_net.biases}

        data = {"generation": generation,
                "id": snake_id,
                "score": score,
                "brain": brain_data}

        pickle_out = open("generation {}/{}.pickle".format(generation, snake_id), "wb")
        pickle.dump(data, pickle_out)
        pickle_out.close()

    def load_snake(self, file_name):
        return
