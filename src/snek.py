import pygame
import numpy as np

import neural_network


class Snek:

    def __init__(self, pos, length=3, base_max_moves=200, field_width=25, field_height=18):
        self.fieldWidth = field_width
        self.fieldHeight = field_height
        self.position = np.array([pos[0], pos[1]])
        self.velocity = np.array((0, 1))
        self.length = length

        self.directions = {0: np.array([0, -1]), 1: np.array([1, -1]), 2: np.array([1, 0]), 3: np.array([1, 1]),
                           4: np.array([0, 1]), 5: np.array([-1, 1]), 6: np.array([-1, 0]), 7: np.array([-1, -1])}

        self.tail = np.empty(shape=[0, 2])
        for i in range(self.length - 1, 0, -1):
            self.tail = np.append(self.tail, [[self.position[0], self.position[1] - i]], axis=0)

        self.alive = True
        self.movesLeft = base_max_moves

        self.growCnt = 0

        self.bodyImg = pygame.image.load("assets/snekpart.png")
        self.headImg = pygame.image.load("assets/snekhead.png")
        self.foodImg = pygame.image.load("assets/snekfood.png")

        self.lastMoveDir = np.array(self.velocity)
        self.food = self.place_food()

        self.vision = np.zeros(24)

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

    def render(self, display):
        for i in self.tail:
            display.blit(self.bodyImg, i * 32)
        display.blit(self.headImg, self.position * 32)
        display.blit(self.foodImg, self.food * 32)

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
        if distance == 1.0:
            print("VERY CLOSE")

        return vision


pygame.init()

displayWidth = 800
displayHeight = 576

display = pygame.display.set_mode((displayWidth, displayHeight))
pygame.display.set_caption("Snek")

clock = pygame.time.Clock()

snek = Snek([1, 3], 3)

lastTime = 0

paused = False
crashed = False
while not crashed:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            crashed = True
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP and not np.array_equal(np.array([0, 1]), snek.lastMoveDir):
                snek.velocity = np.array([0, -1])
            if event.key == pygame.K_DOWN and not np.array_equal(np.array([0, -1]), snek.lastMoveDir):
                snek.velocity = np.array([0, 1])
            if event.key == pygame.K_LEFT and not np.array_equal(np.array([1, 0]), snek.lastMoveDir):
                snek.velocity = np.array([-1, 0])
            if event.key == pygame.K_RIGHT and not np.array_equal(np.array([-1, 0]), snek.lastMoveDir):
                snek.velocity = np.array([1, 0])

            if event.key == pygame.K_SPACE:
                paused = not paused

    display.fill((0, 38, 76))

    snek.render(display)
    pygame.display.update()

    if lastTime >= 80 and not paused:
        snek.move()
        lastTime = 0

    clock.tick(60)
    lastTime += clock.get_time()

pygame.quit()
