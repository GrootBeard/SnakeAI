# from generation import Generation
from snake import AutonomousSnake
import pygame
import sys


def play(_snake):
    pygame.init()

    SCALE = 0.5

    display_width = int(800 * SCALE)
    display_height = int(800 * SCALE)

    display = pygame.display.set_mode((display_width, display_height))
    pygame.display.set_caption("Snek")

    body_img = pygame.image.load("assets/snekpart.png")
    head_img = pygame.image.load("assets/snekhead.png")
    food_img = pygame.image.load("assets/snekfood.png")
    body_img = pygame.transform.scale(body_img, (int(32 * SCALE), int(32 * SCALE)))
    head_img = pygame.transform.scale(head_img, (int(32 * SCALE), int(32 * SCALE)))
    food_img = pygame.transform.scale(food_img, (int(32 * SCALE), int(32 * SCALE)))

    clock = pygame.time.Clock()

    last_time = 0

    moves = 0

    paused = False
    crashed = False
    while not crashed:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                crashed = True

        display.fill((0, 38, 76))

        for i in _snake.tail:
            display.blit(body_img, i * 32 * SCALE)
        display.blit(head_img, _snake.position * 32 * SCALE)
        display.blit(food_img, _snake.food * 32 * SCALE)

        pygame.display.update()

        if last_time >= 80 and not paused:
            _snake.think()
            _snake.move()
            last_time = 0
            moves += 1

        if not _snake.alive:
            crashed = True

        clock.tick(60)
        last_time += clock.get_time()

    print("Snake scored {} points in {} moves".format(_snake.length - 5, moves))
    pygame.quit()


snake = AutonomousSnake.load_snake("champions/gen{}.pickle".format(sys.argv[1]))
# print(snake.moves_left)
print(snake.velocity)
play(snake)
