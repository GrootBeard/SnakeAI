# from generation import Generation
from snake import AutonomousSnake
import pygame


def play(_snake):
    pygame.init()

    display_width = 800
    display_height = 576

    display = pygame.display.set_mode((display_width, display_height))
    pygame.display.set_caption("Snek")

    body_img = pygame.image.load("assets/snekpart.png")
    head_img = pygame.image.load("assets/snekhead.png")
    food_img = pygame.image.load("assets/snekfood.png")

    clock = pygame.time.Clock()

    last_time = 0

    turns = 0

    paused = False
    crashed = False
    while not crashed:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                crashed = True

        display.fill((0, 38, 76))

        for i in _snake.tail:
            display.blit(body_img, i * 32)
        display.blit(head_img, _snake.position * 32)
        display.blit(food_img, _snake.food * 32)

        pygame.display.update()

        if last_time >= 80 and not paused:
            _snake.think()
            _snake.move()
            last_time = 0
            turns += 1

        if not _snake.alive:
            crashed = True

        clock.tick(60)
        last_time += clock.get_time()

    print("Snake scored {} points in {} turns".format(_snake.score, turns))
    pygame.quit()


snake = AutonomousSnake.load_snake("champions/gen8.pickle")
print(snake.moves_left)
play(snake)
