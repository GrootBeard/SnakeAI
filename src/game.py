import pygame
import numpy as np

pygame.init()

display_width = 800
display_height = 600

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED =   (255, 0, 0)
GREEN = (0, 255, 0)
BLUE =  (0, 0, 255)

gameDisplay = pygame.display.set_mode((display_width , display_height))
pygame.display.set_caption("Snek")

clock = pygame.time.Clock()

carImg = pygame.image.load("car.png")

def car(x, y):
    gameDisplay.blit(carImg, (x, y))

x = 0
y = 0
dx = 0
dy = 0

speed = 5

crashed = False

keys_pressed = [False] * 4

while not crashed:
      
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            crashed = True
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                keys_pressed[0] = True
            if event.key == pygame.K_DOWN:
                keys_pressed[1] = True
            if event.key == pygame.K_LEFT:
                keys_pressed[2] = True
            if event.key == pygame.K_RIGHT:
                keys_pressed[3] = True
            
        if event.type == pygame.KEYUP:
            if event.key == pygame.K_UP:
                keys_pressed[0] = False
            if event.key == pygame.K_DOWN:
                keys_pressed[1] = False
            if event.key == pygame.K_LEFT:
                keys_pressed[2] = False
            if event.key ==pygame.K_RIGHT:
                keys_pressed[3] = False
            

    dx = 0
    dy = 0
    
    if keys_pressed[2]:
        dx -= speed
    if keys_pressed[3]:
        dx += speed
    if keys_pressed[0]:
        dy -= speed
    if keys_pressed[1]:
        dy += speed
    
    if dx*dx + dy*dy > speed*speed:
        dx /= np.sqrt(2)
        dy /= np.sqrt(2)
        print("diagonal")
        print(dx*dx+dy*dy)
        
    x += dx
    y += dy
        
    gameDisplay.fill((0, 38, 76))
    car(x, y)
    
    pygame.display.update()
    clock.tick(60)

pygame.quit()
