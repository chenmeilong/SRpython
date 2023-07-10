
import pygame
pygame.init()
screen = pygame.display.set_mode([500,600])
screen.fill([255,255,255])
pygame.draw.circle(screen,[255,0,0],[100,100],30,0) #颜色、坐标、半径、是否填充
pygame.display.flip()
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
