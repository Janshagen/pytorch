from Snake import Snake
from Food import Food
import pygame

w = 30
gridsize = 20

pygame.init()
screen = pygame.display.set_mode((gridsize * w + 1, gridsize * w + 1))
font = pygame.font.Font(None, 72)

start_pos = [10, 10]
start_vel = [0, -1]

snake = Snake(start_pos, w, start_vel, (255, 0, 0), gridsize)
food = Food([0, 0], w, (0, 255, 0), gridsize)
food.add_food(snake)


def draw_grid():
    for line in range(gridsize + 1):
        pygame.draw.line(screen, (255, 255, 255), (line * w, 0), (line * w, gridsize * w))
        pygame.draw.line(screen, (255, 255, 255), (0, line * w), (gridsize * w, line * w))


hej = 0

def draw():
    screen.fill((0, 0, 0))
    draw_grid()
    food.show(screen)
    snake.show(screen)
    pygame.display.flip()


clock = pygame.time.Clock()
running = True

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    pygame.time.delay(100)
    clock.tick(6)

    if not snake.crash():
        snake.eat(food)
        snake.update()
        snake.turn()
        draw()
    else:
        pygame.draw.rect(screen, (0, 0, 255), (snake.body[0][0] * w + 1, snake.body[0][1] * w + 1, w - 1, w - 1))
        score = font.render('Score: ' + str(snake.len - 2), True, (255, 255, 255))
        screen.blit(score, (210, 138))
        pygame.display.flip()
        for event in pygame.event.get(pygame.KEYDOWN):
            if event.key == pygame.K_SPACE:
                snake = Snake(start_pos, w, start_vel, (255, 0, 0), gridsize)
                food.add_food(snake)