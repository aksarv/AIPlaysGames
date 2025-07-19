"""
It could be a lot worse lol, haven't actually run it for too long but presume it shouldn't take more than around 1000 epochs to clear this level
"""

import pygame
import numpy as np
import time

pygame.init()

scr_width, scr_height = 1000, 360
scr = pygame.display.set_mode((scr_width, scr_height))


class Player:
    def __init__(self):
        self.x = 98
        self.y = scr_height // 2 - 15


class Enemy:
    min_x = 200
    max_x = 175 + ((scr_width - 350) // 50 - 1) * 50 + 25

    def __init__(self, x, y, x_vel, y_vel):
        self.x = x
        self.y = y
        self.x_vel = x_vel
        self.y_vel = y_vel
        self.reward = 0

    def update(self, dt):
        self.x += self.x_vel * dt
        self.y += self.y_vel * dt

    def bound(self):
        if self.x > Enemy.max_x or self.x < Enemy.min_x:
            self.x_vel *= -1

        self.x = max(Enemy.min_x, min(self.x, Enemy.max_x))


def get_state(x, y):
    x_coord = (x - 50) // 25
    y_coord = (y - (scr_height // 2 - 100)) // 25
    return int(x_coord + y_coord * 25)


player = Player()
enemies = [Enemy(Enemy.min_x, scr_height // 2 - 75, 500, 0),
           Enemy(Enemy.max_x, scr_height // 2 - 25, 500, 0),
           Enemy(Enemy.min_x, scr_height // 2 + 25, 500, 0),
           Enemy(Enemy.max_x, scr_height // 2 + 75, 500, 0)]

STATES = 236
ACTIONS = 4

LR = 0.9
DISCOUNT = 0.95
PROB_EXPLORE = 0.5
PROB_EXPLORE_DECAY = 0.9995
PROB_EXPLORE_MIN = 0.01
MAX_EPOCHS = 1000

q_table = np.zeros((STATES, ACTIONS))

current_state = get_state(player.x, player.y)

start = time.time()

epoch = 1
reward = 0

run = True
while run:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False

    scr.fill((173, 173, 242))

    end = time.time()
    dt = end - start
    start = time.time()

    pygame.draw.rect(scr, (178, 255, 178), [50, scr_height // 2 - 100, 125, 200])
    pygame.draw.rect(scr, (178, 255, 178), [scr_width - 175, scr_height // 2 - 100, 125, 200])

    for k, i in enumerate(np.linspace(175, scr_width - 175, (scr_width - 350) // 50, endpoint=False)):
        for l, j in enumerate(np.linspace(scr_height // 2 - 100, scr_height // 2 + 100, (scr_height - 150) // 50, endpoint=False)):
            pygame.draw.rect(scr, (240, 240, 240) if (k + l) % 2 == 0 else (210, 210, 240), [i, j, 50, 50])

    pygame.draw.rect(scr, (255, 0, 0), [player.x, player.y, 30, 30])
    pygame.draw.rect(scr, (0, 0, 0), [player.x, player.y, 30, 30], 5)

    over = False
    for enemy in enemies:
        enemy.update(dt)
        enemy.bound()

        if pygame.Rect(enemy.x - 12, enemy.y - 12, 24, 24).colliderect(pygame.Rect(player.x - 15, player.y - 15, 30, 30)):
            over = True

        pygame.draw.circle(scr, (0, 0, 139), (enemy.x, enemy.y), 12)
        pygame.draw.circle(scr, (0, 0, 0), (enemy.x, enemy.y), 12, 5)

    if np.random.rand() < PROB_EXPLORE:
        action = np.random.randint(0, ACTIONS)
    else:
        action = np.argmax(q_table[current_state])

    prev_dist = (player.x - (scr_width - 112.5)) ** 2 + (player.y - scr_height // 2) ** 2

    if action == 0:
        player.x += 400 * dt
    elif action == 1:
        player.x -= 400 * dt
    elif action == 2:
        player.y += 400 * dt
    elif action == 3:
        player.y -= 400 * dt

    curr_dist = (player.x - (scr_width - 112.5)) ** 2 + (player.y - scr_height // 2) ** 2

    reward += prev_dist - curr_dist
    reward -= dt

    next_state = get_state(player.x, player.y)

    episode_over = False
    if pygame.Rect(scr_width - 175, scr_height // 2 - 100, 125, 200).colliderect(player.x - 15, player.y - 15, 30, 30):
        reward += 1e6
        episode_over = True
    elif over or player.x - 15 < 50 or player.x + 15 > scr_width - 50 or player.y - 15 < scr_height // 2 - 100 or player.y + 15 > scr_height // 2 + 100:
        reward -= 1e6
        episode_over = True

    q_table[current_state, action] += LR * (reward + DISCOUNT * np.max(q_table[next_state]) - q_table[current_state, action])

    current_state = next_state

    if episode_over:
        epoch += 1
        if epoch == MAX_EPOCHS:
            break
        reward = 0
        player.x = 98
        player.y = scr_height // 2 - 15
        current_state = get_state(player.x, player.y)
        PROB_EXPLORE = max(PROB_EXPLORE_MIN, PROB_EXPLORE * PROB_EXPLORE_DECAY)
        enemies = [Enemy(Enemy.min_x, scr_height // 2 - 75, 500, 0),
                   Enemy(Enemy.max_x, scr_height // 2 - 25, 500, 0),
                   Enemy(Enemy.min_x, scr_height // 2 + 25, 500, 0),
                   Enemy(Enemy.max_x, scr_height // 2 + 75, 500, 0)]

    scr.blit(pygame.font.SysFont(None, 32).render(f"Epoch {epoch}", True, (0, 0, 0)), (20, 20))

    pygame.display.update()

pygame.quit()
