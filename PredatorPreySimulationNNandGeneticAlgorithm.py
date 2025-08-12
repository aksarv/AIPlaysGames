"""
Doesn't really work but whatever
"""

import numpy as np
import random
import math
import pygame
import time
import copy

pygame.init()

ENTITY_SIZE = 30
N_RAYS = 20
PREY_FOV = 150
PREDATOR_FOV = 60
PREY_FOV, PREDATOR_FOV = math.radians(PREY_FOV), math.radians(PREDATOR_FOV)

PREY_ANG = math.radians(30)
PRED_ANG = math.radians(20)

FONT = pygame.font.SysFont(None, 24)


class NeuralNetwork:
    def __init__(self, i, h, o):
        self.i = i
        self.h = h
        self.o = o

    def initialise(self):
        self.wih = np.random.randn(self.h, self.i)
        self.who = np.random.randn(self.o, self.h)
        self.bih = np.random.randn(self.h)
        self.bho = np.random.randn(self.o)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self, i):
        hidden = self.sigmoid(np.matmul(self.wih, i) + self.bih)
        output = self.sigmoid(np.matmul(self.who, hidden) + self.bho)
        return output


def crossover(mat1, mat2):
    if random.random() > 0.1:
        x = np.random.randint(1, mat1.shape[0])
        tmp = mat2[:x].copy()
        mat2[:x], mat1[:x] = mat1[:x], tmp
    return mat1, mat2


def mutate(mat):
    if len(mat.shape) == 2:
        return np.where(random.random() < 1 / (mat.shape[0] * mat.shape[1]), mat, random.gauss(0, 1))
    elif len(mat.shape) == 1:
        return np.where(random.random() < 1 / mat.size, mat, random.gauss(0, 1))


def collide(x1, y1, x2, y2, r1, r2):
    if math.hypot(x2 - x1, y2 - y1) < r1 + r2:
        return True
    return False


def placement(e_list, e):
    for en in e_list:
        if collide(en.x, en.y, e.x, e.y, ENTITY_SIZE, ENTITY_SIZE):
            return True
    return False


def arctan(x1, y1, x2, y2):
    delta_x = x2 - x1
    delta_y = y2 - y1
    return math.atan2(delta_y, delta_x)


def sign(x):
    return -1 if x < 0 else 1


def delta_angle(c, t):
    if c - t < 0:
        return c - t + math.pi * 2
    return c - t


class Entity:
    def __init__(self, x, y, angle):
        self.x = x
        self.y = y
        self.angle = angle
        self.speed = 30
        self.d_angle = 60
        self.net = NeuralNetwork(N_RAYS, 3, 4)
        self.net.initialise()

    def forward(self, dist):
        self.x += dist * math.cos(self.angle)
        self.y += dist * math.sin(self.angle)

    def backward(self, dist):
        self.x -= dist * math.cos(self.angle)
        self.y -= dist * math.sin(self.angle)

    def left(self, deg):
        self.angle += math.radians(deg)

    def right(self, deg):
        self.angle -= math.radians(deg)

    def bound(self):
        self.x = max(0, min(scr_width, self.x))
        self.y = max(0, min(scr_height, self.y))


class Prey(Entity):
    def __init__(self, x, y, angle):
        super().__init__(x, y, angle)
        self.fitness = 0


class Predator(Entity):
    def __init__(self, x, y, angle):
        super().__init__(x, y, angle)


scr_width, scr_height = 720, 480
scr = pygame.display.set_mode((scr_width, scr_height))

entities = [Predator(random.randint(ENTITY_SIZE, scr_width - ENTITY_SIZE),
                     random.randint(ENTITY_SIZE, scr_height - ENTITY_SIZE), random.uniform(0, 2 * math.pi))]
for _ in range(14):
    pred_to_add = Predator(random.randint(ENTITY_SIZE, scr_width - ENTITY_SIZE),
                           random.randint(ENTITY_SIZE, scr_height - ENTITY_SIZE), random.uniform(0, 2 * math.pi))
    while placement(entities, pred_to_add):
        pred_to_add = Predator(random.randint(ENTITY_SIZE, scr_width - ENTITY_SIZE),
                               random.randint(ENTITY_SIZE, scr_height - ENTITY_SIZE), random.uniform(0, 2 * math.pi))
    entities.append(pred_to_add)
    prey_to_add = Prey(random.randint(ENTITY_SIZE, scr_width - ENTITY_SIZE),
                       random.randint(ENTITY_SIZE, scr_height - ENTITY_SIZE), random.uniform(0, 2 * math.pi))
    while placement(entities, prey_to_add):
        prey_to_add = Prey(random.randint(ENTITY_SIZE, scr_width - ENTITY_SIZE),
                           random.randint(ENTITY_SIZE, scr_height - ENTITY_SIZE), random.uniform(0, 2 * math.pi))
    entities.append(prey_to_add)

prey = [e for e in entities if isinstance(e, Prey)]
prey_copy = copy.copy(prey)
predators = [e for e in entities if isinstance(e, Predator)]

generation = 1
overall_max_fitness = 0

start = time.time()

run = True

while run:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False

    scr.fill((0, 0, 0))

    end = time.time()
    dt = end - start
    start = time.time()

    for ind, p in enumerate(prey):
        inp = [0 for _ in range(N_RAYS)]
        rays = list(map(lambda l: round(l, 5), np.linspace(p.angle - PREY_FOV / 2, p.angle + PREY_FOV / 2, N_RAYS)))
        for pr_i, pr in enumerate(predators):
            dist = math.hypot(pr.x - p.x, pr.y - p.y)
            r = math.atan(ENTITY_SIZE / dist)
            left = arctan(p.x, p.y, pr.x, pr.y) + r
            right = arctan(p.x, p.y, pr.x, pr.y) - r
            between = list(filter(lambda a: left <= a <= right, rays))
            if between:
                b = between[0]
                inp[rays.index(b)] = dist

        out = np.argmax(p.net.forward(np.array(inp)))
        if out == 0:
            p.forward(p.speed * dt)
        elif out == 1:
            p.left(p.d_angle * dt)
        elif out == 2:
            p.right(p.d_angle * dt)
        elif out == 3:
            p.backward(p.speed * dt)

        p.bound()

        if ind == 0:
            for ray in rays:
                pygame.draw.line(scr, (255, 255, 255), (p.x, p.y),
                                 (p.x + math.cos(ray) * 100, p.y + math.sin(ray) * 100))

        pygame.draw.circle(scr, (0, 255, 0), (p.x, p.y), ENTITY_SIZE)
        pygame.draw.circle(scr, (0, 0, 0), (p.x, p.y), ENTITY_SIZE, 3)
        pygame.draw.circle(scr, (255, 255, 255), (p.x + math.cos(p.angle - PREY_ANG) * (ENTITY_SIZE / 7 * 6),
                                                  p.y + math.sin(p.angle - PREY_ANG) * (ENTITY_SIZE / 7 * 6)),
                           ENTITY_SIZE / 4)
        pygame.draw.circle(scr, (255, 255, 255), (p.x + math.cos(p.angle + PREY_ANG) * (ENTITY_SIZE / 7 * 6),
                                                  p.y + math.sin(p.angle + PREY_ANG) * (ENTITY_SIZE / 7 * 6)),
                           ENTITY_SIZE / 4)
        pygame.draw.circle(scr, (0, 0, 0), (p.x + math.cos(p.angle - PREY_ANG) * (ENTITY_SIZE / 7 * 6 + 2),
                                            p.y + math.sin(p.angle - PREY_ANG) * (ENTITY_SIZE / 7 * 6 + 2)),
                           ENTITY_SIZE / 5)
        pygame.draw.circle(scr, (0, 0, 0), (p.x + math.cos(p.angle + PREY_ANG) * (ENTITY_SIZE / 7 * 6 + 2),
                                            p.y + math.sin(p.angle + PREY_ANG) * (ENTITY_SIZE / 7 * 6 + 2)),
                           ENTITY_SIZE / 5)

    # predators will always move closer to the nearest prey, and to do this optimally they'll rotate towards them as well as moving in their direction
    for pr in predators:
        try:
            best = min(prey, key=lambda l: math.hypot(l.y - pr.y, l.x - pr.x))
        except ValueError:
            prey_copy.sort(key=lambda l: l.fitness, reverse=True)
            top = prey_copy[:10]
            new_prey_population = []
            for i in range(0, 10, 2):
                this = top[i]
                next = top[i + 1]
                new_wih = crossover(this.net.wih, next.net.wih)
                new_who = crossover(this.net.who, next.net.who)
                new_bih = crossover(this.net.bih, next.net.bih)
                new_bho = crossover(this.net.bho, next.net.bho)
                new_prey_1 = Prey(random.randint(ENTITY_SIZE, scr_width - ENTITY_SIZE), random.randint(ENTITY_SIZE, scr_height - ENTITY_SIZE), random.uniform(0, 2 * math.pi))
                new_prey_1.net.wih = mutate(new_wih[0])
                new_prey_1.net.who = mutate(new_who[0])
                new_prey_1.net.bih = mutate(new_bih[0])
                new_prey_1.net.bho = mutate(new_bho[0])
                new_prey_2 = Prey(random.randint(ENTITY_SIZE, scr_width - ENTITY_SIZE), random.randint(ENTITY_SIZE, scr_height - ENTITY_SIZE), random.uniform(0, 2 * math.pi))
                new_prey_2.net.wih = mutate(new_wih[1])
                new_prey_2.net.who = mutate(new_who[1])
                new_prey_2.net.bih = mutate(new_bih[1])
                new_prey_2.net.bho = mutate(new_bho[1])
                new_prey_population.append(new_prey_1)
                new_prey_population.append(new_prey_2)
            for _ in range(10):
                new_prey_population.append(Prey(random.randint(ENTITY_SIZE, scr_width - ENTITY_SIZE), random.randint(ENTITY_SIZE, scr_height - ENTITY_SIZE), random.uniform(0, 2 * math.pi)))
            for _ in range(15):
                pred_to_add = Predator(random.randint(ENTITY_SIZE, scr_width - ENTITY_SIZE),
                                       random.randint(ENTITY_SIZE, scr_height - ENTITY_SIZE),
                                       random.uniform(0, 2 * math.pi))
                while placement(new_prey_population, pred_to_add):
                    pred_to_add = Predator(random.randint(ENTITY_SIZE, scr_width - ENTITY_SIZE),
                                           random.randint(ENTITY_SIZE, scr_height - ENTITY_SIZE),
                                           random.uniform(0, 2 * math.pi))
                new_prey_population.append(pred_to_add)
            entities = new_prey_population
            generation += 1
            curr_max = max(prey_copy, key=lambda l: l.fitness).fitness
            if curr_max > overall_max_fitness:
                overall_max_fitness = curr_max
            prey = [e for e in entities if isinstance(e, Prey)]
            prey_copy = copy.copy(prey)
            predators = [e for e in entities if isinstance(e, Predator)]

        angle = arctan(pr.x, pr.y, best.x, best.y) + math.pi
        curr = pr.angle
        if delta_angle(curr, angle) < math.pi:
            pr.left(pr.d_angle * dt)
        else:
            pr.right(pr.d_angle * dt)

        pr.forward(pr.speed * dt)
        pr.bound()

        new_prey = []
        for p in prey:
            p.fitness += dt
            if not collide(p.x, p.y, pr.x, pr.y, ENTITY_SIZE, ENTITY_SIZE):
                new_prey.append(p)
            prey_copy[prey_copy.index(p)].fitness += dt

        prey = copy.copy(new_prey)

        pygame.draw.circle(scr, (255, 0, 0), (pr.x, pr.y), ENTITY_SIZE)
        pygame.draw.circle(scr, (0, 0, 0), (pr.x, pr.y), ENTITY_SIZE, 3)
        pygame.draw.circle(scr, (255, 255, 255), (
            pr.x + math.cos(pr.angle - PRED_ANG) * (ENTITY_SIZE / 7 * 6),
            pr.y + math.sin(pr.angle - PRED_ANG) * (ENTITY_SIZE / 7 * 6)),
                           ENTITY_SIZE / 4)
        pygame.draw.circle(scr, (255, 255, 255), (
            pr.x + math.cos(pr.angle + PRED_ANG) * (ENTITY_SIZE / 7 * 6),
            pr.y + math.sin(pr.angle + PRED_ANG) * (ENTITY_SIZE / 7 * 6)),
                           ENTITY_SIZE / 4)
        pygame.draw.circle(scr, (0, 0, 0), (pr.x + math.cos(pr.angle - PRED_ANG) * (ENTITY_SIZE / 7 * 6 + 2),
                                            pr.y + math.sin(pr.angle - PRED_ANG) * (ENTITY_SIZE / 7 * 6 + 2)),
                           ENTITY_SIZE / 5)
        pygame.draw.circle(scr, (0, 0, 0), (pr.x + math.cos(pr.angle + PRED_ANG) * (ENTITY_SIZE / 7 * 6 + 2),
                                            pr.y + math.sin(pr.angle + PRED_ANG) * (ENTITY_SIZE / 7 * 6 + 2)),
                           ENTITY_SIZE / 5)

    scr.blit(FONT.render(f"Generation {generation}", True, (255, 255, 255)), (20, 20))
    scr.blit(FONT.render(f"Current max fitness: {round(max(prey_copy, key=lambda l: l.fitness).fitness, 1)}", True, (255, 255, 255)), (20, 40))
    scr.blit(FONT.render(f"Overall max fitness: {round(overall_max_fitness, 1)}", True, (255, 255, 255)), (20, 60))

    pygame.display.update()

pygame.quit()
