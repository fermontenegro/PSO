import numpy as np
import math
import time

# Definir la función objetivo
def objective_function(x):
    y = x
    z = x
    v = x
    return (-5 - 2 * math.exp(-x-2*y-2*z-2*v) - 2 * math.exp(-2*x-2*y-z-2*v) + 2 * math.exp(-z-2*v-2*x) - 2 * math.exp(-2*x-2*z-2*v-y) - 2 * math.exp(-z-2*v) + 2 * math.exp(-2*y-z-2*v) + 2 * math.exp(-2*v-y-2*x) - 2 * math.exp(-2*y-2*z-v) - 2 * math.exp(-2*x-2*y-2*z-v) + 2 * math.exp(-2*y-x-2*v) + 2 * math.exp(-2*z-v-2*x) + 2 * math.exp(-x) - 2 * math.exp(-x) - 2 * math.exp(-v-2*y) + 2 * math.exp(-2*y-2*x-v) - 2 * math.exp(-2*z-2*v-y) - 2 * math.exp(-v-2*x) - math.exp(-y-2*v) - 2 * math.exp(-2*z-2*v) - math.exp(-2*z-v) - 2 * math.exp(-2*y-2*z-2*v) + 3 * math.exp(-2*v) - math.exp(-2*v-2*y) + 2 * math.exp(-2*y-x-2*z) - math.exp(-z) + 2 * math.exp(-y) - 2 * math.exp(-2*y-x) - 2 * math.exp(-x-2*z) - 2 * math.exp(-2*z-2*x-y) - 2 * math.exp(-y-2*z) - 2 * math.exp(-2*x-y) + 3 * math.exp(-2*z) - 2 * math.exp(-2*y) - 2 * math.exp(-2*y-z) - math.exp(-2*v-x) + 2 * math.exp(-2*y-2*z) + 2 * math.exp(-2*x-2*y-z-2*v) - math.exp(-v) - math.exp(-z-2*x) - math.exp(-2*x-2*z) - math.exp(-2*x-2*y) + 3 * math.exp(-2*x-2*y-2*z) - 2 * math.exp(-2*x-2*y-2*z-2*v) - math.exp(-2*z-2*v-2*x) - math.exp(-2*y-2*x-2*v) - 2 * math.exp(-2*v-2*x)) / (-1 + math.exp(-2*x) - math.exp(-2*z-2*v) + math.exp(-2*y-2*z-2*v) + math.exp(-2*v) - math.exp(-2*v-2*y) + math.exp(-2*z) - math.exp(-2*y) - math.exp(-2*y-2*z) - math.exp(-2*x-2*z) + math.exp(-2*x-2*y) - math.exp(-2*x-2*y-2*z) - math.exp(-2*x-2*y-2*z-2*v) + math.exp(-2*z-2*v-2*x) - math.exp(-2*y-2*x-2*v) - math.exp(-2*v-2*x))

class Particle:
    def __init__(self, dim, values):
        self.position = np.array([np.random.choice(values) for _ in range(dim)])
        self.velocity = np.zeros(dim)
        self.best_position = self.position.copy()
        self.best_fitness = float('inf')

    def update_best(self, function):
        fitness = function(self.position)
        if fitness < self.best_fitness:
            self.best_fitness = fitness
            self.best_position = self.position.copy()

class PSO:
    def __init__(self, function, dim, size, values, iterations):
        self.function = function
        self.dim = dim
        self.swarm = [Particle(dim, values) for _ in range(size)]
        self.global_best_position = np.array([np.random.choice(values) for _ in range(dim)])
        self.global_best_fitness = float('inf')
        self.iterations = iterations

    def run(self):
        start_time = time.time()  # Guarda el tiempo de inicio
        for i in range(self.iterations):
            for particle in self.swarm:
                particle.update_best(self.function)
                if particle.best_fitness < self.global_best_fitness:
                    self.global_best_fitness = particle.best_fitness
                    self.global_best_position = particle.best_position.copy()
                w = 0.729  # Inertial weight
                c1 = 1.49445  # Cognitive weight
                c2 = 1.49445  # Social weight
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)
                particle.velocity = (w * particle.velocity) + (c1 * r1 * (particle.best_position - particle.position)) + (c2 * r2 * (self.global_best_position - particle.position))
                particle.position += particle.velocity
                particle.position = np.array([np.clip(val, min(values), max(values)) for val in particle.position])

        end_time = time.time()  # Guarda el tiempo de fin
        elapsed_time = end_time - start_time  # Calcula el tiempo transcurrido

        return self.global_best_position, self.global_best_fitness, elapsed_time

# Ejemplo de uso
values = [0.0, 1.0, 2.0]  # Valores posibles para x
pso = PSO(objective_function, 1, 20, values, 100)
best_min_position, best_min_fitness, elapsed_time = pso.run()
print("Mejor posición mínima encontrada:", best_min_position)
print("Mejor valor de aptitud mínima encontrado:", best_min_fitness)
print("Tiempo de ejecución:", elapsed_time, "segundos")

pso = PSO(lambda x: -objective_function(x), 1, 20, values, 100)
best_max_position, best_max_fitness, elapsed_time = pso.run()
print("Mejor posición máxima encontrada:", best_max_position)
print("Mejor valor de aptitud máxima encontrado:", -best_max_fitness)
print("Tiempo de ejecución:", elapsed_time, "segundos")