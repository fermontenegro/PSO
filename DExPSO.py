import numpy as np
import math
import time

class Particle:
    def __init__(self, dim, values):
        self.position = np.random.uniform(values, values, dim)
        self.velocity = np.zeros(dim)
        self.best_position = self.position.copy()
        self.best_fitness = float('-inf')

    def update_best(self, function):
        fitness = function(self.position)
        if fitness > self.best_fitness:
            self.best_fitness = fitness
            self.best_position = self.position.copy()

class DExPSO:
    def __init__(self, function, dim, size, values, iterations):
        self.function = function
        self.dim = dim
        self.swarm = [Particle(dim, values) for _ in range(size)]
        self.global_best_position = np.random.uniform(values, values, dim)
        self.global_best_fitness = float('-inf')
        self.iterations = iterations

    def run(self):
        for i in range(self.iterations):
            for particle in self.swarm:
                particle.update_best(self.function)
                if particle.best_fitness > self.global_best_fitness:
                    self.global_best_fitness = particle.best_fitness
                    self.global_best_position = particle.best_position.copy()
                c1 = 1.49445
                c2 = 1.49445
                w = np.exp(-i / self.iterations)  # Exponential decay of inertia weight
                for d in range(self.dim):
                    r1 = np.random.rand()
                    r2 = np.random.rand()
                    s = -1 if np.random.rand() < 0.5 else 1
                    particle.velocity[d] = w * particle.velocity[d] + c1 * r1 * (particle.best_position[d] - particle.position[d]) + c2 * r2 * (self.global_best_position[d] - particle.position[d])
                    particle.position[d] += s * np.exp(-i / self.iterations) * particle.velocity[d]

        return self.global_best_position, self.global_best_fitness

def objective_function(x):
    y = x
    z = x
    v = x
    return (-5 - 2*math.exp(-x-2*y-2*z-2*v) - 2*math.exp(-2*x-2*y-z-2*v) +
    2*math.exp(-z-2*v-2*x) - 2*math.exp(-2*x-2*z-2*v-y) -
    2*math.exp(-z-2*v) + 2*math.exp(-2*y-z-2*v) +
    2*math.exp(-2*v-y-2*x) + 2*math.exp(-2*y-2*z-v) -
    2*math.exp(-2*x-2*y-2*z-v) + 2*math.exp(-2*y-x-2*v) +
    2*math.exp(-2*z-v-2*x ) + 3*math.exp(-2*x) +
    2*math.exp(-x) - 2*math.exp(-v-2*y) +
    2*math.exp(-2*y-2*x-v) + 2*math.exp(-2*z-2*v-y) -
    2*math.exp(-v-2*x) - 2*math.exp(-y-2*v) -
    math.exp(-2*z-2*v) - 2*math.exp(-2*z-v) -
    math.exp(-2*y-2*z-2*v) + 3*math.exp(-2*v) -
    math.exp(-2*v-2*y) + 2*math.exp(-2*y-x-2*z) +
    2*math.exp(-z) + 2*math.exp(-y) -
    2*math.exp(-2*y-x) - 2*math.exp(-x-2*z) +
    2*math.exp(-2*z-2*x-y) - 2*math.exp(-y-2*z) -
    2*math.exp(-2*x-y) + 3*math.exp(-2*z) +
    3*math.exp(-2*y) - 2*math.exp(-2*y-z) -
    2*math.exp(-2*v-x) -math.exp(-2*y-2*z) +
    2*math.exp(-2*x-2*y-z) + 2*math.exp(-x-2*z-2*v) +
    2*math.exp(-v) - 2*math.exp(-z-2*x) -
    math.exp(-2*x-2*z) - math.exp(-2*x-2*y ) -
    math.exp(-2*x-2*y-2*z) + 3*math.exp(-2*x-2*y-2*z-2*v) - 
    math.exp(-2*z-2*v-2*x) - math.exp(-2*y-2*x-2*v) - math.exp(-2*v-2*x))/(-1 + 
    math.exp(-2*x) - math.exp(-2*z-2*v) + math.exp(-2*y-2*z-2*v) +
    math.exp(-2*v) - math.exp(-2*v-2*y) + math.exp(-2*z) + math.exp(-2*y) -
    math.exp(-2*y-2*z) - math.exp(-2*x-2*z) - math.exp(-2*x-2*y) + math.exp(-2*x-2*y-2*z) -
    math.exp(-2*x-2*y-2*z-2*v) + math.exp(-2*z-2*v-2*x) + math.exp(-2*y-2*x-2*v) -
    math.exp(-2*v-2*x))


# Parámetros del algoritmo DExPSO
dim = 1
values = 0.1
size = 20
iterations = 1000

start_time = time.time()  
convergence_point = float('-inf')  # Punto de convergencia inicial

while True:
    pso = DExPSO(lambda x: -objective_function(x), 1, 20, values, 1000)
    best_max_position, best_max_fitness = pso.run()
    print("Mejor posición máxima encontrada:", best_max_position)
    print("Valor de función máxima encontrado:", -best_max_fitness)

    if convergence_point < -best_max_fitness:
        convergence_point = -best_max_fitness

    if values > 10:
        break
    else:
        values += 0.1

print("Punto de convergencia: ", convergence_point)
end_time = time.time()  # Guarda el tiempo de fin
elapsed_time = end_time - start_time  # Calcula el tiempo transcurrido
print("Tiempo de ejecución:", elapsed_time, "segundos")