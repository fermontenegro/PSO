import numpy as np
import math
import time

r = 1

class Particle:
    def __init__(self):
        self.position = np.random.uniform(low=0, high=100, size=2)
        self.velocity = np.random.uniform(low=-1.0, high=1.0, size=2)
        self.best_position = np.copy(self.position)
        self.best_fitness = float('-inf')

    def update_best(self, objective_function):
        fitness = objective_function(self.position[0], self.position[1])
        if fitness > self.best_fitness:
            self.best_fitness = fitness
            self.best_position = np.copy(self.position)

    def update_velocity_and_position(self, global_best_position, w, c1, c2):
        r1 = np.random.rand(2)
        r2 = np.random.rand(2)
        self.velocity = (w * self.velocity) + (c1 * r1 * (self.best_position - self.position)) + (c2 * r2 * (global_best_position - self.position))
        self.position += self.velocity
        self.position = np.clip(self.position, -5.0, 5.0)

class PSO:
    def __init__(self, objective_function, n_particles, iterations):
        self.objective_function = objective_function
        self.swarm = [Particle() for _ in range(n_particles)]
        self.global_best_position = np.random.uniform(low=0, high=100, size=2)
        self.global_best_fitness = float('-inf')
        self.w = 0.5
        self.c1 = 1.5
        self.c2 = 1.5
        self.iterations = iterations

    def optimize(self):
        start_time = time.time()
        for i in range(self.iterations):
            for particle in self.swarm:
                particle.update_best(self.objective_function)
                if particle.best_fitness > self.global_best_fitness:
                    self.global_best_fitness = particle.best_fitness
                    self.global_best_position = np.copy(particle.best_position)

            for particle in self.swarm:
                particle.update_velocity_and_position(self.global_best_position, self.w, self.c1, self.c2)

        end_time = time.time()
        elapsed_time = end_time - start_time
        return self.global_best_position, self.global_best_fitness, elapsed_time

# Definir la función de prueba (puedes reemplazarla con tu propia función)
def test_function(x, y):
    x = r 
    y = r
    z = y
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


while True:

    pso = PSO(test_function, 20, 1000)
    best_position, best_fitness, execution_time = pso.optimize()
    r = r + 0.1
 
    # Imprimir resultados
    print("Mejor posición encontrada:", best_position)
    print("Valor de función encontrado:", best_fitness)

    if r > 1:
        break
    
print("Tiempo de ejecución:", execution_time, "segundos")
