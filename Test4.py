import numpy as np
import math
import time

class Particle:
    def __init__(self, x, y):
        self.position = np.array([x, y])
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
        self.position = np.clip(self.position, 0.0, 5.0)

class PSO:
    def __init__(self, objective_function, n_particles, iterations):
        self.objective_function = objective_function
        self.swarm = [Particle(x, y) for x in np.arange(0.1, 5.1, 0.1) for y in np.arange(0.1, 5.1, 0.1)]
        self.global_best_position = np.random.uniform(low=0, high=5, size=2)
        self.global_best_fitness = float('-inf')
        self.w = 0.5
        self.c1 = 1.5
        self.c2 = 1.5
        self.iterations = iterations

    def optimize(self):
        for particle in self.swarm:
            particle.update_best(self.objective_function)
            if particle.best_fitness > self.global_best_fitness:
                self.global_best_fitness = particle.best_fitness
                self.global_best_position = np.copy(particle.best_position)

        for _ in range(self.iterations):
            for particle in self.swarm:
                particle.update_velocity_and_position(self.global_best_position, self.w, self.c1, self.c2)

        return self.global_best_position, self.global_best_fitness

def test_function(x, y):
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


best_positions = []
best_fitnesses = []

# Crear una instancia de PSO y ejecutar la optimación
for x in np.arange(0.1, 5.1, 0.1):
    for y in np.arange(0.1, 5.1, 0.1):
        pso = PSO(lambda x, y: test_function(x, y), 20, 1000)
        best_position, best_fitness = pso.optimize()
        best_positions.append(best_position)
        best_fitnesses.append(best_fitness)

for i in range(len(best_positions)):
    print(f"Iteración {i+1}: Mejor posición encontrada: {best_positions[i]}, Valor de función encontrado: {best_fitnesses[i]}")


# Para los resultados finales de la última iteración
print("Final: Mejor posición encontrada:", best_positions[-1])
print("Final: Valor de función encontrado:", best_fitnesses[-1])