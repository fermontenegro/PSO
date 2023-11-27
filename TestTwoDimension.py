import numpy as np
import math
import time

def objective_function(x, y):
    # x e y se usan en lugar de la única variable x del ejemplo original
    z = np.random.uniform(2, 3)  
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

class Particle:
    def __init__(self, dim, values):
        # dim deberá ser 2 para trabajar en 2 dimensiones.
        self.position = np.full(dim, values)
        self.velocity = np.zeros(dim)
        self.best_position = np.copy(self.position)
        self.best_fitness = float('-inf')

    def update_best(self, function):
        # Ahora pasamos la posición de la partícula x e y como argumentos a la función objetivo
        fitness = function(self.position[0], self.position[1])
        if fitness > self.best_fitness:
            self.best_fitness = fitness
            self.best_position = np.copy(self.position)
    
    def update_velocity_and_position(self, global_best_position, w, c1, c2):
        r1 = np.random.rand(self.position.shape[0])
        r2 = np.random.rand(self.position.shape[0])
        self.velocity = (w * self.velocity) + (c1 * r1 * (self.best_position - self.position)) + (c2 * r2 * (global_best_position - self.position))
        self.position += self.velocit

class PSO:
    def __init__(self, objective_function, dim, n_particles, min_values, max_values, iterations):
        w = 0.729  # Inertial weight
        c1 = 1.49445  # Cognitive weight
        c2 = 1.49445  # Social weight
        self.objective_function = objective_function
        self.dim = dim
        self.swarm = [Particle(dim, min_values, max_values) for _ in range(n_particles)]
        self.global_best_position = np.random.uniform(low=min_values, high=max_values, size=dim)
        self.global_best_fitness = float('inf')
        self.w = w 
        self.c1 = c1
        self.c2 = c2
        self.iterations = iterations

    def optimize(self):
        for i in range(self.iterations):
            for particle in self.swarm:
                fitness = self.objective_function(particle.position)

                if fitness < particle.best_fitness:
                    particle.best_position = particle.position.copy()
                    particle.best_fitness = fitness

                if fitness < self.global_best_fitness:
                    self.global_best_position = particle.position.copy()
                    self.global_best_fitness = fitness

            for particle in self.swarm:
                particle.update_velocity_and_position(self.global_best_position, self.w, self.c1, self.c2)

        return self.global_best_position, self.global_best_fitness

start_time = time.time()  
values = 0.1
convergence_point = 0

while True:
    # Observe que ahora estamos pasando lambda x, y: -objective_function(x, y) en lugar de solo x
    pso = PSO(lambda x, y: -objective_function(x, y), 2, 20, 0, 5, 1000)
    best_max_position, best_max_fitness = pso.run()
    # Ahora, la mejor posición máxima es un conjunto de coordenadas x e y
    print("Mejor posición máxima encontrada:", best_max_position)
    print("Valor de función máxima encontrado:", -best_max_fitness)

    if convergence_point < -best_max_fitness :
        convergence_point = -best_max_fitness

    if values > 5:
        break
    else:
        values += 0.1

print("Punto de convergencia: ",convergence_point)
end_time = time.time()  
elapsed_time = end_time - start_time
print("Tiempo de ejecución:", elapsed_time, "segundos")