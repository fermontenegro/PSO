import numpy as np
import math
import time

def objective_function(x):
    y, z = x
    v = y
    return (-5 - 2*math.exp(-x[0]-2*y-2*z-2*v) - 2*math.exp(-2*x[0]-2*y-z-2*v) +
            2*math.exp(-z-2*v-2*x[0]) - 2*math.exp(-2*x[0]-2*z-2*v-y) -
            2*math.exp(-z-2*v) + 2*math.exp(-2*y-z-2*v) +
            2*math.exp(-2*v-y-2*x[0]) + 2*math.exp(-2*y-2*z-v) -
            2*math.exp(-2*x[0]-2*y-2*z-v) + 2*math.exp(-2*y-x[0]-2*v) +
            2*math.exp(-2*z-v-2*x[0]) + 3*math.exp(-2*x[0]) +
            2*math.exp(-x[0]) - 2*math.exp(-v-2*y) +
            2*math.exp(-2*y-2*x[0]-v) + 2*math.exp(-2*z-2*v-y) -
            2*math.exp(-v-2*x[0]) - 2*math.exp(-y-2*v) -
            math.exp(-2*z-2*v) - 2*math.exp(-2*z-v) -
            math.exp(-2*y-2*z-2*v) + 3*math.exp(-2*v) -
            math.exp(-2*v-2*y) + 2*math.exp(-2*y-x[0]-2*z) +
            2*math.exp(-z) + 2*math.exp(-y) -
            2*math.exp(-2*y-x[0]) - 2*math.exp(-x[0]-2*z) +
            2*math.exp(-2*z-2*x[0]-y) - 2*math.exp(-y-2*z) -
            2*math.exp(-2*x[0]-y) + 3*math.exp(-2*z) +
            3*math.exp(-2*y) - 2*math.exp(-2*y-z) -
            2*math.exp(-2*v-x[0]) -math.exp(-2*y-2*z) +
            2*math.exp(-2*x[0]-2*y-z) + 2*math.exp(-x[0]-2*z-2*v) +
            2*math.exp(-v) - 2*math.exp(-z-2*x[0]) -
            math.exp(-2*x[0]-2*z) - math.exp(-2*x[0]-2*y ) -
            math.exp(-2*x[0]-2*y-2*z) + 3*math.exp(-2*x[0]-2*y-2*z-2*v) - 
            math.exp(-2*z-2*v-2*x[0]) - math.exp(-2*y-2*x[0]-2*v) - math.exp(-2*v-2*x[0])) / (-1 + 
            math.exp(-2*x[0]) - math.exp(-2*z-2*v) + math.exp(-2*y-2*z-2*v) +
            math.exp(-2*v) - math.exp(-2*v-2*y) + math.exp(-2*z) + math.exp(-2*y) -
            math.exp(-2*y-2*z) - math.exp(-2*x[0]-2*z) - math.exp(-2*x[0]-2*y) + math.exp(-2*x[0]-2*y-2*z) -
            math.exp(-2*x[0]-2*y-2*z-2*v) + math.exp(-2*z-2*v-2*x[0]) + math.exp(-2*y-2*x[0]-2*v) -
            math.exp(-2*v-2*x[0]))

class Particle:
    def __init__(self, dim, values):
        self.position = np.random.uniform(values, values, dim)  # Inicializa la posición con valores aleatorios en el rango dado
        self.velocity = np.zeros(dim)
        self.best_position = self.position.copy()
        self.best_fitness = float('-inf')

    def update_best(self, function):
        fitness = function(self.position)
        if fitness > self.best_fitness:
            self.best_fitness = fitness
            self.best_position = self.position.copy()

class PSO:
    def __init__(self, function, dim, size, values, iterations):
        self.function = function
        self.dim = dim
        self.swarm = [Particle(dim, values) for _ in range(size)]
        self.global_best_position = np.random.uniform(values, values, dim)  # Inicializa la mejor posición global con valores aleatorios en el rango dado
        self.global_best_fitness = float('inf')
        self.iterations = iterations

    def run(self):
        for i in range(self.iterations):
            for particle in self.swarm:
                particle.update_best(self.function)
                if particle.best_fitness < self.global_best_fitness:
                    self.global_best_fitness = particle.best_fitness
                    self.global_best_position = particle.best_position.copy()
                w = 0.729
                c1 = 1.49445
                c2 = 1.49445
                r1 = np.random.rand(dim)
                r2 = np.random.rand(dim)
                particle.velocity = (w * particle.velocity) + (c1 * r1 * (particle.best_position - particle.position)) + (c2 * r2 * (self.global_best_position - particle.position))
                particle.position += particle.velocity
                particle.position = np.clip(particle.position, values, values)
        return self.global_best_position, self.global_best_fitness

# Parámetros del algoritmo PSO
dim = 2
values = 0.1
size = 20
iterations = 1000

start_time = time.time()  
convergence_point = float('inf')  

while True:
    pso = PSO(lambda x,y: -objective_function(x,y), dim, size, values, iterations)
    best_max_position, best_max_fitness = pso.run()
    print("Mejor posición máxima encontrada:", best_max_position)
    print("Valor de función máxima encontrado:", -best_max_fitness)

    if convergence_point > -best_max_fitness:
        convergence_point = -best_max_fitness

    if values > 10:
        break
    else:
        values += 0.1

print("Punto de convergencia: ", convergence_point)
end_time = time.time()  
elapsed_time = end_time - start_time  
print("Tiempo de ejecución:", elapsed_time, "segundos")
