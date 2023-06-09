import numpy as np
import math
import time

def objective_function(x,y):
    
    return x**2 + y**2

class Particle:
    def __init__(self, dim, values):
        self.position = np.full(dim, values) #
        self.velocity = np.zeros(dim) # Inicializa la velocidad de la partícula como un vector de ceros de la misma dimensión que la posición.
        self.best_position = self.position.copy() # Inicializa la mejor posición de la partícula como su posición actual
        self.best_fitness = float('-inf') # Inicializa la mejor aptitud de la partícula como infinito positivo. 
                                         # Esta es una estrategia común para asegurarse de que cualquier valor de aptitud real sea menor y, por lo tanto, se actualizará en la primera iteración.

    def update_best(self, function):
        fitness = function(self.position) # Calcula la aptitud de la partícula actual utilizando la función de aptitud dada como argumento.
        if fitness > self.best_fitness:
            self.best_fitness = fitness
            self.best_position = self.position.copy()

class PSO:
    def __init__(self, function, dim, size, values, iterations):
        self.function = function
        self.dim = dim
        self.swarm = [Particle(dim, values) for _ in range(size)]
        self.global_best_position = values
        self.global_best_fitness = float('inf')
        self.iterations = iterations

    def run(self):
        #start_time = time.time()  # Guarda el tiempo de inicio
        for i in range(self.iterations):
            for particle in self.swarm:
                particle.update_best(self.function)
                if particle.best_fitness < self.global_best_fitness:
                    self.global_best_fitness = particle.best_fitness
                    self.global_best_position = particle.best_position.copy()
                w = 0.729  # Inertial weight
                c1 = 1.49445  # Cognitive weight
                c2 = 1.49445  # Social weight
                r1 = 1
                r2 = 1
                particle.velocity = (w * particle.velocity) + (c1 * r1 * (particle.best_position - particle.position)) + (c2 * r2 * (self.global_best_position - particle.position))
                particle.position += particle.velocity
                particle.position = np.clip(particle.position, values, values)
        return self.global_best_position, self.global_best_fitness

start_time = time.time()  # Guarda el tiempo de inicio
values = np.array([0.5, 0.5])
convergence_point = 0

while True:
    pso = PSO(lambda x: -objective_function(x[0], x[1]), 2, 20, values, 1000)
    best_max_position, best_max_fitness = pso.run()
    print("Mejor posición máxima encontrada:", best_max_position)
    print("Valor de función máxima encontrado:", -best_max_fitness)

    if convergence_point < -best_max_fitness :
        convergence_point = -best_max_fitness

    if values[1] >= 1:
        break
    else:
        values[0] += 0.1
        values[1] += 0.1

print(values)
print("Punto de convergencia: ",convergence_point)
end_time = time.time()  # Guarda el tiempo de fin
elapsed_time = end_time - start_time  # Calcula el tiempo transcurrido
print("Tiempo de ejecución:", elapsed_time, "segundos")