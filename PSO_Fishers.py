import numpy as np
import math
import time

# Definir la función objetivo
def objective_function(x):
    y = x
    z = x
    v = x
    return (-5 - 2 * math.exp(-x-2*y-2*z-2*v) - 2 * math.exp(-2*x-2*y-z-2*v) + 2 * math.exp(-z-2*v-2*x) - 2 * math.exp(-2*x-2*z-2*v-y) - 2 * math.exp(-z-2*v) + 2 * math.exp(-2*y-z-2*v) + 2 * math.exp(-2*v-y-2*x) + 2 * math.exp(-2*y-2*z-v) - 2 * math.exp(-2*x-2*y-2*z-v) + 2 * math.exp(-2*y-x-2*v) + 2 * math.exp(-2*z-v-2*x) + 3 * math.exp(-2*x) + 2 * math.exp(-x) - 2 * math.exp(-v-2*y) + 2 * math.exp(-2*y-2*x-v) + 2 * math.exp(-2*z-2*v-y) - 2 * math.exp(-v-2*x) - 2* math.exp(-y-2*v) - math.exp(-2*z-2*v) - 2 * math.exp(-2*z-v) - math.exp(-2*y-2*z-2*v) + 3 * math.exp(-2*v) - math.exp(-2*v-2*y) + 2 * math.exp(-2*y-x-2*z) + 2 * math.exp(-z) + 2 * math.exp(1) - y - 2 * math.exp(-2*y-x) - 2 * math.exp(-x-2*z) + 2 * math.exp(-2*z-2*x-y) - 2 * math.exp(-y-2*z) - 2 * math.exp(-2*x-y) + 3 * math.exp(-2*z) + 3 * math.exp(-2*y) - 2 * math.exp(-2*y-z) - 2 * math.exp(-2*v-x) - math.exp(-2*y-2*z) + 2 * math.exp(-2*x-2*y-z)+ 2 * math.exp(-x-2*z-2*v) + 2 * math.exp(-v) + 2 * math.exp(-z-2*x) - math.exp(-2*x-2*z) - math.exp(-2*x-2*y) -  math.exp(-2*x-2*y-2*z) + 3 * math.exp(-2*x-2*y-2*z-2*v) - math.exp(-2*z-2*v-2*x) - math.exp(-2*y-2*x-2*v) - math.exp(-2*v-2*x)) / (-1 + math.exp(-2*x) - math.exp(-2*z-2*v) + math.exp(-2*y-2*z-2*v) + math.exp(-2*v) - math.exp(-2*v-2*y) + math.exp(-2*z) + math.exp(-2*y) - math.exp(-2*y-2*z) - math.exp(-2*x-2*z) - math.exp(-2*x-2*y) + math.exp(-2*x-2*y-2*z) - math.exp(-2*x-2*y-2*z-2*v) + math.exp(-2*z-2*v-2*x) + math.exp(-2*y-2*x-2*v) - math.exp(-2*v-2*x))

class Particle:
    def __init__(self, dim, values):
        self.position = np.array([np.random.choice(values) for _ in range(dim)]) # se escoje un valor al azar de la lista.
        self.velocity = np.zeros(dim) # Inicializa la velocidad de la partícula como un vector de ceros de la misma dimensión que la posición.
        self.best_position = self.position.copy() # Inicializa la mejor posición de la partícula como su posición actual
        self.best_fitness = float('inf') # Inicializa la mejor aptitud de la partícula como infinito positivo. 
                                         # Esta es una estrategia común para asegurarse de que cualquier valor de aptitud real sea menor y, por lo tanto, se actualizará en la primera iteración.

    def update_best(self, function):
        fitness = function(self.position) # Calcula la aptitud de la partícula actual utilizando la función de aptitud dada como argumento.
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
values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.0, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]  # Valores posibles para x
pso = PSO(objective_function, 1, 20, values, 1000)
best_min_position, best_min_fitness, elapsed_time = pso.run()
print("Mejor posición mínima encontrada:", best_min_position)
print("Valor de función mínima encontrado:", best_min_fitness) # posición en la cual se obtiene el valor mínimo de la función.
print("Tiempo de ejecución:", elapsed_time, "segundos")