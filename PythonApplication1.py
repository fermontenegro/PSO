import numpy as np

# Definir la funci�n objetivo
def objective_function(x):
    return x**2


# Define una clase Particle que representa una partícula en el algoritmo PSO. Cada partícula tiene una posición y una velocidad, 
# ambas son vectores de longitud dim. Además, tiene una posición y una aptitud (valor de la función objetivo) mejores conocidas 
# hasta el momento (best_position y best_fitness, respectivamente).
class Particle:
    def __init__(self, dim, minx, maxx):
        self.position = np.random.uniform(low=minx, high=maxx, size=dim)
        self.velocity = np.zeros(dim)
        self.best_position = self.position.copy()
        self.best_fitness = float('inf')

#La función update_best actualiza la posición y la aptitud mejores conocidas si la aptitud de la posición actual de la partícula es mejor.
    def update_best(self, function):
        fitness = function(self.position)
        if fitness < self.best_fitness:
            self.best_fitness = fitness
            self.best_position = self.position.copy()

#PSO que representa el algoritmo PSO. El constructor de esta clase toma como entrada la función objetivo, 
# el tamaño del vector de entrada dim, el tamaño de la población size, los límites inferior y superior del espacio de búsqueda minx y maxx, 
# y el número de iteraciones iterations.

class PSO:
    def __init__(self, function, dim, size, minx, maxx, iterations):
        self.function = function
        self.dim = dim
        self.swarm = [Particle(dim, minx, maxx) for _ in range(size)]
        self.global_best_position = np.random.uniform(low=minx, high=maxx, size=dim)
        self.global_best_fitness = float('inf')
        self.iterations = iterations

    def run(self):
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
                particle.position = np.clip(particle.position, -100, 100)

        return self.global_best_position, self.global_best_fitness

# Ejemplo de uso
pso = PSO(objective_function, 1, 50, -100, 100, 100)
best_position, best_fitness = pso.run()
print("Mejor posici�n encontrada:", best_position)
print("Mejor valor de aptitud encontrado:", best_fitness)
