import numpy as np

# Definir la función objetivo
def objective_function(x):
    return (x**2 - 1)**2

class Particle:
    def __init__(self, dim, minx, maxx):
        self.position = np.random.uniform(low=minx, high=maxx, size=dim)
        self.velocity = np.zeros(dim)
        self.best_position = self.position.copy()
        self.best_fitness = float('inf')

    def update_best(self, function):
        fitness = function(self.position)
        if fitness < self.best_fitness:
            self.best_fitness = fitness
            self.best_position = self.position.copy()

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
            for particle in self.swarm: #es la población de partículas que se utilizan para explorar el espacio de búsqueda y encontrar la mejor solución a la función objetivo.
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
pso = PSO(objective_function, 1, 5, -10, 10, 100)
best_min_position, best_min_fitness = pso.run()
print("Mejor posición mínima encontrada:", best_min_position)
print("Mejor valor de aptitud mínima encontrado:", best_min_fitness)

pso = PSO(lambda x: -objective_function(x), 1, 5, -10, 10, 100)
best_max_position, best_max_fitness = pso.run()
print("Mejor posición máxima encontrada:", best_max_position)
print("Mejor valor de aptitud máxima encontrado:", -best_max_fitness)


# En este caso, la aptitud (también conocida como valor de fitness) 
# se refiere a la evaluación de qué tan buena es una solución o posición 
# de una partícula en el espacio de búsqueda


# COGNITIVE WEIGHT
# En el algoritmo PSO, el peso cognitivo (cognitive weight) es un parámetro que controla la influencia de la mejor posición histórica de una partícula en su movimiento. 
# En otras palabras, es un factor que determina cuánto se debe mover una partícula hacia su mejor posición histórica (local best position).

# La fórmula para actualizar la velocidad de la partícula en el PSO incluye el peso cognitivo como un factor multiplicativo para el vector que va desde la posición actual de la partícula hacia su mejor posición histórica. 
# Un valor mayor de peso cognitivo aumenta la importancia de la mejor posición histórica de la partícula en su movimiento y puede hacer que la partícula converja más rápidamente hacia esa posición. Por otro lado, un valor menor de peso cognitivo da más peso al componente social del algoritmo PSO y puede permitir una exploración más amplia del espacio de búsqueda.

# El valor típico recomendado para el peso cognitivo es 1.49445, como se muestra en el código del ejemplo que compartiste. 
# Sin embargo, este valor se puede ajustar según las necesidades específicas del problema de optimización.


# SOCIAL WIEGHT
# En el algoritmo PSO, el peso social (social weight) es un parámetro que controla la influencia de la mejor posición global encontrada por cualquier partícula en la población en el movimiento de una partícula individual. 
# En otras palabras, es un factor que determina cuánto se debe mover una partícula hacia la mejor posición global encontrada por cualquier partícula en la población.

# La fórmula para actualizar la velocidad de la partícula en el PSO incluye el peso social como un factor multiplicativo para el vector que va desde la posición actual de la partícula hacia la mejor posición global de la población. 
# Un valor mayor de peso social aumenta la importancia de la mejor posición global en el movimiento de la partícula y puede hacer que la población converja más rápidamente hacia una solución óptima. Por otro lado, un valor menor de peso social da más peso al componente cognitivo del algoritmo PSO y puede permitir una exploración más amplia del espacio de búsqueda.

# El valor típico recomendado para el peso social es 1.49445, al igual que para el peso cognitivo. Sin embargo, al igual que con el peso cognitivo, 
# el valor del peso social se puede ajustar según las necesidades específicas del problema de optimización.


# DEF __INIT__
# En el caso de la clase "Particle" y la clase "PSO" en el algoritmo que mostraste, la función __init__ se utiliza para inicializar los atributos de las instancias de la clase. Por ejemplo, en la clase "Particle", se utilizan los argumentos dim, minx, y maxx para crear una partícula con una posición y velocidad aleatorias dentro de los límites dados. 
# Además, se inicializan los valores de la mejor posición histórica y la mejor aptitud histórica en los valores iniciales.