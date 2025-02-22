import numpy as np
import random
import math
from collections import deque
import heapq

# -----------------------------
# Sample Objective Function
# -----------------------------
def sphere(x):
    """Sphere function (minimization): f(x) = sum(x_i^2)"""
    return np.sum(x**2)

# -----------------------------
# 1. Wheel Optimizer
# -----------------------------
# This optimizer uses a roulette wheel–like selection mechanism.
# Each candidate’s chance to be perturbed is inversely related to its fitness.
class WheelOptimizer:
    def __init__(self, func, bounds, pop_size=30, iterations=100):
        self.func = func
        self.bounds = bounds  # list of (min, max) for each dimension
        self.pop_size = pop_size
        self.iterations = iterations
        self.dim = len(bounds)
        
    def optimize(self):
        # Initialize population uniformly within the bounds.
        population = np.random.uniform(
            low=[b[0] for b in self.bounds],
            high=[b[1] for b in self.bounds],
            size=(self.pop_size, self.dim)
        )
        fitness = np.array([self.func(ind) for ind in population])
        best = population[np.argmin(fitness)]
        best_fitness = np.min(fitness)
        
        for it in range(self.iterations):
            # Compute selection probabilities using inverse fitness (avoid division by zero)
            epsilon = 1e-10
            prob = 1 / (fitness + epsilon)
            prob = prob / np.sum(prob)
            # Roulette wheel selection: choose one candidate
            idx = np.random.choice(self.pop_size, p=prob)
            candidate = population[idx]
            # Perturb the candidate solution slightly
            new_candidate = candidate + np.random.uniform(-0.1, 0.1, self.dim)
            new_candidate = np.clip(new_candidate, [b[0] for b in self.bounds], [b[1] for b in self.bounds])
            new_fitness = self.func(new_candidate)
            # Accept new candidate if improved
            if new_fitness < fitness[idx]:
                population[idx] = new_candidate
                fitness[idx] = new_fitness
                if new_fitness < best_fitness:
                    best_fitness = new_fitness
                    best = new_candidate
        return best, best_fitness

# -----------------------------
# 2. Grey Wolf Optimizer (GWO)
# -----------------------------
# Mimics the leadership hierarchy and hunting behavior of grey wolves.
class GreyWolfOptimizer:
    def __init__(self, func, bounds, pop_size=30, iterations=100):
        self.func = func
        self.bounds = bounds
        self.pop_size = pop_size
        self.iterations = iterations
        self.dim = len(bounds)
        
    def optimize(self):
        # Initialize the population
        population = np.random.uniform(
            low=[b[0] for b in self.bounds],
            high=[b[1] for b in self.bounds],
            size=(self.pop_size, self.dim)
        )
        fitness = np.array([self.func(ind) for ind in population])
        # The three best wolves are alpha, beta, and delta.
        idx = np.argsort(fitness)
        alpha, beta, delta = population[idx[0]], population[idx[1]], population[idx[2]]
        best_fitness = fitness[idx[0]]
        
        for t in range(self.iterations):
            a = 2 - t * (2 / self.iterations)  # Linearly decreases from 2 to 0
            for i in range(self.pop_size):
                for j in range(self.dim):
                    # Update based on the three leaders
                    r1, r2 = np.random.rand(), np.random.rand()
                    A1 = 2 * a * r1 - a
                    C1 = 2 * r2
                    D_alpha = abs(C1 * alpha[j] - population[i, j])
                    X1 = alpha[j] - A1 * D_alpha

                    r1, r2 = np.random.rand(), np.random.rand()
                    A2 = 2 * a * r1 - a
                    C2 = 2 * r2
                    D_beta = abs(C2 * beta[j] - population[i, j])
                    X2 = beta[j] - A2 * D_beta

                    r1, r2 = np.random.rand(), np.random.rand()
                    A3 = 2 * a * r1 - a
                    C3 = 2 * r2
                    D_delta = abs(C3 * delta[j] - population[i, j])
                    X3 = delta[j] - A3 * D_delta

                    # New position is the average influence of the three leaders
                    population[i, j] = (X1 + X2 + X3) / 3.0
                # Ensure the new position is within bounds.
                population[i] = np.clip(population[i],
                                        [b[0] for b in self.bounds],
                                        [b[1] for b in self.bounds])
                fitness[i] = self.func(population[i])
            
            idx = np.argsort(fitness)
            alpha, beta, delta = population[idx[0]], population[idx[1]], population[idx[2]]
            best_fitness = fitness[idx[0]]
        return alpha, best_fitness

# -----------------------------
# 3. Bat Algorithm
# -----------------------------
# Inspired by the echolocation behavior of bats.
class BatAlgorithm:
    def __init__(self, func, bounds, pop_size=30, iterations=100):
        self.func = func
        self.bounds = bounds
        self.pop_size = pop_size
        self.iterations = iterations
        self.dim = len(bounds)
        self.Qmin = 0   # Minimum frequency
        self.Qmax = 2   # Maximum frequency
        
    def optimize(self):
        population = np.random.uniform(
            low=[b[0] for b in self.bounds],
            high=[b[1] for b in self.bounds],
            size=(self.pop_size, self.dim)
        )
        velocity = np.zeros((self.pop_size, self.dim))
        fitness = np.array([self.func(ind) for ind in population])
        best = population[np.argmin(fitness)]
        best_fitness = np.min(fitness)
        
        # Initialize loudness and pulse emission rate
        loudness = np.ones(self.pop_size)
        pulse_rate = np.ones(self.pop_size)
        
        for t in range(self.iterations):
            for i in range(self.pop_size):
                Q = self.Qmin + (self.Qmax - self.Qmin) * np.random.rand()
                velocity[i] += (population[i] - best) * Q
                new_position = population[i] + velocity[i]
                new_position = np.clip(new_position,
                                       [b[0] for b in self.bounds],
                                       [b[1] for b in self.bounds])
                # Local search with pulse rate condition
                if np.random.rand() > pulse_rate[i]:
                    new_position = best + 0.001 * np.random.randn(self.dim)
                new_fitness = self.func(new_position)
                if new_fitness < fitness[i] and np.random.rand() < loudness[i]:
                    population[i] = new_position
                    fitness[i] = new_fitness
                    # Update loudness and pulse rate
                    loudness[i] *= 0.9
                    pulse_rate[i] = 1 - np.exp(-0.9 * t)
                if new_fitness < best_fitness:
                    best = new_position
                    best_fitness = new_fitness
        return best, best_fitness



# -----------------------------
# 6. Genetic Algorithm (GA)
# -----------------------------
# Standard GA with roulette wheel selection, single-point crossover, and mutation.
class GeneticAlgorithm:
    def __init__(self, func, bounds, pop_size=50, iterations=100, 
                 crossover_rate=0.8, mutation_rate=0.1):
        self.func = func
        self.bounds = bounds
        self.pop_size = pop_size
        self.iterations = iterations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.dim = len(bounds)
        
    def optimize(self):
        population = np.random.uniform(
            low=[b[0] for b in self.bounds],
            high=[b[1] for b in self.bounds],
            size=(self.pop_size, self.dim)
        )
        fitness = np.array([self.func(ind) for ind in population])
        
        for _ in range(self.iterations):
            new_population = []
            # Create new generation
            for _ in range(self.pop_size // 2):
                parent1 = self.selection(population, fitness)
                parent2 = self.selection(population, fitness)
                # Crossover
                if np.random.rand() < self.crossover_rate:
                    child1, child2 = self.crossover(parent1, parent2)
                else:
                    child1, child2 = parent1.copy(), parent2.copy()
                # Mutation
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                new_population.extend([child1, child2])
            population = np.array(new_population)
            fitness = np.array([self.func(ind) for ind in population])
        best_index = np.argmin(fitness)
        return population[best_index], fitness[best_index]
    
    def selection(self, population, fitness):
        # Roulette wheel selection (using inverse fitness)
        epsilon = 1e-10
        inv_fitness = 1 / (fitness + epsilon)
        probs = inv_fitness / np.sum(inv_fitness)
        idx = np.random.choice(len(population), p=probs)
        return population[idx]
    
    def crossover(self, parent1, parent2):
        point = np.random.randint(1, self.dim)
        child1 = np.concatenate((parent1[:point], parent2[point:]))
        child2 = np.concatenate((parent2[:point], parent1[point:]))
        return child1, child2
    
    def mutate(self, individual):
        for i in range(self.dim):
            if np.random.rand() < self.mutation_rate:
                individual[i] = np.random.uniform(self.bounds[i][0], self.bounds[i][1])
        return individual

# -----------------------------
# 7. Firefly Algorithm
# -----------------------------
# Fireflies move towards brighter (better) solutions; attractiveness decreases with distance.
class FireflyAlgorithm:
    def __init__(self, func, bounds, pop_size=30, iterations=100,
                 alpha=0.2, beta0=1, gamma=1):
        self.func = func
        self.bounds = bounds
        self.pop_size = pop_size
        self.iterations = iterations
        self.alpha = alpha    # randomization parameter
        self.beta0 = beta0    # attractiveness at zero distance
        self.gamma = gamma    # light absorption coefficient
        self.dim = len(bounds)
        
    def optimize(self):
        population = np.random.uniform(
            low=[b[0] for b in self.bounds],
            high=[b[1] for b in self.bounds],
            size=(self.pop_size, self.dim)
        )
        fitness = np.array([self.func(ind) for ind in population])
        best = population[np.argmin(fitness)]
        best_fitness = np.min(fitness)
        
        for _ in range(self.iterations):
            for i in range(self.pop_size):
                for j in range(self.pop_size):
                    if fitness[j] < fitness[i]:
                        r = np.linalg.norm(population[i] - population[j])
                        beta = self.beta0 * np.exp(-self.gamma * r**2)
                        # Move firefly i toward j
                        population[i] = population[i] + beta * (population[j] - population[i]) \
                                        + self.alpha * (np.random.rand(self.dim) - 0.5)
                        population[i] = np.clip(population[i],
                                                [b[0] for b in self.bounds],
                                                [b[1] for b in self.bounds])
                        fitness[i] = self.func(population[i])
                        if fitness[i] < best_fitness:
                            best = population[i]
                            best_fitness = fitness[i]
        return best, best_fitness

# -----------------------------
# 8. Particle Swarm Optimization (PSO)
# -----------------------------
# PSO simulates a swarm where particles update their velocity and position based on their own experience and the swarm’s best.
class ParticleSwarmOptimizer:
    def __init__(self, func, bounds, pop_size=30, iterations=100,
                 w=0.5, c1=1, c2=1):
        self.func = func
        self.bounds = bounds
        self.pop_size = pop_size
        self.iterations = iterations
        self.w = w    # inertia weight
        self.c1 = c1  # cognitive coefficient
        self.c2 = c2  # social coefficient
        self.dim = len(bounds)
        
    def optimize(self):
        population = np.random.uniform(
            low=[b[0] for b in self.bounds],
            high=[b[1] for b in self.bounds],
            size=(self.pop_size, self.dim)
        )
        velocity = np.zeros((self.pop_size, self.dim))
        fitness = np.array([self.func(ind) for ind in population])
        pbest = population.copy()
        pbest_fitness = fitness.copy()
        gbest = population[np.argmin(fitness)]
        gbest_fitness = np.min(fitness)
        
        for _ in range(self.iterations):
            for i in range(self.pop_size):
                velocity[i] = (self.w * velocity[i] +
                               self.c1 * np.random.rand(self.dim) * (pbest[i] - population[i]) +
                               self.c2 * np.random.rand(self.dim) * (gbest - population[i]))
                population[i] = population[i] + velocity[i]
                population[i] = np.clip(population[i],
                                        [b[0] for b in self.bounds],
                                        [b[1] for b in self.bounds])
                fitness[i] = self.func(population[i])
                if fitness[i] < pbest_fitness[i]:
                    pbest[i] = population[i]
                    pbest_fitness[i] = fitness[i]
                    if fitness[i] < gbest_fitness:
                        gbest = population[i]
                        gbest_fitness = fitness[i]
        return gbest, gbest_fitness



def euclidean_distance(city1, city2):
    """Compute Euclidean distance between two cities (points)."""
    return math.sqrt((city1[0] - city2[0])**2 + (city1[1] - city2[1])**2)

import random

class AntColonyShortestPath:
    def __init__(self, graph, source, destination, n_ants, n_iterations, decay, alpha=1, beta=2, Q=100):
        """
        graph       : Dictionary representation of the graph. Each key is a node and its value is another
                      dictionary where keys are neighboring nodes and values are the weights (costs) of the edge.
        source      : The starting node for each ant.
        destination : The target node each ant must reach.
        n_ants      : Number of ants to simulate in each iteration.
        n_iterations: Total number of iterations to run.
        decay       : Pheromone evaporation rate (between 0 and 1).
        alpha       : Influence of the pheromone (exponent factor).
        beta        : Influence of the heuristic (inverse of distance; exponent factor).
        Q           : Constant used to scale the pheromone deposit (larger Q increases pheromone contribution).
        """
        self.graph = graph
        self.source = source
        self.destination = destination
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.decay = decay
        self.alpha = alpha
        self.beta = beta
        self.Q = Q
        # Initialize pheromone on each edge uniformly
        self.pheromone = {}
        for node in graph:
            for neighbor in graph[node]:
                self.pheromone[(node, neighbor)] = 1.0  # initial pheromone value

    def run(self):
        """Run the ACO algorithm over a number of iterations and return the best path and its cost."""
        best_path = None
        best_cost = float('inf')
        for iteration in range(self.n_iterations):
            all_paths = []
            for ant in range(self.n_ants):
                path = self.find_path(self.source)
                if path is not None:
                    cost = self.path_cost(path)
                    all_paths.append((path, cost))
                    if cost < best_cost:
                        best_cost = cost
                        best_path = path
            # Evaporate pheromones globally
            self.evaporate_pheromones()
            # Update pheromones based on the paths found in this iteration
            self.spread_pheromones(all_paths)
            print(f"Iteration {iteration+1}/{self.n_iterations}, Best cost so far: {best_cost}")
        return best_path, best_cost

    def find_path(self, start):
        """
        Build a path from the source to the destination.
        Each ant starts at the source and picks moves probabilistically until the destination is reached.
        To avoid cycles, we maintain a visited set.
        """
        path = [start]
        visited = set()
        visited.add(start)
        current = start
        while current != self.destination:
            next_node = self.pick_next(current, visited)
            if next_node is None:
                # If no valid move is available, the ant fails to find a complete path
                return None
            path.append(next_node)
            visited.add(next_node)
            current = next_node
        return path

    def pick_next(self, current, visited):
        """
        Choose the next node from the current node based on a probability distribution that takes into account:
          - Pheromone levels (the higher, the more attractive the edge).
          - Heuristic information (the inverse of the edge weight, so lower cost edges are preferred).
        """
        moves = []
        probabilities = []
        for neighbor, distance in self.graph[current].items():
            if neighbor in visited:
                continue  # avoid cycles by not revisiting nodes
            pheromone = self.pheromone[(current, neighbor)] ** self.alpha
            heuristic = (1.0 / distance) ** self.beta  # favor lower weight edges
            moves.append(neighbor)
            probabilities.append(pheromone * heuristic)
        if not moves:
            return None
        total = sum(probabilities)
        probabilities = [p / total for p in probabilities]
        # Randomly choose the next node based on computed probabilities
        return random.choices(moves, weights=probabilities, k=1)[0]

    def path_cost(self, path):
        """Compute the total cost of a given path."""
        cost = 0
        for i in range(len(path) - 1):
            cost += self.graph[path[i]][path[i+1]]
        return cost

    def evaporate_pheromones(self):
        """Reduce all pheromone levels by a factor (evaporation) to avoid unlimited accumulation."""
        for edge in self.pheromone:
            self.pheromone[edge] *= self.decay

    def spread_pheromones(self, all_paths):
        """
        Deposit pheromone on the edges used in the paths found by the ants.
        The amount deposited is scaled inversely to the path cost, so lower cost (better) paths receive more pheromone.
        """
        for path, cost in all_paths:
            if cost == 0:
                continue
            deposit = self.Q / cost
            for i in range(len(path) - 1):
                edge = (path[i], path[i+1])
                self.pheromone[edge] += deposit


def dfs(graph, start, visited=None):
    if visited is None:
        visited = set()
    visited.add(start)
    print(start)  # Process the node (here, simply print it)
    for neighbor in graph[start]:
        if neighbor not in visited:
            dfs(graph, neighbor, visited)
    return visited

def bfs(graph, start):
    visited = set([start])
    queue = deque([start])
    
    while queue:
        vertex = queue.popleft()
        print(vertex)  # Process the node (here, simply print it)
        for neighbor in graph[vertex]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

def dijkstra(graph, start):
    # Initialize distances with infinity, except the start node.
    distances = {vertex: float('infinity') for vertex in graph}
    distances[start] = 0
    priority_queue = [(0, start)]
    
    while priority_queue:
        current_distance, current_vertex = heapq.heappop(priority_queue)
        
        # If this distance is not up to date, skip it.
        if current_distance > distances[current_vertex]:
            continue
        
        for neighbor, weight in graph[current_vertex]:
            distance = current_distance + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(priority_queue, (distance, neighbor))
    
    return distances

def heuristic(a, b):
    # Manhattan distance heuristic
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def astar(grid, start, goal):
    rows, cols = len(grid), len(grid[0])
    open_set = []
    # Heap entry: (f_score, g_score, current_position, parent)
    heapq.heappush(open_set, (heuristic(start, goal), 0, start, None))
    
    came_from = {}  # To reconstruct the path
    g_score = {start: 0}
    closed_set = set()
    
    while open_set:
        f, g, current, parent = heapq.heappop(open_set)
        
        if current in closed_set:
            continue
        
        came_from[current] = parent
        if current == goal:
            # Reconstruct path
            path = []
            while current is not None:
                path.append(current)
                current = came_from[current]
            return path[::-1]
        
        closed_set.add(current)
        x, y = current
        
        # Check neighbors in 4 directions (up, down, left, right)
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            neighbor = (x + dx, y + dy)
            if (0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols 
                    and grid[neighbor[0]][neighbor[1]] == 0):
                tentative_g = g + 1
                if neighbor in g_score and tentative_g >= g_score[neighbor]:
                    continue
                g_score[neighbor] = tentative_g
                f_score = tentative_g + heuristic(neighbor, goal)
                heapq.heappush(open_set, (f_score, tentative_g, neighbor, current))
    
    return None  # No path found

def hill_climbing(f, initial, step_size=0.1, max_iter=1000):
    current = initial
    current_value = f(current)
    
    for i in range(max_iter):
        # Generate neighbors (to the left and right)
        neighbors = [current - step_size, current + step_size]
        next_value = current_value
        next_position = current
        
        for n in neighbors:
            value = f(n)
            if value > next_value:
                next_value = value
                next_position = n
        
        # If no neighbor is better, we've reached a local maximum
        if next_position == current:
            break
        
        current = next_position
        current_value = next_value
    
    return current, current_value

# Komodo Mlipir

def initialize_population(n, m, lb, ub):
    return np.random.uniform(lb, ub, (n, m))

def evaluate_quality(population, objective_function):
    return np.array([objective_function(ind) for ind in population])

def rank_population(population, qualities):
    sorted_indices = np.argsort(qualities)
    return population[sorted_indices], qualities[sorted_indices]

def update_big_males(big_males, weights):
    for i in range(len(big_males)):
        big_males[i] += np.sum(weights, axis=0)
    return big_males

def update_female(female, big_males, lb, ub):
    if len(big_males) == 0: #if big males array is empty return female without changes
        return female
    r = np.random.rand()
    winner = big_males[0]
    if np.random.rand() < 0.5:
        female = r * female + (1 - r) * winner
    else:
        alpha = np.random.rand()
        female += (2 * r - 1) * alpha * (ub - lb)
    return female

def update_small_males(small_males, weights):
    for i in range(len(small_males)):
        small_males[i] += np.sum(weights, axis=0)
    return small_males

def update_population_size(n, delta_f1, delta_f2, a):
    if delta_f1 > 0 and delta_f2 > 0:
        return max(1, n - a)
    elif delta_f1 == 0 and delta_f2 == 0:
        return n + a
    return n

def komodo_mlipir_algorithm(n, m, p, d, lb, ub, objective_function, max_iters=100):
    population = initialize_population(n, m, lb, ub)
    best_solution = None
    best_quality = float('inf')
    
    for iteration in range(max_iters):
        qualities = evaluate_quality(population, objective_function)
        population, qualities = rank_population(population, qualities)
        
        q = int(p * n) #Fixed calculation of q
        s = n - q - 1
        big_males = population[:q]
        female = population[q]
        small_males = population[q+1:]
        
        print(f"Iteration: {iteration}, q: {q}, big_males shape: {big_males.shape}") #Added print statement
        
        weights = np.random.rand(q, m) * 0.1
        big_males = update_big_males(big_males, weights)
        female = update_female(female, big_males, lb, ub)
        small_males = update_small_males(small_males, weights)
        
        population = np.vstack((big_males, [female], small_males))
        
        best_idx = np.argmin(qualities)
        if qualities[best_idx] < best_quality:
            best_quality = qualities[best_idx]
            best_solution = population[best_idx]
        
        delta_f1, delta_f2 = np.random.randint(0, 2, size=2)
        n = update_population_size(n, delta_f1, delta_f2, a=1)
        
    return best_solution, best_quality

# Example usage with a sample objective function
def sphere_function(x):
    return np.sum(x**2)

best_solution, best_quality = komodo_mlipir_algorithm(n=20, m=5, p=0.3, d=0.1, lb=-10, ub=10, objective_function=sphere_function)
print("Best Solution:", best_solution)
print("Best Quality:", best_quality)