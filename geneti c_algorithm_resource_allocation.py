```python
import numpy as np
import random
from typing import List, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

class GeneticAlgorithm:
    def __init__(
        self,
        population_size: int = 50,
        num_tasks: int = 15,
        num_cells: int = 8,
        task_durations: List[float] = None,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.8,
        max_generations: int = 100,
        tournament_size: int = 5,
        elitism_count: int = 2,
        balance_weight: float = 0.5
    ):
        """
        Initialize the Genetic Algorithm for resource allocation.
        
        Args:
            population_size: Number of individuals
            num_tasks: Number of tasks (chromosome length)
            num_cells: Number of manufacturing cells
            task_durations: Processing times for tasks
            mutation_rate: Probability of mutation per gene
            crossover_rate: Probability of crossover
            max_generations: Maximum generations
            tournament_size: Size for tournament selection
            elitism_count: Number of elite individuals
            balance_weight: Weight for workload balance penalty
        """
        self.population_size = population_size
        self.num_tasks = num_tasks
        self.num_cells = num_cells
        self.task_durations = task_durations if task_durations else np.random.uniform(1, 10, num_tasks)
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.max_generations = max_generations
        self.tournament_size = tournament_size
        self.elitism_count = max(1, min(elitism_count, population_size // 2))
        self.balance_weight = balance_weight
        self.population = self.initialize_population()
        self.best_fitness_history = []

    def initialize_population(self) -> np.ndarray:
        """Initialize population with random task-to-cell assignments."""
        return np.random.randint(0, self.num_cells, (self.population_size, self.num_tasks))

    def is_valid_chromosome(self, individual: np.ndarray) -> bool:
        """
        Validate a chromosome.
        
        Args:
            individual: Chromosome to validate
        Returns:
            True if valid, False otherwise
        """
        return (
            len(individual) == self.num_tasks and
            np.all((individual >= 0) & (individual < self.num_cells))
        )

    def fitness_function(self, individual: np.ndarray) -> float:
        """
        Evaluate fitness: minimize makespan and balance workload.
        
        Args:
            individual: Task-to-cell assignments
        Returns:
            Fitness score (higher is better)
        """
        if not self.is_valid_chromosome(individual):
            return -np.inf
        
        # Calculate cell loads
        cell_loads = np.zeros(self.num_cells)
        for task_idx, cell_idx in enumerate(individual):
            cell_loads[cell_idx] += self.task_durations[task_idx]
        
        # Makespan: maximum load across cells
        makespan = np.max(cell_loads)
        
        # Workload balance: standard deviation of loads
        balance_penalty = np.std(cell_loads)
        
        # Fitness: negative of (makespan + weighted balance penalty)
        return -(makespan + self.balance_weight * balance_penalty)

    def evaluate_population(self) -> List[float]:
        """Evaluate fitness for all individuals."""
        return [self.fitness_function(individual) for individual in self.population]

    def tournament_selection(self, fitness_scores: List[float]) -> np.ndarray:
        """
        Select individual via tournament selection.
        
        Args:
            fitness_scores: List of fitness scores
        Returns:
            Selected individual
        """
        tournament_indices = random.sample(range(self.population_size), self.tournament_size)
        tournament_fitness = [fitness_scores[i] for i in tournament_indices]
        winner_idx = tournament_indices[np.argmax(tournament_fitness)]
        return self.population[winner_idx]

    def one_point_crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform one-point crossover.
        
        Args:
            parent1, parent2: Parent chromosomes
        Returns:
            Two offspring
        """
        if random.random() < self.crossover_rate:
            point = random.randint(1, self.num_tasks - 1)
            child1 = np.concatenate((parent1[:point], parent2[point:]))
            child2 = np.concatenate((parent2[:point], parent1[point:]))
            return child1, child2
        return parent1.copy(), parent2.copy()

    def uniform_crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform uniform crossover.
        
        Args:
            parent1, parent2: Parent chromosomes
        Returns:
            Two offspring
        """
        child1, child2 = parent1.copy(), parent2.copy()
        if random.random() < self.crossover_rate:
            for i in range(self.num_tasks):
                if random.random() < 0.5:
                    child1[i], child2[i] = child2[i], child1[i]
        return child1, child2

    def single_swap_mutation(self, individual: np.ndarray) -> np.ndarray:
        """
        Mutate by swapping one task's cell assignment.
        
        Args:
            individual: Chromosome to mutate
        Returns:
            Mutated chromosome
        """
        if random.random() < self.mutation_rate:
            idx = random.randint(0, self.num_tasks - 1)
            new_cell = random.randint(0, self.num_cells - 1)
            individual[idx] = new_cell
        return individual

    def task_reassignment_mutation(self, individual: np.ndarray) -> np.ndarray:
        """
        Mutate by reassigning 2-3 tasks to new cells.
        
        Args:
            individual: Chromosome to mutate
        Returns:
            Mutated chromosome
        """
        num_swaps = random.randint(2, 3)
        if random.random() < self.mutation_rate:
            indices = random.sample(range(self.num_tasks), num_swaps)
            for idx in indices:
                individual[idx] = random.randint(0, self.num_cells - 1)
        return individual

    def run(self, crossover_method: str = 'one_point') -> Tuple[np.ndarray, float]:
        """
        Run the genetic algorithm.
        
        Args:
            crossover_method: 'one_point' or 'uniform'
        Returns:
            Best individual and its fitness
        """
        crossover = self.one_point_crossover if crossover_method == 'one_point' else self.uniform_crossover
        
        for generation in range(self.max_generations):
            fitness_scores = self.evaluate_population()
            self.best_fitness_history.append(max(fitness_scores))

            best_idx = np.argmax(fitness_scores)
            best_individual = self.population[best_idx]
            makespan = np.max([sum(self.task_durations[i] for i, cell in enumerate(best_individual) if cell == c) 
                              for c in range(self.num_cells)])
            logger.info(
                f"Generation {generation + 1}/{self.max_generations}: "
                f"Best Fitness = {fitness_scores[best_idx]:.2f}, "
                f"Makespan = {makespan:.2f}, "
                f"Best Individual = {best_individual}"
            )

            # Convergence check
            if generation > 10 and abs(self.best_fitness_history[-1] - self.best_fitness_history[-10]) < 1e-3:
                logger.info("Convergence detected. Stopping early.")
                break

            # Create new population
            new_population = []
            elite_indices = np.argsort(fitness_scores)[-self.elitism_count:]
            for idx in elite_indices:
                new_population.append(self.population[idx].copy())

            while len(new_population) < self.population_size:
                parent1 = self.tournament_selection(fitness_scores)
                parent2 = self.tournament_selection(fitness_scores)
                child1, child2 = crossover(parent1, parent2)
                new_population.append(self.single_swap_mutation(child1))
                if len(new_population) < self.population_size:
                    new_population.append(self.task_reassignment_mutation(child2))

            self.population = np.array(new_population)

        fitness_scores = self.evaluate_population()
        best_idx = np.argmax(fitness_scores)
        return self.population[best_idx], fitness_scores[best_idx]

def main():
    # Random task durations for demonstration
    task_durations = np.random.uniform(1, 10, 15)
    
    # Run GA
    ga = GeneticAlgorithm(
        population_size=50,
        num_tasks=15,
        num_cells=8,
        task_durations=task_durations,
        mutation_rate=0.1,
        crossover_rate=0.8,
        max_generations=100,
        tournament_size=5,
        elitism_count=2,
        balance_weight=0.5
    )
    best_solution, best_fitness = ga.run(crossover_method='one_point')
    
    # Calculate makespan for best solution
    cell_loads = np.zeros(ga.num_cells)
    for task_idx, cell_idx in enumerate(best_solution):
        cell_loads[cell_idx] += task_durations[task_idx]
    makespan = np.max(cell_loads)
    
    print(f"\nFinal Best Solution: {best_solution}")
    print(f"Best Fitness: {best_fitness:.2f}")
    print(f"Makespan: {makespan:.2f}")
    print(f"Cell Loads: {cell_loads}")

if __name__ == "__main__":
    main()
```