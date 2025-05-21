import numpy as np
import random
from typing import List, Tuple
import logging

# Configure logging for better debugging and tracking
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

class GeneticAlgorithm:
    def __init__(
        self,
        population_size: int = 100,
        chromosome_length: int = 2,
        bounds: List[Tuple[float, float]] = [(-5.0, 5.0), (-5.0, 5.0)],
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.8,
        max_generations: int = 100,
        tournament_size: int = 5,
        elitism_count: int = 2
    ):
        """
        Initialize the Genetic Algorithm with customizable parameters.
        
        Args:
            population_size: Number of individuals in the population
            chromosome_length: Number of genes (variables) in each individual
            bounds: List of (min, max) tuples for each gene
            mutation_rate: Probability of mutation per gene
            crossover_rate: Probability of crossover between parents
            max_generations: Maximum number of generations to run
            tournament_size: Number of individuals in tournament selection
            elitism_count: Number of best individuals to preserve each generation
        """
        self.population_size = population_size
        self.chromosome_length = chromosome_length
        self.bounds = bounds
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.max_generations = max_generations
        self.tournament_size = tournament_size
        self.elitism_count = max(1, min(elitism_count, population_size // 2))
        self.population = self.initialize_population()
        self.best_fitness_history = []

    def initialize_population(self) -> np.ndarray:
        """Initialize a random population within the specified bounds."""
        population = np.zeros((self.population_size, self.chromosome_length))
        for i in range(self.chromosome_length):
            min_val, max_val = self.bounds[i]
            population[:, i] = np.random.uniform(min_val, max_val, self.population_size)
        return population

    def fitness_function(self, individual: np.ndarray) -> float:
        """
        Evaluate the fitness of an individual.
        Here, we maximize f(x1, x2) = -(x1^2 + x2^2).
        
        Args:
            individual: Array of genes (x1, x2, ...)
        Returns:
            Fitness score (higher is better)
        """
        return -sum(x ** 2 for x in individual)

    def evaluate_population(self) -> List[float]:
        """Evaluate fitness for all individuals in the population."""
        return [self.fitness_function(individual) for individual in self.population]

    def tournament_selection(self, fitness_scores: List[float]) -> np.ndarray:
        """
        Select an individual using tournament selection.
        
        Args:
            fitness_scores: List of fitness scores for the population
        Returns:
            Selected individual
        """
        tournament_indices = random.sample(range(self.population_size), self.tournament_size)
        tournament_fitness = [fitness_scores[i] for i in tournament_indices]
        winner_idx = tournament_indices[np.argmax(tournament_fitness)]
        return self.population[winner_idx]

    def crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform single-point crossover between two parents.
        
        Args:
            parent1, parent2: Parent individuals
        Returns:
            Two offspring individuals
        """
        if random.random() < self.crossover_rate:
            crossover_point = random.randint(1, self.chromosome_length - 1)
            child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
            child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
            return child1, child2
        return parent1.copy(), parent2.copy()

    def mutate(self, individual: np.ndarray) -> np.ndarray:
        """
        Apply Gaussian mutation to an individual.
        
        Args:
            individual: Individual to mutate
        Returns:
            Mutated individual
        """
        for i in range(self.chromosome_length):
            if random.random() < self.mutation_rate:
                min_val, max_val = self.bounds[i]
                # Apply small Gaussian perturbation
                mutation = np.random.normal(0, (max_val - min_val) / 10)
                individual[i] = np.clip(individual[i] + mutation, min_val, max_val)
        return individual

    def run(self) -> Tuple[np.ndarray, float]:
        """
        Run the genetic algorithm.
        
        Returns:
            Best individual and its fitness score
        """
        for generation in range(self.max_generations):
            # Evaluate population
            fitness_scores = self.evaluate_population()
            self.best_fitness_history.append(max(fitness_scores))

            # Log progress
            best_idx = np.argmax(fitness_scores)
            logger.info(
                f"Generation {generation + 1}/{self.max_generations}: "
                f"Best Fitness = {fitness_scores[best_idx]:.6f}, "
                f"Best Individual = {self.population[best_idx]}"
            )

            # Check for convergence (optional early stopping)
            if (
                generation > 10
                and abs(self.best_fitness_history[-1] - self.best_fitness_history[-10]) < 1e-6
            ):
                logger.info("Convergence detected. Stopping early.")
                break

            # Create new population
            new_population = []

            # Elitism: Preserve the best individuals
            elite_indices = np.argsort(fitness_scores)[-self.elitism_count:]
            for idx in elite_indices:
                new_population.append(self.population[idx].copy())

            # Generate the rest of the population
            while len(new_population) < self.population_size:
                parent1 = self.tournament_selection(fitness_scores)
                parent2 = self.tournament_selection(fitness_scores)
                child1, child2 = self.crossover(parent1, parent2)
                new_population.append(self.mutate(child1))
                if len(new_population) < self.population_size:
                    new_population.append(self.mutate(child2))

            self.population = np.array(new_population)

        # Return the best individual and its fitness
        fitness_scores = self.evaluate_population()
        best_idx = np.argmax(fitness_scores)
        return self.population[best_idx], fitness_scores[best_idx]

def main():
    # Example usage
    ga = GeneticAlgorithm(
        population_size=100,
        chromosome_length=2,
        bounds=[(-5.0, 5.0), (-5.0, 5.0)],
        mutation_rate=0.1,
        crossover_rate=0.8,
        max_generations=100,
        tournament_size=5,
        elitism_count=2
    )
    best_solution, best_fitness = ga.run()
    print(f"\nBest Solution: {best_solution}")
    print(f"Best Fitness: {best_fitness}")

if __name__ == "__main__":
    main()