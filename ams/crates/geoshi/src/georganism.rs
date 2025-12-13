//! Self-organizing geometric entities with evolutionary algorithms
//!
//! # GEORGANISM MODULE
//!▫~•◦------------------------------------------------‣
//!
//! Living geometric structures with evolutionary behaviors, implementing
//! artificial life systems where geometric organisms evolve through natural
//! selection, genetic algorithms, and environmental adaptation in geometric spaces.
//!
//! ### Key Capabilities
//! - **Evolutionary Algorithms:** Genetic crossover, mutation, and fitness evaluation.
//! - **Geometric Self-Organization:** Spatial adaptation and topological evolution.
//! - **Energy Dynamics:** Resource management and metabolic processes in organisms.
//! - **Population Ecology:** Multi-organism interactions and emergent behaviors.
//! - **Environmental Adaptation:** Fitness landscapes and evolutionary pressures.
//!
//! ### Technical Features
//! - **Tournament Selection:** Competitive parent selection for genetic diversity.
//! - **Genome Representation:** Real-valued genetic encoding for continuous traits.
//! - **Interaction Modeling:** Cooperative and competitive organism relationships.
//! - **Spatial Dynamics:** Movement and positioning in geometric neighborhoods.
//! - **Resource Ecosystems:** Dynamic environmental resource distributions.
//!
//! ### Usage Patterns
//! ```rust
//! use geoshi::georganism::{Georganism, EvolutionaryAlgorithm, Environment};
//!
//! let mut ea = EvolutionaryAlgorithm::new(100, 10, Environment::new(2, 32));
//! ea.evolve_generation().unwrap();
//! let stats = ea.get_statistics();
//! ```
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2025 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use crate::GsaResult;
use rand::prelude::*;

/// Geometric organism with evolutionary capabilities
#[derive(Debug, Clone)]
pub struct Georganism {
    pub id: usize,
    pub position: Vec<f64>,
    pub fitness: f64,
    pub genome: Genome,
    pub age: usize,
    pub energy: f64,
}

impl Georganism {
    /// Create a new geometric organism
    pub fn new(id: usize, position: Vec<f64>, genome: Genome) -> Self {
        Self {
            id,
            position,
            fitness: 0.0,
            genome,
            age: 0,
            energy: 1.0,
        }
    }

    /// Evolve the organism by applying genetic operations
    pub fn evolve(&mut self, mutation_rate: f64, rng: &mut impl Rng) {
        self.genome.mutate(mutation_rate, rng);
        self.age += 1;
        self.energy *= 0.95; // Energy decay
    }

    /// Update fitness based on environment
    pub fn update_fitness(&mut self, environment: &Environment) {
        self.fitness = self.genome.evaluate_fitness(&self.position, environment);
    }

    /// Interact with another organism
    pub fn interact(&mut self, other: &mut Georganism, environment: &Environment) {
        let distance = self.distance_to(other);
        let interaction_strength = environment.interaction_strength(distance);

        if interaction_strength > 0.0 {
            // Cooperative interaction
            let energy_transfer = interaction_strength * 0.1;
            self.energy += energy_transfer;
            other.energy -= energy_transfer;
        } else {
            // Competitive interaction
            let fitness_transfer = interaction_strength.abs() * 0.05;
            self.fitness += fitness_transfer;
            other.fitness -= fitness_transfer;
        }
    }

    /// Calculate distance to another organism
    pub fn distance_to(&self, other: &Georganism) -> f64 {
        self.position
            .iter()
            .zip(&other.position)
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt()
    }

    /// Move organism in geometric space
    pub fn move_in_space(&mut self, direction: &[f64], step_size: f64) {
        for (pos, dir) in self.position.iter_mut().zip(direction) {
            *pos += dir * step_size;
        }
        self.energy -= step_size * 0.01; // Movement cost
    }
}

/// Genome representing genetic information of a geometric organism
#[derive(Debug, Clone)]
pub struct Genome {
    pub genes: Vec<f64>,
    pub mutation_strength: f64,
}

impl Genome {
    /// Create a new genome with random genes
    pub fn new(size: usize, rng: &mut impl Rng) -> Self {
        let genes = (0..size).map(|_| rng.random_range(-1.0..1.0)).collect();
        Self {
            genes,
            mutation_strength: 0.1,
        }
    }

    /// Mutate the genome
    pub fn mutate(&mut self, mutation_rate: f64, rng: &mut impl Rng) {
        for gene in &mut self.genes {
            if rng.random::<f64>() < mutation_rate {
                *gene += rng.random_range(-self.mutation_strength..self.mutation_strength);
                *gene = gene.clamp(-1.0, 1.0); // Keep in bounds
            }
        }
    }

    /// Evaluate fitness based on position and environment
    pub fn evaluate_fitness(&self, position: &[f64], environment: &Environment) -> f64 {
        // Fitness based on genome-environment interaction
        let mut fitness = 0.0;

        // Environmental fitness
        fitness += environment.evaluate_position(position);

        // Genome-based fitness (self-expression)
        fitness += self.genes.iter().sum::<f64>() * 0.1;

        // Complexity bonus
        let complexity = self.genes.iter().map(|g| g.abs()).sum::<f64>() / self.genes.len() as f64;
        fitness += complexity * 0.2;

        fitness
    }

    /// Create offspring through crossover
    pub fn crossover(&self, other: &Genome, rng: &mut impl Rng) -> Genome {
        let mut child_genes = Vec::new();

        for i in 0..self.genes.len() {
            let gene = if rng.random::<bool>() {
                self.genes[i]
            } else {
                other.genes[i]
            };
            child_genes.push(gene);
        }

        Genome {
            genes: child_genes,
            mutation_strength: (self.mutation_strength + other.mutation_strength) / 2.0,
        }
    }
}

/// Environment in which geometric organisms evolve
#[derive(Debug, Clone)]
pub struct Environment {
    pub dimensions: usize,
    pub fitness_landscape: Vec<f64>, // Simplified fitness grid
    pub grid_size: usize,
    pub interaction_range: f64,
    pub resources: Vec<Vec<f64>>,
}

impl Environment {
    /// Create a new environment
    pub fn new(dimensions: usize, grid_size: usize) -> Self {
        let total_cells = grid_size.pow(dimensions as u32);
        let fitness_landscape = (0..total_cells)
            .map(|_| rand::random::<f64>() * 2.0 - 1.0)
            .collect();

        let resources = (0..grid_size)
            .map(|_| (0..grid_size).map(|_| rand::random::<f64>()).collect())
            .collect();

        Self {
            dimensions,
            fitness_landscape,
            grid_size,
            interaction_range: 1.0,
            resources,
        }
    }

    /// Evaluate fitness at a given position
    pub fn evaluate_position(&self, position: &[f64]) -> f64 {
        if position.len() != self.dimensions {
            return 0.0;
        }

        // Convert position to grid indices
        let indices: Vec<usize> = position
            .iter()
            .map(|p| ((p + 1.0) * 0.5 * (self.grid_size - 1) as f64) as usize)
            .map(|i| i.min(self.grid_size - 1))
            .collect();

        // Simple linear indexing for 2D case
        if self.dimensions == 2 {
            let index = indices[1] * self.grid_size + indices[0];
            if index < self.fitness_landscape.len() {
                return self.fitness_landscape[index];
            }
        }

        0.0
    }

    /// Calculate interaction strength between organisms
    pub fn interaction_strength(&self, distance: f64) -> f64 {
        if distance > self.interaction_range {
            0.0
        } else {
            // Gaussian interaction
            (-distance * distance / (2.0 * 0.25)).exp()
        }
    }

    /// Update environment (resources, conditions)
    pub fn update(&mut self, rng: &mut impl Rng) {
        // Random resource fluctuations
        for row in &mut self.resources {
            for resource in row {
                *resource += rng.random_range(-0.1..0.1);
                *resource = resource.clamp(0.0, 1.0);
            }
        }

        // Fitness landscape evolution
        for fitness in &mut self.fitness_landscape {
            *fitness += rng.random_range(-0.05..0.05);
            *fitness = fitness.clamp(-1.0, 1.0);
        }
    }
}

/// Evolutionary algorithm for geometric organisms
pub struct EvolutionaryAlgorithm {
    pub population: Vec<Georganism>,
    pub environment: Environment,
    pub generation: usize,
    pub selection_pressure: f64,
    pub elitism_count: usize,
}

impl EvolutionaryAlgorithm {
    /// Create a new evolutionary algorithm
    pub fn new(population_size: usize, genome_size: usize, environment: Environment) -> Self {
        let mut rng = rand::rng();
        let population = (0..population_size)
            .map(|id| {
                let position = (0..environment.dimensions)
                    .map(|_| rng.random_range(-1.0..1.0))
                    .collect();
                let genome = Genome::new(genome_size, &mut rng);
                Georganism::new(id, position, genome)
            })
            .collect();

        Self {
            population,
            environment,
            generation: 0,
            selection_pressure: 2.0,
            elitism_count: population_size / 10,
        }
    }

    /// Run one generation of evolution
    pub fn evolve_generation(&mut self) -> GsaResult<()> {
        let mut rng = rand::rng();

        // Update fitness for all organisms
        for organism in &mut self.population {
            organism.update_fitness(&self.environment);
        }

        // Sort by fitness (descending)
        self.population
            .sort_by(|a, b| b.fitness.partial_cmp(&a.fitness).unwrap());

        // Select parents using tournament selection
        let parents = self.select_parents(&mut rng);

        // Create offspring
        let mut offspring = Vec::new();
        while offspring.len() < self.population.len() - self.elitism_count {
            let parent1 = parents.choose(&mut rng).unwrap();
            let parent2 = parents.choose(&mut rng).unwrap();

            let child_genome = parent1.genome.crossover(&parent2.genome, &mut rng);
            let child_position =
                self.crossover_positions(&parent1.position, &parent2.position, &mut rng);

            let child = Georganism::new(
                self.population.len() + offspring.len(),
                child_position,
                child_genome,
            );

            offspring.push(child);
        }

        // Elitism: keep best organisms
        let elites: Vec<Georganism> = self
            .population
            .iter()
            .take(self.elitism_count)
            .cloned()
            .collect();

        // Combine elites and offspring
        self.population = elites.into_iter().chain(offspring).collect();

        // Apply mutations
        for organism in &mut self.population {
            organism.evolve(0.1, &mut rng);
        }

        // Update environment
        self.environment.update(&mut rng);

        self.generation += 1;

        Ok(())
    }

    /// Select parents using tournament selection
    fn select_parents(&self, rng: &mut impl Rng) -> Vec<&Georganism> {
        let tournament_size = 3;
        let num_parents = self.population.len() / 2;

        let mut parents = Vec::new();

        for _ in 0..num_parents {
            let mut tournament = Vec::new();

            // Select random organisms for tournament
            for _ in 0..tournament_size {
                let idx = rng.random_range(0..self.population.len());
                tournament.push(&self.population[idx]);
            }

            // Find winner
            let winner = tournament
                .into_iter()
                .max_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap())
                .unwrap();

            parents.push(winner);
        }

        parents
    }

    /// Crossover positions
    fn crossover_positions(&self, pos1: &[f64], pos2: &[f64], rng: &mut impl Rng) -> Vec<f64> {
        pos1.iter()
            .zip(pos2)
            .map(|(p1, p2)| if rng.random::<bool>() { *p1 } else { *p2 })
            .collect()
    }

    /// Get statistics about the current population
    pub fn get_statistics(&self) -> PopulationStats {
        let fitnesses: Vec<f64> = self.population.iter().map(|o| o.fitness).collect();
        let energies: Vec<f64> = self.population.iter().map(|o| o.energy).collect();

        PopulationStats {
            generation: self.generation,
            population_size: self.population.len(),
            avg_fitness: fitnesses.iter().sum::<f64>() / fitnesses.len() as f64,
            max_fitness: fitnesses.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b)),
            min_fitness: fitnesses.iter().fold(f64::INFINITY, |a, &b| a.min(b)),
            avg_energy: energies.iter().sum::<f64>() / energies.len() as f64,
        }
    }
}

/// Statistics about a population
#[derive(Debug, Clone)]
pub struct PopulationStats {
    pub generation: usize,
    pub population_size: usize,
    pub avg_fitness: f64,
    pub max_fitness: f64,
    pub min_fitness: f64,
    pub avg_energy: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_georganism_creation() {
        let mut rng = rand::rng();
        let genome = Genome::new(10, &mut rng);
        let position = vec![0.0, 0.0];
        let organism = Georganism::new(0, position, genome);

        assert_eq!(organism.id, 0);
        assert_eq!(organism.position.len(), 2);
        assert_eq!(organism.genome.genes.len(), 10);
    }

    #[test]
    fn test_genome_mutation() {
        let mut rng = rand::rng();
        let mut genome = Genome::new(5, &mut rng);
        let original = genome.genes.clone();

        genome.mutate(1.0, &mut rng); // 100% mutation rate

        // At least one gene should have changed
        let changed = original
            .iter()
            .zip(&genome.genes)
            .any(|(a, b)| (a - b).abs() > 1e-10);

        assert!(changed);
    }

    #[test]
    fn test_genome_crossover() {
        let mut rng = rand::rng();
        let genome1 = Genome::new(5, &mut rng);
        let genome2 = Genome::new(5, &mut rng);

        let child = genome1.crossover(&genome2, &mut rng);

        assert_eq!(child.genes.len(), 5);

        // Child should inherit genes from either parent
        for gene in &child.genes {
            assert!(genome1.genes.contains(gene) || genome2.genes.contains(gene));
        }
    }

    #[test]
    fn test_environment_evaluation() {
        let environment = Environment::new(2, 10);

        let position = vec![0.0, 0.0];
        let fitness = environment.evaluate_position(&position);

        // Should return a value (exact value depends on random initialization)
        assert!((-1.0..=1.0).contains(&fitness));
    }

    #[test]
    fn test_distance_calculation() {
        let mut rng = rand::rng();
        let genome = Genome::new(2, &mut rng);
        let org1 = Georganism::new(0, vec![0.0, 0.0], genome.clone());
        let org2 = Georganism::new(1, vec![3.0, 4.0], genome);

        let distance = org1.distance_to(&org2);
        assert!((distance - 5.0).abs() < 1e-10); // Distance should be 5.0 (3-4-5 triangle)
    }

    #[test]
    fn test_evolutionary_algorithm() {
        let environment = Environment::new(2, 10);
        let mut ea = EvolutionaryAlgorithm::new(20, 5, environment);

        let initial_stats = ea.get_statistics();
        assert_eq!(initial_stats.generation, 0);
        assert_eq!(initial_stats.population_size, 20);

        // Run a few generations
        for _ in 0..3 {
            ea.evolve_generation().unwrap();
        }

        let final_stats = ea.get_statistics();
        assert_eq!(final_stats.generation, 3);
        assert_eq!(final_stats.population_size, 20);

        // Fitness should be a valid number
        assert!(!final_stats.avg_fitness.is_nan());
        assert!(!final_stats.max_fitness.is_nan());
    }

    #[test]
    fn test_organism_interaction() {
        let mut rng = rand::rng();
        let genome = Genome::new(2, &mut rng);
        let mut org1 = Georganism::new(0, vec![0.0, 0.0], genome.clone());
        let mut org2 = Georganism::new(1, vec![0.1, 0.1], genome);

        let environment = Environment::new(2, 10);

        let initial_energy1 = org1.energy;
        let initial_energy2 = org2.energy;

        org1.interact(&mut org2, &environment);

        // Energy conservation (approximately)
        let total_energy = org1.energy + org2.energy;
        let initial_total = initial_energy1 + initial_energy2;
        assert!((total_energy - initial_total).abs() < 0.01);
    }
}
