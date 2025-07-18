import os
import pygad
import numpy as np
import pandas as pd
from datetime import datetime
from CODECbreakCode.AudioMixer import FullTrackAudioMixer
import CODECbreakCode.Evaluator as Evaluator
from CODECbreakCode.Evaluator import MeasureHAAQIOutput
import argparse
import matplotlib.pyplot as plt

from typing import Callable, Type
from functools import partial
from Optimiser.config import get_config

cfg = get_config()
genetic_cfg = cfg["genetic_optimiser"]
env_cfg = cfg["env"]

class GeneticOptimiser:
    def __init__(self):
        self._population_size = genetic_cfg["population_size"]
        self._num_generations = genetic_cfg["num_generations"]
        self._mutation_rate = genetic_cfg["mutation_rate"]
        self._parents_mating = genetic_cfg["parents_mating"]
        self._gene_space = []
        self._gene_type = []
        self._fitness_fn = None
        self._collect_data_list = []
        self._best_solutions_fitness = []
        self._best_solutions = []
        self._low_boundary = env_cfg["act_min"]
        self._high_boundary = env_cfg["act_max"]
        self._step_size = genetic_cfg["step"]
        self._gene_num = env_cfg["state_dim"]
        self._starting_population = env_cfg["x0_reinforce"]
        self._ga_instance = None
        
    def ga_init_env(self):
        """
        Given parallel lists low_boundary, high_boundary, and step_size,
        returns:
        - gene_space: List of {"low": low, "high": high, "step": step}
        - gene_type:  List of types (int or float)
        """
        if not (len(self._low_boundary) == len(self._high_boundary) == len(self._step_size)):
            raise ValueError("All input lists must have the same length")

        for low, high, step in zip(self._low_boundary, self._high_boundary, self._step_size):
            # build the dict for this “gene”
            self._gene_space.append({
                "low": low,
                "high": high,
                "step": step
            })

            if all(isinstance(v, int) or (isinstance(v, float) and v.is_integer())
                for v in (low, high, step)):
                self._gene_type.append(int)
            else:
                # determine decimal precision from step_size
                self._gene_type.append([float, 3])

    def on_gen(self, ga_instance):
        print("Generation : ", ga_instance.generations_completed)
        best_solutions = tuple(ga_instance.best_solutions[ga_instance.generations_completed])
        print(f"The last best Solution : ", {best_solutions})
        best_fitness = ga_instance.best_solutions_fitness[ga_instance.generations_completed-1]
        print(f"Fitness of the last best solution :", {best_fitness})


    def set_fitnessfun(self, fitness_fn: Callable[[np.ndarray], float], is_normalised: bool = False):
        def wrapped_reward_fn(ga_instance, solution, solution_idx):
            # 1) compute the score via the user’s raw reward fn
            score = fitness_fn(solution, is_normalised)

            # 2) build [score, *solution] and append to our log
            row = np.concatenate(([score], solution))
            self._collect_data_list.append(row)

            # 3) return the plain score so any caller still just sees a float
            return score
        
        self._fitness_fn = wrapped_reward_fn


    def run(self, num_generations = 0,
            num_genes = 0,
            sol_per_pop = 0,
            preset_population = [],
            parallel_processing = None):
        
        if num_generations == 0:
            num_generations = self._num_generations
        if num_genes == 0:
            num_genes = self._gene_num
        if sol_per_pop == 0:
            sol_per_pop = self._population_size 
        if not preset_population:
            preset_population = [self._starting_population]

        print(f"sol_per_pop: {sol_per_pop}, num_genes: {num_genes}, num_generations: {num_generations}, preset_population: {preset_population}")
        self._ga_instance = pygad.GA(num_generations=num_generations,
                            num_parents_mating=self._parents_mating,
 #                           initial_population=preset_population,
                            num_genes = num_genes,
                            on_generation = self.on_gen,
                            sol_per_pop = sol_per_pop,
                            fitness_func = self._fitness_fn,
                            gene_type = self._gene_type,
                            gene_space = self._gene_space,
                            crossover_type = "uniform",
                            mutation_percent_genes = self._mutation_rate,
                            keep_elitism = 1,
                            save_best_solutions = True,
                            save_solutions = True,
                            parallel_processing = parallel_processing)
        self._ga_instance.run()

    def plot_results(self):
        plt.plot(self._best_solutions_fitness)
        plt.xlabel('Generation')
        plt.ylabel('Best Solution Fitness')
        plt.title('Genetic Algorithm Optimization Progress')
        plt.show()

    def save_results(self, filefold, genre_columns=[], is_outputfulldata = True):
        """
        Save the results of the genetic algorithm to a CSV file.

        """
        if filefold is None:
            raise ValueError("filefold cannot be None")
        else:
            if not os.path.exists(filefold+ 'Data/'):
                os.makedirs(filefold+ 'Data/')

        if not genre_columns:
            genre_columns = [f'gene_{i}' for i in range(self._gene_num)]

        score_df = pd.DataFrame(self._ga_instance.best_solutions_fitness, columns=['score'])
        manip_df = pd.DataFrame(self._ga_instance.best_solutions, columns=genre_columns)
        data_file_path = os.path.join(filefold, 'Data', f'Evo_Data_BestResults_{datetime.now().strftime("%Y%m%d%H%M")}.csv')


        Evo_Data = pd.concat([score_df, manip_df], axis=1)
        Evo_Data.to_csv(data_file_path, index=False)

        if is_outputfulldata:
            # Save the full data collected during the evolution
            self._collect_data_list = np.array(self._collect_data_list)
            if self._collect_data_list.size == 0:
                raise ValueError("No data collected during the evolution process.")
            
            # Create a DataFrame with the collected data
            Evo_Data_Full = pd.DataFrame(self._collect_data_list, columns=['score'] + genre_columns)
            Evo_Data_Full_Path = os.path.join(filefold, 'Data', f'Evo_Data_FullResults_{datetime.now().strftime("%Y%m%d%H%M")}.csv')
            Evo_Data_Full.to_csv(Evo_Data_Full_Path, index=False)

        return Evo_Data, Evo_Data_Full if is_outputfulldata else None

