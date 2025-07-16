import numpy as np
import pandas as pd
import os,gc
from datetime import datetime
from Optimiser.config import get_config
from typing import Callable

cfg = get_config()

class MonteCarloOptimiser:
    def __init__(self,
                 reward_fn: Callable[[np.ndarray, bool], float],
                 solution_dim: int = cfg["env"]["state_dim"],
                 low: float = cfg["env"]["act_min"],
                 high: float = cfg["env"]["act_max"],
                 sampling_per_step: int = cfg["mc_optimiser"]["sampling_per_step"],
                 total_steps: int = cfg["mc_optimiser"]["total_step"],
                 seed: int = 0):
        """
        reward_fn        : your objective, maps a candidate vector → scalar score
        solution_dim     : dimensionality of each candidate
        [low, high]      : bounds for uniform sampling in each dimension
        total_samples    : total number of Monte-Carlo draws
        """
        self._reward_fn = reward_fn
        self._solution_dim = solution_dim
        self._arg_min = low
        self._arg_max = high
        self._sampling_per_step = sampling_per_step
        self._total_steps = total_steps
        self._best_solution_list = []
        self._best_reward_list = []

        self.rng = np.random.default_rng(seed)
    
    def optimise_by_steps(self, n_steps: int = 0, sampling_per_step: int = 0):
        if n_steps <= 0:
            n_steps = self._total_steps

        if sampling_per_step <= 0:
            sampling_per_step = self._sampling_per_step

        for step in range(n_steps):
            best_R = 0
            best_x = None

            for _ in range(sampling_per_step):
                # one independent draw
                x = np.round(self.rng.uniform(self._arg_min, self._arg_max, size=self._solution_dim),2)
                R = self._reward_fn(x, False)
                if R > best_R:
                    best_R, best_x = R, x
            
            self._best_solution_list.append(best_x)
            self._best_reward_list.append(best_R)
            print(f" Step {step+1}/{n_steps} → best R = {best_R} with the solution {best_x}")
        
    def save_results(self, filefold, para_columns=[], is_outputfulldata = False):
        """
        Save the results of the genetic algorithm to a CSV file.

        """
        if filefold is None:
            raise ValueError("filefold cannot be None")
        else:
            if not os.path.exists(filefold+ 'Data/'): 
                os.makedirs(filefold+ 'Data/')

        dims = cfg["env"]["state_dim"]

        if not para_columns:
            para_columns = [f'dim_{i}' for i in range(dims)]

        score_df = pd.DataFrame(self._best_reward_list, columns=['score'])
        manip_df = pd.DataFrame(self._best_solution_list, columns=para_columns)
        data_file_path = os.path.join(filefold, 'Data', f'MC_Data_BestResults_{datetime.now().strftime("%Y%m%d%H%M")}.csv')


        MC_Data = pd.concat([score_df, manip_df], axis=1)
        MC_Data.to_csv(data_file_path, index=False)

        if is_outputfulldata and self._REINFORCE_logs is not None:
            # Save the full data collected during the evolution
            self._REINFORCE_logs = np.array(self._REINFORCE_logs)
            if self._REINFORCE_logs.size == 0:
                raise ValueError("No data collected during the evolution process.")
            
            # Create a DataFrame with the collected data
            MC_Data_Full = pd.DataFrame(self._REINFORCE_logs, columns=['score'] + para_columns)
            MC_Data_Full_Path = os.path.join(filefold, 'Data', f'MC_Data_FullResults_{datetime.now().strftime("%Y%m%d%H%M")}.csv')
            MC_Data_Full.to_csv(MC_Data_Full_Path, index=False)
            return MC_Data,MC_Data_Full
        else:
            return MC_Data, None