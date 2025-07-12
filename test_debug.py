import numpy as np
import Optimiser.genetic_evo
from Optimiser.genetic_evo import GeneticOptimiser

def main():
    GA_Opt = GeneticOptimiser()
    GA_Opt.ga_init_env()
    GA_Opt.set_fitnessfun(haaqi_reward_fn)
    GA_Opt.run(num_generations = 2)

def haaqi_reward_fn(solution: np.ndarray) -> float:
    """
    Evaluate the fitness of a candidate solution vector.

    This example uses the Sphere function:
      f(x) = sum(x_i^2) for i = 1..n

    Args:
        solution: 1D numpy array of shape (n,), the candidate solution.

    Returns:
        A float representing the objective value. Lower is better.
    """
    # ensure it's a 1-D float array
    x = solution.astype(np.float64).ravel()
    return float(np.sum(x ** 2))

if __name__ == "__main__":
    main()