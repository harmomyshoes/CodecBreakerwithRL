import json
import numpy as np
## A typical https://www.soundonsound.com/techniques/compression-limiting?utm_source=chatgpt.com compresor setting

CONFIG = {
    "env": {
        "state_dim": 24,    
        "act_min": [35, 0.1, -25.0,  1.0,  1.0, 200.0, 35, 0.1, -25.0,  1.0,  1.0, 200.0, 35, 0.1, -25.0,  1.0,  1.0, 200.0, 35, 0.1, -25.0,  1.0,  1.0, 200.0],
        "act_max": [60, 3.0, 0.0, 5.0, 10.0, 500.0, 60, 3.0, 0.0, 5.0, 10.0, 500.0, 60, 3.0, 0.0, 5.0, 10.0, 500.0, 60, 3.0, 0.0, 5.0, 10.0, 500.0],
        "x0_reinforce": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.0, 0.0],
    },
    "training": {
        "final_reward": -1e9,
        "disc_factor": 1.0,
        "generation_num": 200, #number of theta updates for REINFORCE-IP
        "entropy_regularization": 0.05, 
        "gradient_clipping": 1.0,
        "sub_episode_length": 5, #number of time_steps in a sub-episode.
        "sub_episode_num_single_batch": 6, #number of sub-episodes in each episode
        "env_num": 1,
        "alpha": 0.2, #regularization coefficient
        "param_alpha": 0.15,
 #       "initial_lr": 0.0001, #initial learning rate for optimiser
        "initial_lr": 0.001, ###FOR Stress test
       "importance_ratio_clipping": 0.15,###The value regulate the PPO policy update range
 #       "importance_ratio_clipping": 0.05,###FOR Stress test
        "lr_half_decay_steps": 50000, #number of steps after which learning rate is decayed to half
        "fc_layer_params_discrete": (30,15), #hidden layer sizes for the policy network
        "fc_layer_params_continuous_crtic_net": (32,32,32), #hidden layer sizes for the critic network
        "fc_layer_params_continuous_actor_net": (128,128,64,64), #hidden layer sizes for the value network        
        "eval_every": 3, #number of episodes after which the policy is evaluated
        "plot_every": 100, #number of episodes after which the training progress is plotted
    },
    "genetic_optimiser": {
        "population_size": 30, ## equal to sub_episode_num_single_batch * sub_episode_length 
        "num_generations": 200,
        "mutation_rate": 0.3,
        "parents_mating": 2,
        "step": [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,]
    },
    "mc_optimiser": {
        "sampling_per_step": 30, ## equal to sub_episode_num_single_batch * sub_episode_length 
        "total_step": 100,
    }
}

def get_config():
    # you could add validation here or overlay
    # environment‐specific overrides, etc.
    return CONFIG

def normalize_action(a_real, a_min = -1, a_max = 1):
    """
    Map a_real in [a_min, a_max] to a_norm in [-1, +1].
    """
    a_min = np.array(a_min, dtype=np.float32)
    a_max = np.array(a_max, dtype=np.float32)
    # First clip a_real to its valid range
    a_real_clipped = np.clip(a_real, a_min, a_max)

    # Then map to [-1,1]
    return ( (a_real_clipped - a_min) / (a_max - a_min) ) * 2.0 - 1.0


'''The function below is used to map the action from [-1, +1] back to the real action space [a_min, a_max].
it also applied the clip and round the result to avoid floating‐point drift.'''
def denormalize_action(a_norm, a_min = CONFIG["env"]["act_min"], a_max = CONFIG["env"]["act_max"]):
    """
    Map a_norm in [-1, +1] back to a_real in [a_min, a_max].
    """
    a_min = np.array(a_min, dtype=np.float32)
    a_max = np.array(a_max, dtype=np.float32)
    # first to [0,1], then to [a_min, a_max]
    a_real = ((a_norm + 1.0) / 2.0) * (a_max - a_min) + a_min
    a_real = np.clip(a_real, a_min, a_max)
    a_real = np.round(a_real, decimals=2)  # Round to 2 decimal places

    # Final safeguard in case of floating‐point drift
    return a_real