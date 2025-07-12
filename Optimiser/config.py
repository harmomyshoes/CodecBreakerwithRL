import json
## A typical https://www.soundonsound.com/techniques/compression-limiting?utm_source=chatgpt.com compresor setting

CONFIG = {
    "env": {
        "state_dim": 4,
        "act_min": [-15.0, -4.0, -10.0, -300.0],
        "act_max": [15.0, 4.0, 10.0, 300.0],
        "x0_reinforce": [1.0, 0.0, 1.0, -50.0],
        "step_size": [0.1, 0.1, 0.1, 0.01],
    },
    "training": {
        "final_reward": -1e9,
        "disc_factor": 1.0,
        "generation_num": 200, #number of theta updates for REINFORCE-IP
        "entropy_regularization": 0.05, 
        "gradient_clipping": 1.0,
        "sub_episode_length": 10, #number of time_steps in a sub-episode.
        "sub_episode_num_single_batch": 3, #number of sub-episodes in each episode
        "env_num": 1,
        "alpha": 0.2, #regularization coefficient
        "param_alpha": 0.15,
        "initial_lr": 0.0001, #initial learning rate for optimiser
        "lr_half_decay_steps": 50000, #number of steps after which learning rate is decayed to half
        "fc_layer_params_discrete": (30,15), #hidden layer sizes for the policy network
        "fc_layer_params_continuous": (30,30,30), #hidden layer sizes for the value network
        "eval_every": 20, #number of episodes after which the policy is evaluated
        "plot_every": 100, #number of episodes after which the training progress is plotted
    },
    "genetic_optimiser": {
        "population_size": 30, ## equal to sub_episode_num_single_batch * sub_episode_length 
        "num_generations": 200,
        "mutation_rate": 0.2,
        "parents_mating": 2,
        "step": [0.1, 0.1, 0.1, 1]
    }
}

def get_config():
    # you could add validation here or overlay
    # environment‐specific overrides, etc.
    return CONFIG