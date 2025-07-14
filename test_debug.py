import numpy as np
import Optimiser.genetic_evo
from Optimiser.genetic_evo import GeneticOptimiser
import Optimiser.continous_RL_train
from Optimiser.continous_RL_train import continous_RL_train as CRLTrain

def main():
    # GA_Opt = GeneticOptimiser()
    # GA_Opt.ga_init_env()
    # GA_Opt.set_fitnessfun(haaqi_reward_fn)
    # GA_Opt.run(num_generations = 2)
    trainner = CRLTrain(sub_episode_length=5, sub_episode_num_single_batch=3, env_num=1)
    trainner.set_environments(f)
    trainner.train(update_num=5, eval_intv=2)

m1=np.array([-0.5,-0.5,-0.5,-0.5])
m2=np.array([0.5,0.5,0.5,0.5])

def f(x):
    return -np.sum(np.log(np.sum((x-m1)**2)+0.00001)-np.log(np.sum((x-m2)**2)+0.01))

if __name__ == "__main__":
    main()