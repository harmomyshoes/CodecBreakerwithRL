#import driver
from tf_agents.drivers.dynamic_episode_driver import DynamicEpisodeDriver
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error as mse

#import environment
from tf_agents.policies import policy_saver
from tf_agents.environments import tf_py_environment #allows parallel computing for generating experiences
from tf_agents.environments.parallel_py_environment import ParallelPyEnvironment

#import replay buffer
from tf_agents import replay_buffers as rb

#import agent
from tf_agents.agents.reinforce import reinforce_agent
from tf_agents.utils import value_ops
from tf_agents.trajectories import StepType,time_step


#other used packages
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d
from matplotlib.axes import Axes as ax
import numpy as np
import pandas as pd
import os,gc
from datetime import datetime
import tensorflow as tf
import tf_agents
from tf_agents.networks import actor_distribution_network
from tf_agents.agents.ppo import ppo_agent
from tf_agents.networks.value_network import ValueNetwork
from tf_agents.utils import common
from tf_agents.trajectories import time_step as ts
from Optimiser.config import get_config, denormalize_action, normalize_action
from Optimiser.env import Env_Continue as Env
from Optimiser.custom_normal_projection_network import NormalProjectionNetwork
from typing import Callable
from functools import partial

#To limit TensorFlow to a specific set of GPUs
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
cpus = tf.config.experimental.list_physical_devices('CPU') 
tf.config.experimental.set_visible_devices(cpus[0], 'CPU')
#enable multiprocessing for parallel computing
tf_agents.system.multiprocessing.enable_interactive_mode()
gc.collect()

cfg = get_config()
train_cfg = cfg["training"]

class continous_RL_train_PPO: 
    def __init__(self,sub_episode_length=0, sub_episode_num_single_batch=0, env_num=0):

        if env_num == 0:
            self._env_num = train_cfg["env_num"]
        else:
            self._env_num = env_num
        if sub_episode_length == 0:
            self._sub_episode_length = train_cfg["sub_episode_length"] #number of time_steps in a sub-episode
        else:
            self._sub_episode_length = sub_episode_length

        if sub_episode_num_single_batch == 0:
            self._sub_episode_num_single_batch = train_cfg["sub_episode_num_single_batch"] #number of sub-episodes in each episodes(trajectories)
        else:
            self._sub_episode_num_single_batch = sub_episode_num_single_batch

        self._episode_length = int(self._sub_episode_length * self._sub_episode_num_single_batch) #number of time_steps in an single env of the trajectory
        self._sub_episode_num = int(self._env_num* self._sub_episode_num_single_batch) #number of all sub-episodes in the parallel environments
        # print(f"In current setting, each epoch includes env number :{self._env_num}, {self._sub_episode_num_single_batch} episodes "
        #       f"in each env(trajectory), {self._sub_episode_length} is the time_steps in each episode, "
        #       f"Therefore, total {self._episode_length} time_steps in each trajectory per env.")
        print("number of sub_episodes used for a single param update:", self._sub_episode_num)

        self._train_env = None
        self._eval_env = None
#        self._REINFORCE_agent = None
        self._train_step_num = 0 #number of training steps
        self._final_reward = train_cfg["final_reward"] #the best reward found so far
        self._final_solution = cfg["env"]["x0_reinforce"].copy() #inital solution
        self._reward_list = [] #the best reward found so far
        self._solution_list = [] #the best solution found so far
        self._REINFORCE_logs = [] #for logging the best objective value of the best solution among all the solutions used for one update of theta
        #Learning Schedule = initial_lr * (C/(step+C))
        self._opt = tf.keras.optimizers.SGD(
            learning_rate=lr_schedule(initial_lr=train_cfg["initial_lr"], C=train_cfg["lr_half_decay_steps"]))
        self._state_dim = cfg["env"]["state_dim"] #dimension of the state space

    def train(self, update_num=0, eval_intv=0):
        self.set_agent()
        """
        Main function for training the agent.
        It sets up the environment, agent, and runs the training loop.
        """
        #self.set_environments()

        #################
        #replay_buffer is used to store policy exploration data
        #################
        replay_buffer = rb.TFUniformReplayBuffer(
                        data_spec = self._REINFORCE_agent.collect_data_spec,  # describe spec for a single iterm in the buffer. A TensorSpec or a list/tuple/nest of TensorSpecs describing a single item that can be stored in this buffer.
                        batch_size = self._env_num,    # number of parallel worlds, where in each world there is an agent generating trajectories
                                                # One batch corresponds to one parallel environment
                        max_length = self._episode_length*5    # The maximum number of items that can be stored in a single batch segment of the buffer.     
                                                            # if exceeding this number previous trajectories will be dropped
        )



        #A driver uses an agent to perform its policy in the environment.
        #The trajectory is saved in replay_buffer
        collect_driver = DynamicEpisodeDriver(
                                                env = self._train_env, #train_env contains parallel environments (no.: env_num)
                                                policy = self._REINFORCE_agent.collect_policy,
                                                observers = [replay_buffer.add_batch],
                                                num_episodes = self._sub_episode_num   #SUM_i (number of episodes to be performed in the ith parallel environment)
                                            )




        ####Examine on the update number and interval
        if update_num == 0:
            update_num = train_cfg["generation_num"]

        if eval_intv == 0:
            eval_intv = train_cfg["eval_every"]
        

        print(f"update_num: {update_num}, eval_intv: {eval_intv}")
        tf.random.set_seed(0)
        for n in range(update_num):
            # 1) collect
            replay_buffer.clear()
            collect_driver.run()

            # 2) grab the Trajectory
            experience = replay_buffer.gather_all()
            rewards = self.extract_episode(traj_batch=experience,epi_length=self._sub_episode_length,attr_name = 'reward') #shape=(sub_episode_num, sub_episode_length)
            observations = self.extract_episode(traj_batch=experience,epi_length=self._sub_episode_length,attr_name = 'observation') #shape=(sub_episode_num, sub_episode_length, state_dim)

            ###The comment out code below is for the purpose if want monitor the interaction with the output of the guassian model.
            # actions = self.extract_episode(traj_batch=experience,epi_length=self._sub_episode_length,attr_name = 'action') #shape=(sub_episode_num, sub_episode_length, state_dim)
            # step_types = self.extract_episode(traj_batch=experience,epi_length=self._sub_episode_length,attr_name = 'step_type')
            # discounts = self.extract_episode(traj_batch=experience,epi_length=self._sub_episode_length,attr_name = 'discount')
            # time_steps = ts.TimeStep(step_types,
            #             tf.zeros_like(rewards),
            #             tf.zeros_like(discounts),
            #             observations
            #             )
            # actions_distribution = self._REINFORCE_agent.collect_policy.distribution(
            #                         time_steps, policy_state=None).action
            #############################################################

            # 3) let the agent do its own loss‐compute + apply_gradients
            loss_info = self._REINFORCE_agent.train(experience)
            

            # 5) grab metrics off loss_info (e.g. loss_info.loss, loss_info.extra)
            print(f"Epoch {n}: policy_loss={loss_info.extra.policy_gradient_loss.numpy():.3f}"
                f"  value_loss={loss_info.extra.value_estimation_loss:.3f}"
                f"  entropy_loss={loss_info.extra.entropy_regularization_loss:.3f}"
                f"  clip_fraction={loss_info.extra.clip_fraction:.3f}")
            
            ###collecting the key information for the training process
            batch_rewards = rewards.numpy()
            batch_obs = observations.numpy()
            
            batch_rewards[:,-1] = -np.power(10,8) #The initial reward is set as 0, we set it as this value to not affect the best_obs_index 
            best_step_reward = np.max(batch_rewards)
            best_step_index = [int(batch_rewards.argmax()/self._sub_episode_length),batch_rewards.argmax()%self._sub_episode_length+1]
            best_step = batch_obs[best_step_index[0],best_step_index[1],:] #best solution
            ###The observation structure is [sub_episode_num, sub_episode_length, state_dim]
            #best_step_reward = f(best_solution)
            avg_step_reward = np.mean(batch_rewards[:,0:-1])
            self._reward_list.append(best_step_reward)
            self._solution_list.append(denormalize_action(best_step))

            for ep_idx in range(self._sub_episode_num-1):
                ep_reward   = batch_rewards[ep_idx,0:-1] #get the reward of the sub-episode, excluding the last time step
                # pick final-step observation as the “solution” for that episode:
                ep_solution = denormalize_action(batch_obs[ep_idx, 1:, :])       
                
                self._REINFORCE_logs.append({
                    'update':   n,           # which generation
                    'episode':  ep_idx,      # which sub-episode
                    'reward':   ep_reward,
                    'solution': ep_solution, # numpy array of shape (state_dim,)
                })


            ###collecting the best solution and reward
            if self._final_reward is None or best_step_reward>self._final_reward: 
                print("final reward before udpate:",self._final_reward)
                self._final_reward = best_step_reward
                self._final_solution = best_step
                print("final reward after udpate:",self._final_reward)
                print('updated final_solution=', denormalize_action(self._final_solution))

            #print(compute_reward(best_obs,alpha))
            if n%eval_intv==0:
                print("train_step no.=",self._train_step_num)
                print('best_solution of this generation=', denormalize_action(best_step))
                print('best step reward=',best_step_reward)
                print('avg step reward=', avg_step_reward)
                #print('episode of rewards', rewards.round(3))
                #### This return on the std and means, actually give the parameters that use to prediction the 
                ###next action dependens on the current state, monitor it not directly reflct the change on the sample
                #print('std:', denormalize_action(actions_distribution.stddev()[0,0].numpy()))
                #print('mean:', denormalize_action(actions_distribution.mean()[0,0].numpy())) 

                ####print out the observations
                flat_obs = tf.reshape(batch_obs, [-1, self._state_dim])
                obs_mean = tf.reduce_mean(flat_obs, axis=0)
                print('obs_mean:', denormalize_action(obs_mean))
                obs_std  = tf.sqrt(tf.reduce_mean((flat_obs - obs_mean)**2, axis=0))
                print('obs_std:', denormalize_action(obs_std))
                print('best_step_index:',best_step_index)
                print('until now, best final_solution=', denormalize_action(self._final_solution), 'best final_reward=', self._final_reward)
                print(' ')
            self._train_step_num += 1

        print('final_solution=',denormalize_action(self._final_solution),'final_reward=',self._final_reward)
            
    
    def set_environments(self, reward_fn: Callable[[np.ndarray, bool], float]):
        make_env = partial(Env,reward_fn = reward_fn, sub_episode_length=self._sub_episode_length)
        print('train_env.batch_size = parallel environment number = ', self._env_num)        
        if self._env_num > 1:
            parallel_env = ParallelPyEnvironment(env_constructors=[make_env]*self._env_num, 
                                     start_serially=False,
                                     blocking=False,
                                     flatten=False
                                    )
#Use the wrapper to create a TFEnvironments obj. (so that parallel computation is enabled)
            self._train_env = tf_py_environment.TFPyEnvironment(parallel_env, check_dims=True) 
            #instance of parallel environments
        else:
            self._train_env = tf_py_environment.TFPyEnvironment(make_env(), check_dims=True)


        

    def set_agent(self):
        train_step = tf.compat.v1.train.get_or_create_global_step()
        actor_net = actor_distribution_network.ActorDistributionNetwork(   
                                                self._train_env.observation_spec(),
                                                self._train_env.action_spec(),
                                                fc_layer_params=train_cfg["fc_layer_params_continuous_actor_net"], #Hidden layers
                                                seed=0, #seed used for Keras kernal initializers for NormalProjectionNetwork.
                                                #discrete_projection_net=_categorical_projection_net
                                                activation_fn = tf.math.tanh,
                                                #continuous_projection_net=(NormalProjectionNetwork),
                                                continuous_projection_net=lambda spec: NormalProjectionNetwork(
                                                    spec,
                                                    state_dependent_std=True     # <— HERE!
                                                    )
                                                    # <-- makes std a function of the state, adding 
                                                # in time of 17/7, in this way to let the prediction of seperate in different dimensions
                                                )
        value_net=ValueNetwork(self._train_env.observation_spec(),
                                    fc_layer_params=train_cfg["fc_layer_params_continuous_crtic_net"], #Hidden layers
                                    activation_fn = tf.math.tanh)

        #train_step_counter = tf.Variable(0)

        self._REINFORCE_agent = ppo_agent.PPOAgent(
                    time_step_spec = self._train_env.time_step_spec(),
                    action_spec = self._train_env.action_spec(),
                    discount_factor = 1,
                    optimizer = self._opt,
                    actor_net = actor_net,
                    value_net = value_net,
                    importance_ratio_clipping= train_cfg["importance_ratio_clipping"],
 #                   entropy_regularization = 0.0,
                    entropy_regularization =0.001,
                    num_epochs=10, ### turning to the default value
                    use_gae= False, ###ultilise the GAE
                    use_td_lambda_return= False,###ultilise the Lamda return
                    train_step_counter = train_step
                    )

        self._REINFORCE_agent.initialize()



    #Functions needed for training
    def extract_episode(self,traj_batch,epi_length,attr_name = 'observation'):
        """
        This function extract episodes (each episode consists of consecutive time_steps) from a batch of trajectories.
        Inputs.
        -----------
        traj_batch:replay_buffer.gather_all(), a batch of trajectories
        epi_length:int, number of time_steps in each extracted episode
        attr_name:str, specify which data from traj_batch to extract
        
        Outputs.
        -----------
        tf.constant(new_attr,dtype=attr.dtype), shape = [new_batch_size, epi_length, state_dim]
                                            or shape = [new_batch_size, epi_length]
        """
        attr = getattr(traj_batch,attr_name)
        original_batch_dim = attr.shape[0]
        traj_length = attr.shape[1]
        epi_num = int(traj_length/epi_length) #number of episodes out of each trajectory
        batch_dim = int(original_batch_dim*epi_num) #new batch_dim
        
        if len(attr.shape)==3:
            stat_dim = attr.shape[2]
            new_attr = np.zeros([batch_dim, epi_length, stat_dim])
        else:
            new_attr = np.zeros([batch_dim, epi_length])
            
        for i in range(original_batch_dim):
            for j in range(epi_num):
                new_attr[i*epi_num+j] = attr[i,j*epi_length:(j+1)*epi_length].numpy()
            
        return tf.constant(new_attr,dtype=attr.dtype)

    def save_policy(self, filefold):
            if filefold is None:
                raise ValueError("filefold cannot be None")
            else:
                if not os.path.exists(filefold+ 'Policy/'): 
                    os.makedirs(filefold+ 'Policy/')

            policy_dir = os.path.join(filefold,'Policy/', 'exported_policy')
            saver = policy_saver.PolicySaver(self._REINFORCE_agent.policy)
            saver.save(policy_dir)

    def evaluate_saved_policy(self, policy_dir, reward_fn: Callable[[np.ndarray, bool], float], num_episodes = 3):
        make_env = partial(Env,reward_fn = reward_fn, sub_episode_length=self._sub_episode_length)
        eval_Env = tf_py_environment.TFPyEnvironment(make_env(), check_dims=True)

        if not os.path.isdir(policy_dir):
            raise FileNotFoundError(f"Policy directory not found: {policy_dir}")
        policy =  tf.compat.v2.saved_model.load(policy_dir)
        policy_state = policy.get_initial_state(batch_size=eval_Env.batch_size)

        # Collect data across episodes
        records = []

        for ep in range(num_episodes):
            time_step = eval_Env.reset()
            step = 0
            while not time_step.is_last():
                # 1) Get the full distribution info (mean & std), not just a sampled action
                dist_info, policy_state = policy.distribution(time_step, policy_state)
                dist = dist_info.action  # this is a tfd.MultivariateNormalDiag

                # 2) Extract mean & std
                mean   = dist.mean().numpy()[0]     # shape [action_dim]
                std    = dist.stddev().numpy()[0]   # same shape

                # 3a) Sample stochastically (training/exploration)
                action = dist.sample().numpy()[0]

                # 3b) —or— take the deterministic “best” action (evaluation)
                # action = mean

                # 4) Step the env
                next_time_step = eval_Env.step(action)

                # 5) Gather reward & state
                reward = next_time_step.reward.numpy()[0]
                state  = time_step.observation.numpy()[0]

                # 6) Record everything
                records.append({
                    'episode': ep,
                    'step':    step,
                    'state':   state,
                    'mean':    mean,
                    'std':     std,
                    'action':  action,
                    'reward':  reward
                })
                print(f"Ep{ep} St{step} | s={state} μ={mean} σ={std} → a={action} r={reward}")

                # 7) Advance
                time_step = next_time_step
                step += 1

        # Build DataFrame and display
        df = pd.DataFrame(records)
        return df


    
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

            score_df = pd.DataFrame(self._reward_list, columns=['score'])
            manip_df = pd.DataFrame(self._solution_list, columns=para_columns)
            data_file_path = os.path.join(filefold, 'Data', f'RL_Data_BestResults_{datetime.now().strftime("%Y%m%d%H%M")}.csv')


            RL_Data = pd.concat([score_df, manip_df], axis=1)
            RL_Data.to_csv(data_file_path, index=False)

            if is_outputfulldata and self._REINFORCE_logs is not None:
                # Save the full data collected during the evolution
                self._REINFORCE_logs = np.array(self._REINFORCE_logs)
                if self._REINFORCE_logs.size == 0:
                    raise ValueError("No data collected during the evolution process.")
                
                # Create a DataFrame with the collected data
                RL_Data_Full_Path = os.path.join(filefold, 'Data', f'RL_Data_FullResults_{datetime.now().strftime("%Y%m%d%H%M")}.csv')

                rows = []
                for rec in self._REINFORCE_logs:
                    for t, r in enumerate(rec['reward']):
                        sol = rec['solution'][t]
                        row = {
                            'update': rec['update'],
                            'episode': rec['episode'],
                            'timestep': t,
                            'reward': r,
                        }
                        for label, val in zip(para_columns, sol):
                            row[label] = val

                        rows.append(row)
                RL_Data_Full = pd.DataFrame(rows)
                RL_Data_Full.to_csv(RL_Data_Full_Path, index=False)
                return RL_Data,RL_Data_Full             
            else:
                return RL_Data, None


class lr_schedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_lr, C):
        self.initial_learning_rate = initial_lr
        self.C = tf.constant(C, dtype=tf.float32)
    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        return self.initial_learning_rate*self.C/(self.C+step)