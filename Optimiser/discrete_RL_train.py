#import driver
from tf_agents.drivers import py_driver
from tf_agents.drivers.dynamic_episode_driver import DynamicEpisodeDriver
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error as mse
#import environment
from tf_agents.environments import py_environment
from tf_agents.environments import tf_py_environment #allows parallel computing for generating experiences
from tf_agents.environments.parallel_py_environment import ParallelPyEnvironment

#import replay buffer
from tf_agents import replay_buffers as rb

#import agent
from tf_agents.agents.reinforce import reinforce_agent
from tf_agents.utils import value_ops
from tf_agents.trajectories import StepType

#other used packages
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d
from matplotlib.axes import Axes as ax
import numpy as np
import tensorflow as tf
import tf_agents
from tf_agents.networks import actor_distribution_network
from tf_agents.specs import array_spec
from tf_agents.utils import common
from tf_agents.trajectories import time_step as ts
import os,gc
from config import get_config
from env import Env_Discrete as Env
from multicategorical_projection_network import MultiCategoricalProjectionNetwork


#To limit TensorFlow to CPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
cpus = tf.config.experimental.list_physical_devices('CPU') 
tf.config.experimental.set_visible_devices(cpus[0], 'CPU')
#enable multiprocessing for parallel computing
tf_agents.system.multiprocessing.enable_interactive_mode()
gc.collect()


cfg = get_config()
train_cfg = cfg["training"]

class discrete_RL_train:

    def __init__(self):
        self._env_num = train_cfg["env_num"]
        self._sub_episode_length = train_cfg["sub_episode_length"] #number of time_steps in a sub-episode
        self._episode_length = self._sub_episode_length * train_cfg["sub_episode_num_single_batch"] #number of time_steps in an episode
        self._sub_episode_num = int(self._env_num* train_cfg["sub_episode_num_single_batch"])
        self._train_env = None
        self._eval_env = None
        self._REINFORCE_agent = None
        self._train_step_num = 0 #number of training steps
        self._final_reward = train_cfg["final_reward"] #the best reward found so far
        self._final_solution = cfg["env"]["x0_reinforce"].copy() #the best solution found so far
        #Learning Schedule = initial_lr * (C/(step+C))
        self._opt = tf.keras.optimizers.legacy.SGD(
            learning_rate=lr_schedule(initial_lr=train_cfg["initial_lr"], C=train_cfg["lr_half_decay_steps"]))


    def train(self):
        #Set the environments
        self.set_environments()
        #Set the agent
        self.set_agent()

        #################
        #replay_buffer is used to store policy exploration data
        #################
        replay_buffer = rb.TFUniformReplayBuffer(
                        data_spec = self._REINFORCE_agent.collect_data_spec,  # describe spec for a single iterm in the buffer. A TensorSpec or a list/tuple/nest of TensorSpecs describing a single item that can be stored in this buffer.
                        batch_size = self._env_num,    # number of parallel worlds, where in each world there is an agent generating trajectories
                                                # One batch corresponds to one parallel environment
                        max_length = self._episode_length*2    # The maximum number of items that can be stored in a single batch segment of the buffer.     
                                                            # if exceeding this number previous trajectories will be dropped
        )

        #test_buffer is used for evaluating a policy
        test_buffer = rb.TFUniformReplayBuffer(
                        data_spec= self._REINFORCE_agent.collect_data_spec,  # describe a single iterm in the buffer. A TensorSpec or a list/tuple/nest of TensorSpecs describing a single item that can be stored in this buffer.
                        batch_size= 1,    # number of parallel worlds, where in each world there is an agent generating trajectories
                                                            # train_env.batch_size: The batch size expected for the actions and observations.  
                                                            # batch_size: Batch dimension of tensors when adding to buffer. 
                        max_length = self._episode_length         # The maximum number of items that can be stored in a single batch segment of the buffer.     
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

        #For policy evaluation
        test_driver = py_driver.PyDriver(
                                            env = self._eval_env, #PyEnvironment or TFEnvironment class
                                            policy = self._REINFORCE_agent.policy,
                                            observers = [test_buffer.add_batch],
                                            max_episodes=1, #optional. If provided, the data generation ends whenever
                                                            #either max_steps or max_episodes is reached.
                                            max_steps=self._sub_episode_length
                                        )

        ##### - Train REINFORCE_agent's actor_network multiple times.
        REINFORCE_logs = [] #for logging the best objective value of the best solution among all the solutions used for one update of theta
        
        update_num = train_cfg["generation_num"] #number of updates for the policy 
        eval_intv = train_cfg["eval_every"] #number of updates required before each policy evaluation
        plot_intv = train_cfg["plot_every"] #number of updates required before each training progress plot

        tf.random.set_seed(0)
        for n in range(0,update_num):
            #Generate Trajectories
            replay_buffer.clear()
            collect_driver.run()  #a batch of trajectories will be saved in replay_buffer
            
            experience = replay_buffer.gather_all() #get the batch of trajectories, shape=(batch_size, episode_length)
            rewards = self.extract_episode(traj_batch=experience,epi_length=self._sub_episode_length,attr_name = 'reward') #shape=(sub_episode_num, sub_episode_length)
            observations = self.extract_episode(traj_batch=experience,epi_length=self._sub_episode_length,attr_name = 'observation') #shape=(sub_episode_num, sub_episode_length, state_dim)
            actions = self.extract_episode(traj_batch=experience,epi_length=self._sub_episode_length,attr_name = 'action') #shape=(sub_episode_num, sub_episode_length, state_dim)
            step_types = self.extract_episode(traj_batch=experience,epi_length=self._sub_episode_length,attr_name = 'step_type')
            discounts = self.extract_episode(traj_batch=experience,epi_length=self._sub_episode_length,attr_name = 'discount')

            time_steps = ts.TimeStep(step_types,
                                    tf.zeros_like(rewards),
                                    tf.zeros_like(discounts),
                                    observations
                                    )
            
            rewards_sum = tf.reduce_sum(rewards, axis=1) #shape=(sub_episode_num,)
            
            with tf.GradientTape() as tape:
                #trainable parameters in the actor_network in REINFORCE_agent
                variables_to_train = self._REINFORCE_agent._actor_network.trainable_weights
            
                ###########Compute J_loss = -J
                actions_distribution = self._REINFORCE_agent.collect_policy.distribution(
                                    time_steps, policy_state=None).action
                
                ######Entropy trace during “burn-in”, Watch the policy’s entropy during the very first few updates
                ent = actions_distribution.entropy().numpy()   # shape = (batch_size, state_dim)
                print(f"[step {n:3d}] mean entropy per-dim = {ent.mean():.4f}")
                #######
            
                #log(pi(action|state)), shape = (batch_size, epsode_length)
                action_log_prob = common.log_probability(actions_distribution, 
                                                        actions,
                                                        self._REINFORCE_agent.action_spec)

                J = tf.reduce_sum(tf.reduce_sum(action_log_prob,axis=1)*rewards_sum)/self._sub_episode_num

                ###########Compute regularization loss from actor_net params
                regu_term = tf.reduce_sum(variables_to_train[0]**2)
                num = len(variables_to_train) #number of vectors in variables_to_train
                for i in range(1,num):
                    regu_term += tf.reduce_sum(variables_to_train[i]**2)
                
                total = -J + train_cfg["param_alpha"] *regu_term
            
            #update parameters in the actor_network in the policy
            grads = tape.gradient(total, variables_to_train)
            grads_and_vars = list(zip(grads, variables_to_train))
            self._opt.apply_gradients(grads_and_vars=grads_and_vars)
            self._train_step_num += 1
            
            batch_rewards = rewards.numpy()
            batch_rewards[:,-1] = -np.power(10,8) #The initial reward is set as 0, we set it as this value to not affect the best_obs_index 
            best_step_reward = np.max(batch_rewards)
            best_step_index = [int(batch_rewards.argmax()/self._sub_episode_length),batch_rewards.argmax()%self._sub_episode_length+1]
            best_step = observations[best_step_index[0],best_step_index[1],:] #best solution
            #best_step_reward = f(best_solution)
            avg_step_reward = np.mean(batch_rewards[:,0:-1])
            REINFORCE_logs.append(best_step_reward)

            if self._final_reward is None or best_step_reward>self._final_reward:
                print("final reward before udpate:",self._final_reward)
                self._final_reward = best_step_reward
                self._final_solution = best_step.numpy()
                print("final reward after udpate:",self._final_reward)
                print('updated final_solution=', self._final_solution)

            #print(compute_reward(best_obs,alpha))
            if n%eval_intv==0:
                print("train_step no.=",self._train_step_num)
                print('best_solution of this generation=', best_step.numpy())
                #print('best step reward=',best_step_reward.round(3),f(best_step.numpy()))
                print('best step reward=',best_step_reward.round(3))
                print('avg step reward=', round(avg_step_reward,3))
                #print('act_logits:', actions_distribution._logits.numpy()[0] ) #second action mean
                print('best_step_index:',best_step_index)
                print('collect_traj:', observations[0,:].numpy())
                
                #Test trajectory generated by mode of action_distribution in agent policy
                test_buffer.clear()
                test_driver.run(self._eval_env.reset())  #generate batches of trajectories with agent.collect_policy, and save them in replay_buffer
                experience = test_buffer.gather_all()  #get all the stored items in replay_buffer
                test_trajectory = experience.observation[0,:].numpy()
                print('test_traj', test_trajectory)
                
                    
        print('final_solution=',self._final_solution,
            'final_reward=',self._final_reward,
            )
        REINFORCE_logs = [max(REINFORCE_logs[0:i]) for i in range(1,update_num+1)] #rolling max
        return REINFORCE_logs, self._final_solution, self._final_reward

    def plot_reinforce_logs(logs):
        """
        Plots the best reward per generation from REINFORCE_logs.
        
        Args:
        logs (list or array-like): Best reward recorded at each generation.
        """
        plt.figure()
        plt.plot(logs)
        plt.xlabel('Generation')
        plt.ylabel('Best Reward')
        plt.title('Best Reward per Generation')
        plt.show()
    
    def set_environments(self):
        if self._env_num > 1:
            parallel_env = ParallelPyEnvironment(env_constructors=[Env]*self._env_num, 
                                     start_serially=False,
                                     blocking=False,
                                     flatten=False
                                    )
#Use the wrapper to create a TFEnvironments obj. (so that parallel computation is enabled)
            self._train_env = tf_py_environment.TFPyEnvironment(parallel_env, check_dims=True) #instance of parallel environments
            self._eval_env = tf_py_environment.TFPyEnvironment(Env(), check_dims=False) #instance
        # train_env.batch_size: The batch size expected for the actions and observations.  
            print('train_env.batch_size = parallel environment number = ', self._env_num)
        else:
            self._train_env = tf_py_environment.TFPyEnvironment(Env(), check_dims=True)
            self._eval_env = tf_py_environment.TFPyEnvironment(Env(), check_dims=False)

    def set_agent(self):
        actor_net = actor_distribution_network.ActorDistributionNetwork(   
                    self._train_env.observation_spec(),
                    self._train_env.action_spec(),
                    fc_layer_params=train_cfg["fc_layer_params_discrete"], #Hidden layers
                    seed=0, #seed used for Keras kernal initializers for NormalProjectionNetwork.
                    discrete_projection_net=MultiCategoricalProjectionNetwork,
                    activation_fn = tf.math.tanh,
                    #continuous_projection_net=(NormalProjectionNetwork)
                    )
        
        train_step_counter = tf.Variable(0)

        self._REINFORCE_agent = reinforce_agent.ReinforceAgent(
                    time_step_spec = self._train_env.time_step_spec(),
                    action_spec = self._train_env.action_spec(),
                    actor_network = actor_net,
                    value_network = None,
                    value_estimation_loss_coef = 0.2,
                    optimizer = self._opt,
                    advantage_fn = None,
                    use_advantage_loss = False,
                    gamma = 1.0, #discount factor for future returns
                    normalize_returns = False, #The instruction says it's better to normalize
                    gradient_clipping = None,
                    entropy_regularization = None,
                    train_step_counter = train_step_counter
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

class lr_schedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_lr, C):
        self.initial_learning_rate = initial_lr
        self.C = C
    def __call__(self, step):
        return self.initial_learning_rate*self.C/(self.C+step)
