import numpy as np
import Optimiser.config
from Optimiser.config import get_config
from Optimiser.evaluator import f
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.trajectories.time_step import StepType
from typing import Callable

cfg = get_config()

def compute_reward(state):
    '''
    The function to compute the reward R(S_t,A_t) at time t.
    
    Input.
    --------
    state: S_t, a 2d-array.
    
    Output.
    --------
    R: a float value, the reward at time t.
    '''
    #For this case, we set the reward as the negative of the sum of squares of the state vector
    #This is just an example, you can change it to any other function
    return f(state)

class Env_Continue(py_environment.PyEnvironment):
    def __init__(self, reward_fn: Callable[[np.ndarray], float], sub_episode_length:int = 0):
        env_cfg = cfg["env"]
        self._state_dim = env_cfg["state_dim"]
        self._x0_reinforce = env_cfg["x0_reinforce"]
        if sub_episode_length <= 0:
            self._sub_episode_length = cfg["training"]["sub_episode_length"]
        else:
            self._sub_episode_length = sub_episode_length
        '''The function to initialize an Env obj.
        '''
        #Specify the requirement for the value of action, (It is a 2d-array for this case)
        #which is an argument of _step(self, action) that is later defined.
        #tf_agents.specs.BoundedArraySpec is a class.
        #_action_spec.check_array( arr ) returns true if arr conforms to the specification stored in _action_spec
        self._action_spec = array_spec.BoundedArraySpec(
                            shape=(self._state_dim,), dtype=np.float32, minimum=np.array([-1.0]) * self._state_dim, maximum=np.array([1.0]) * self._state_dim, name='action') #a_t is an 2darray

        #Specify the format requirement for observation (It is a 2d-array for this case), 
        #i.e. the observable part of S_t, and it is stored in self._state
        self._observation_spec = array_spec.BoundedArraySpec(
                                 shape=(self._state_dim,), dtype=np.float32, minimum=np.array([-1.0]) * self._state_dim, maximum=np.array([1.0]) * self._state_dim, name='observation') #default max and min is None
        self._state = np.array(self._x0_reinforce,dtype=np.float32)
        #self.A = mat
        self._episode_ended = False
        #stop_threshold is a condition for terminating the process for looking for the solution
        #self._stop_threshold = 0.01
        self._step_counter = 0
        self._reward_fn = reward_fn
        self._disc_factor = np.float32(cfg["training"]["disc_factor"])

    def action_spec(self):
        #return the format requirement for action
        return self._action_spec

    def observation_spec(self):
        #return the format requirement for observation
        return self._observation_spec
    


        ####Origin code on  restting the environment
    def _reset(self):
        self._state = np.array(self._x0_reinforce,dtype=np.float32)  #initial state
        self._episode_ended = False
        self._step_counter = 0
        
        #Reward
        initial_r = np.float32(0.0)
        
        #return ts.restart(observation=np.array(self._state, dtype=np.float32))
        return ts.TimeStep(step_type=StepType.FIRST, 
                           reward=initial_r, 
                           discount=np.float32(self._disc_factor), 
                           observation=np.array(self._state, dtype=np.float32)
                           )
    # updated on 14/July, due to think to expanding the evn by reseting the state to explorer the space more

    # def _reset(self):
    #     # sample each coordinate uniformly between min and max
    #     self.state = np.random.uniform(self._act_min, self._act_max, size=self._state_dim).astype(np.float32)
    #     self._episode_ended = False
    #     self._step_count = 0

    #     #     #Reward
    #     initial_r = np.float32(0.0)
    #     return ts.TimeStep(step_type=StepType.FIRST, 
    #                        reward=initial_r, 
    #                        discount=np.float32(self._disc_factor), 
    #                        observation=np.array(self._state, dtype=np.float32)
    #                        )    

    
    def set_state(self,new_state):
        self._state = new_state
    
    def get_state(self):
        return self._state
    
    def _step(self, action):
        '''
        The function for the transtion from (S_t, A_t) to (R_{t+1}, S_{t+1}).
    
        Input.
        --------
        self: contain S_t.
        action: A_t.
    
        Output.
        --------
        an TimeStep obj, TimeStep(step_type_{t+1}, R_{t+1}, discount_{t+1}, observation S_{t+1})
        '''

        # Suppose that we are at the beginning of time t 
        
        ################## --- Determine whether we should end the episode.
        if self._episode_ended:  # its time-t value is set at the end of t-1
            return self.reset()
        # Move on to the following if self._episode_ended=False
        
        
        ################# --- Compute S_{t+1} 
        self._state = self._state + action    
        self._step_counter +=1
        
        ################# --- Compute R_{t+1}=R(S_t,A_t)
        R = np.float32(self._reward_fn(self._state))  #is_normalised is set to False, since the reward function is not normalised
        #Set conditions for termination
        if self._step_counter>=self._sub_episode_length-1:
            self._episode_ended = True  #value for t+1

 
        #Now we are at the end of time t, when self._episode_ended may have changed
        if self._episode_ended:
            #if self._step_counter>100:
            #    reward += np.float32(-100)
            #ts.termination(observation,reward,outer_dims=None): Returns a TimeStep obj with step_type set to StepType.LAST.
            return ts.termination(np.array(self._state, dtype=np.float32), reward=R)
        else:
            #ts.transition(observation,reward,discount,outer_dims=None): Returns 
            #a TimeStep obj with step_type set to StepType.MID.
            return ts.transition(np.array(self._state, dtype=np.float32), reward=R, discount=self._disc_factor)

