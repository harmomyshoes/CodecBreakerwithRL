import numpy as np
import config
from config import get_config
from evaluator import f
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.trajectories.time_step import StepType

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

class Env_Discrete(py_environment.PyEnvironment):
    def __init__(self):
        '''The function to initialize an Env obj.
        '''
        #Specify the requirement for the value of action, (It is a 2d-array for this case)
        #which is an argument of _step(self, action) that is later defined.
        #tf_agents.specs.BoundedArraySpec is a class.
        #_action_spec.check_array( arr ) returns true if arr conforms to the specification stored in _action_spec
        #self._action_spec = array_spec.BoundedArraySpec(
        #                    shape=(state_dim,), dtype=np.int32, 
        #                    minimum=np.array([0]*N), 
        #                    maximum=act_max-act_min, 
        #                    name='action') #a_t is an 2darray
        env_cfg = cfg["env"]
        self._state_dim = env_cfg["state_dim"]
        self._act_min   = np.array(env_cfg["act_min"], dtype=np.int32)
        self._act_max   = np.array(env_cfg["act_max"], dtype=np.int32)
        self._x0_reinforce = env_cfg["x0_reinforce"]
        self._step_size = np.array(env_cfg["step_size"], dtype=np.float32)

        self._action_spec = array_spec.BoundedArraySpec(
                                shape=(self._state_dim,),
                                dtype=np.int32,
                                minimum=np.array([0, 0, 0], dtype=np.int32),
                                maximum=self._act_max-self._act_min,
                                name='action')
    
        #Specify the format requirement for observation (It is a 2d-array for this case), 
        #i.e. the observable part of S_t, and it is stored in self._state
        self._observation_spec = array_spec.BoundedArraySpec(
                                 shape=(self._state_dim,), dtype=np.float32, name='observation') #default max and min is None
        self._state = np.array(self._x0_reinforce,dtype=np.float32)
        #self.A = mat
        self._episode_ended = False
        #stop_threshold is a condition for terminating the process for looking for the solution
        #self._stop_threshold = 0.01
        self._step_counter = 0

    def action_spec(self):
        #return the format requirement for action
        return self._action_spec

    def observation_spec(self):
        #return the format requirement for observation
        return self._observation_spec

    def _reset(self):
        self._state = np.array(self._x0_reinforce,dtype=np.float32)  #initial state
        self._episode_ended = False
        self._step_counter = 0
        
        #Reward
        initial_r = np.float32(0.0)
        
        #return ts.restart(observation=np.array(self._state, dtype=np.float32))
        return ts.TimeStep(step_type=StepType.FIRST, 
                           reward=initial_r, 
                           discount=np.float32(cfg["training"]["disc_factor"]), 
                           observation=np.array(self._state, dtype=np.float32)
                           )
    
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
        #Note that we set the action space as a set of non-negative vectors
        #action-act_max converts the set to the desired set of negative vectors.

        self._state = self._state + (action-self._act_max)*self._step_size    
        self._step_counter +=1
        
        ################# --- Compute R_{t+1}=R(S_t,A_t)
        R = compute_reward(self._state)
        
        #Set conditions for termination
        if self._step_counter>=cfg["training"]["sub_episode_length"]-1:
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
            return ts.transition(np.array(self._state, dtype=np.float32), reward=R, discount=cfg["training"]["disc_factor"])

class Env_Continue(py_environment.PyEnvironment):
    def __init__(self):
        env_cfg = cfg["env"]
        self._state_dim = env_cfg["state_dim"]
        self._act_min   = np.array(env_cfg["act_min"], dtype=np.float32)
        self._act_max   = np.array(env_cfg["act_max"], dtype=np.float32)
        self._x0_reinforce = env_cfg["x0_reinforce"]
        '''The function to initialize an Env obj.
        '''
        #Specify the requirement for the value of action, (It is a 2d-array for this case)
        #which is an argument of _step(self, action) that is later defined.
        #tf_agents.specs.BoundedArraySpec is a class.
        #_action_spec.check_array( arr ) returns true if arr conforms to the specification stored in _action_spec
        self._action_spec = array_spec.BoundedArraySpec(
                            shape=(self._state_dim,), dtype=np.float32, minimum=np.array(self._act_min), maximum=np.array(self._act_max), name='action') #a_t is an 2darray

        #Specify the format requirement for observation (It is a 2d-array for this case), 
        #i.e. the observable part of S_t, and it is stored in self._state
        self._observation_spec = array_spec.BoundedArraySpec(
                                 shape=(self._state_dim,), dtype=np.float32, name='observation') #default max and min is None
        self._state = np.array(self._x0_reinforce,dtype=np.float32)
        #self.A = mat
        self._episode_ended = False
        #stop_threshold is a condition for terminating the process for looking for the solution
        #self._stop_threshold = 0.01
        self._step_counter = 0

    def action_spec(self):
        #return the format requirement for action
        return self._action_spec

    def observation_spec(self):
        #return the format requirement for observation
        return self._observation_spec

    def _reset(self):
        self._state = np.array(self._x0_reinforce,dtype=np.float32)  #initial state
        self._episode_ended = False
        self._step_counter = 0
        
        #Reward
        initial_r = np.float32(0.0)
        
        #return ts.restart(observation=np.array(self._state, dtype=np.float32))
        return ts.TimeStep(step_type=StepType.FIRST, 
                           reward=initial_r, 
                           discount=np.float32(cfg["training"]["disc_factor"]), 
                           observation=np.array(self._state, dtype=np.float32)
                           )
    
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
        R = compute_reward(self._state)
        
        #Set conditions for termination
        if self._step_counter>=cfg["training"]["sub_episode_length"]-1:
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
            return ts.transition(np.array(self._state, dtype=np.float32), reward=R, discount=cfg["training"]["disc_factor"])
