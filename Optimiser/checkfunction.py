import numpy as np
from tf_agents.environments import py_environment
from tf_agents.environments import tf_py_environment
from tf_agents.trajectories import StepType

def sample_random_action(spec):
    """Draw one uniform random integer action respecting a BoundedArraySpec."""
    # spec.minimum and spec.maximum may be scalars or arrays
    low = np.array(spec.minimum, dtype=spec.dtype)
    high = np.array(spec.maximum, dtype=spec.dtype)
    # +1 because np.randint upper bound is exclusive
    return np.random.randint(low, high + 1, size=spec.shape, dtype=spec.dtype)

def test_env(env, max_steps=10):
    """
    Run a little sanity‐check loop on any PyEnvironment.

    Args:
      env:    an instance of tf_agents.environments.py_environment.PyEnvironment
      max_steps: how many random steps to take before giving up

    Prints out specs, reset state, then a sequence of (action, state, reward, step_type).
    """
    np.random.seed(0)
    print("=== ACTION SPEC ===")
    print(env.action_spec())
    print("\n=== OBSERVATION SPEC ===")
    print(env.observation_spec())
    print()

    # 1) RESET
    timestep = env.reset()
    print("RESET →")
    print("  state    =", timestep.observation)
    print("  reward   =", timestep.reward)
    print("  stepType =", timestep.step_type)
    print()

    # 2) STEP RANDOMLY
    for i in range(1, max_steps+1):
        a = sample_random_action(env.action_spec())
        timestep = env.step(a)
        print(f"STEP {i:2d}")
        print("  action   =", a)
        print("  new state=", timestep.observation)
        print("  reward   =", timestep.reward)
        print("  stepType =", timestep.step_type)
        print()
        if timestep.step_type == StepType.LAST:
            print(f"Episode terminated at step {i}.")
            break
    else:
        print(f"Did not terminate in {max_steps} steps.")

# Function to examine a TF‑Agents ActorDistributionNetwork
def examine_actor_network(actor_net):
    """
    Print a high‑level summary, dig into the projection sub‑network,
    and list all trainable variables of a TF‑Agents ActorDistributionNetwork.
    """
    print("\n=== ActorDistributionNetwork Summary ===")
    try:
        actor_net.summary()
    except Exception:
        print("No .summary() on actor_net; printing repr instead:\n", actor_net)
    
    print("\n=== Projection Sub‑network ===")
    # TF‑Agents uses _projection_networks attribute to hold projection layers
    proj_nets = getattr(actor_net, "_projection_networks", None)
    if proj_nets is None:
        # Fallback for different TF‑Agents versions
        proj_nets = getattr(actor_net, "_projection_net", None)
    print("Projection network object:", proj_nets)
    
    if hasattr(proj_nets, "summary"):
        try:
            proj_nets.summary()
        except Exception:
            pass
    else:
        # Fall back to listing layers
        layers = getattr(proj_nets, "layers", [])
        if layers:
            print("Layers in projection network:")
            for layer in layers:
                print(f"  • {layer.name}: {layer}")
        else:
            print("No accessible .layers on projection network.")
    
    print("\n=== Trainable Variables ===")
    for var in actor_net.trainable_variables:
        print(f"{var.name}  {var.shape}")
    
    print("\n Examination complete.\n")

# Function to compare greedy vs. stochastic sampling from a TF-Agents policy
def examine_policy_sampling(agent, env, num_samples=10):
    """
    Prints the greedy action from agent.policy and a number of stochastic samples
    from agent.collect_policy at the initial state of the environment.
    
    Args:
      agent: A TF-Agents agent with `policy` and `collect_policy`.
      env: A TF-Agents TFEnvironment (e.g. TFPyEnvironment).
      num_samples: Number of stochastic samples to draw.
    """
    # Reset environment to get initial TimeStep
    time_step = env.reset()
    print("Greedy vs. sampled actions at initial state:\n")
    
    # Greedy "best guess" action
    greedy_action = agent.policy.action(time_step).action.numpy()
    print("  Greedy action =", greedy_action)
    
    # Stochastic samples
    for i in range(num_samples):
        sample_action = agent.collect_policy.action(time_step).action.numpy()
        print(f"  Sample {i+1:2d} →", sample_action)

# Example usage:
# examine_policy_sampling(REINFORCE_agent, train_env, num_samples=10)

