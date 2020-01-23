# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

import os
from tqdm import tqdm

from rl_glue import RLGlue
from environment import BaseEnvironment
from agent import BaseAgent
from dummy_environment import DummyEnvironment
from dummy_agent import DummyAgent


def run_experiment(environment, agent, environment_parameters, agent_parameters, experiment_parameters):
    
    """
    Assume environment_parameters dict contains:
    {
        input_dim: integer,
        num_actions: integer,
        discount_factor: float
    }
    
    Assume agent_parameters dict contains:
    {
        step_size: 1D numpy array of floats,
        tau: 1D numpy array of floats
    }
    
    Assume experiment_parameters dict contains:
    {
        num_runs: integer,
        num_episodes: integer
    }    
    """
    
    ### Instantiate rl_glue from RLGlue    
    rl_glue = RLGlue(environment, agent)

    os.system('sleep 1') # to prevent tqdm printing out-of-order
        
     ### Initialize agent_sum_reward to zero in the form of a numpy array 
    # with shape (number of values for tau, number of step-sizes, number of runs, number of episodes)
    agent_sum_reward = np.zeros((len(agent_parameters["tau"]), len(agent_parameters["step_size"]), experiment_parameters["num_runs"], experiment_parameters["num_episodes"]))
    
    # for loop over different values of tau
    # tqdm is used to show a progress bar for completing the parameter study
    for i in tqdm(range(len(agent_parameters["tau"]))):
    
        # for loop over different values of the step-size
        for j in range(len(agent_parameters["step_size"])): 

            ### Specify env_info 
            env_info = {}

            ### Specify agent_info
            agent_info = {"num_actions": environment_parameters["num_actions"],
                          "input_dim": environment_parameters["input_dim"],
                          "discount_factor": environment_parameters["discount_factor"],
                          "tau": agent_parameters["tau"][i],
                          "step_size": agent_parameters["step_size"][j]}

            # for loop over runs
            for run in range(experiment_parameters["num_runs"]): 
                
                # Set the seed
                agent_info["seed"] = agent_parameters["seed"] * experiment_parameters["num_runs"] + run
                
                # Beginning of the run            
                rl_glue.rl_init(agent_info, env_info)

                for episode in range(experiment_parameters["num_episodes"]): 
                    
                    # Run episode
                    rl_glue.rl_episode(0) # no step limit

                    ### Store sum of reward
                    agent_sum_reward[i, j, run, episode] = rl_glue.rl_agent_message("get_sum_reward")

            if not os.path.exists('results'):
                    os.makedirs('results')

            save_name = "{}".format(rl_glue.agent.name).replace('.','')

            # save sum reward
            np.save("results/sum_reward_{}".format(save_name), agent_sum_reward)



# Experiment parameters
experiment_parameters = {
    "num_runs" : 100,
    "num_episodes" : 100,
}

# Environment parameters
environment_parameters = {
    "input_dim" : 8,
    "num_actions": 4, 
    "discount_factor" : 0.99
}

agent_parameters = {
    "step_size": 3e-5 * np.power(2.0, np.array([-6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5])),
    "tau": np.array([0.001, 0.01, 0.1, 1.0]),
    "seed": 0
}

test_env = DummyEnvironment
test_agent = DummyAgent

run_experiment(test_env, 
               test_agent, 
               environment_parameters, 
               agent_parameters, 
               experiment_parameters)

sum_reward_dummy_agent = np.load("results/sum_reward_dummy_agent.npy")
sum_reward_dummy_agent_answer = np.load("asserts/sum_reward_dummy_agent.npy")
assert(np.allclose(sum_reward_dummy_agent, sum_reward_dummy_agent_answer))

print("Passed the assert!")

