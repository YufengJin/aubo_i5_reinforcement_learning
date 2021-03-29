# -*- coding: utf-8 -*-

import gym
import random
import torch
import rospy
import wandb
import numpy as np
from collections import deque
import push_cube_sim
from ddpg_agent import Agent

N_EPISODES = 1000
PRINT_EVERY = 10
wandb.init(name='DDPG', project="PushCubeSim")



def main():
    #rospy.init_node("train_aubo_pick_place", log_level=rospy.ERROR)
    rospy.init_node("train_aubo_push")

    env = gym.make('PushCubeSim-v0')
    env.seed(2)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]
    action_bound = {'low': env.action_space.low,
                    'high': env.action_space.high}
    agent = Agent(state_size=state_size, action_size=action_size, random_seed=2, action_bound=action_bound)

    """###  Load the saved torch file for actor and critic"""

    # agent.actor_local.load_state_dict(torch.load('checkpoint_actor.pth'))
    # agent.critic_local.load_state_dict(torch.load('checkpoint_critic.pth'))

    scores_deque = deque(maxlen=PRINT_EVERY)
    for i_episode in range(1, N_EPISODES+1):
        state = env.reset()
        agent.reset()
        score = 0
        for i in range(30):
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break 
        wandb.log({'Reward': score})
        scores_deque.append(score)
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)), end="")
        torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
        torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')
        if i_episode % PRINT_EVERY == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
            



if __name__ == "__main__":
    main()

"""###  Explore

In this exercise, we have provided a sample DDPG agent and demonstrated how to use it to solve an OpenAI Gym environment.  To continue your learning, you are encouraged to complete any (or all!) of the following tasks:
- Amend the various hyperparameters and network architecture to see if you can get your agent to solve the environment faster than this benchmark implementation.  Once you build intuition for the hyperparameters that work well with this environment, try solving a different OpenAI Gym task!
- Write your own DDPG implementation.  Use this code as reference only when needed -- try as much as you can to write your own algorithm from scratch.
- You may also like to implement prioritized experience replay, to see if it speeds learning.  
- The current implementation adds Ornsetein-Uhlenbeck noise to the action space.  However, it has [been shown](https://blog.openai.com/better-exploration-with-parameter-noise/) that adding noise to the parameters of the neural network policy can improve performance.  Make this change to the code, to verify it for yourself!
- Write a blog post explaining the intuition behind the DDPG algorithm and demonstrating how to use it to solve an RL environment of your choosing.  
"""