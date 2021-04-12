import gym
import numpy as np
import rospy
import push_cube_sim
import reach_sim

from stable_baselines3 import SAC

rospy.init_node("train_aubo_taskEnv")
env_name = 'ReachSim-v0'

env = gym.make(env_name)


model = SAC("MlpPolicy", env, verbose=1, tensorboard_log="./sac_reach_tensorboard/")
model.learn(total_timesteps=4e5, log_interval=4)
model.save("sac_reachsim")


#model = SAC.load("sac_reachsim")
