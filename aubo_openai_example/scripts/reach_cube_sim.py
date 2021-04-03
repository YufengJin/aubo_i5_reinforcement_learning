#! /usr/bin/env python

from gym import utils
import math
import rospy
import aubo_simple_env
from gym.envs.registration import register
import numpy as np

max_episode_steps = 1000 # Can be any Value

register(
        id='ReachCubeSim-v0',
        entry_point='reach_cube_sim:ReachCubeSimEnv',
        max_episode_steps=max_episode_steps,
    )


class ReachCubeSimEnv(aubo_simple_env.AuboSimpleEnv):
    def __init__(self):
        rospy.loginfo("Entered CubePush Env")
        
        initial_gripper_pos = [0., 0., 1.1, 0.8]


        aubo_simple_env.AuboSimpleEnv.__init__(self, gripper_block = True, object_name = "block", 
                                                has_object = False, reward_type = 'sparse', initial_gripper_pos = initial_gripper_pos, 
                                               target_range = 0.1,  target_in_the_air = False, height_offset= 0.7725, distance_threshold = 0.05, target_offset = None)