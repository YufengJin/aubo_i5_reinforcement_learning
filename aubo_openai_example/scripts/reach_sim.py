#! /usr/bin/env python

from gym import utils
import math
import rospy
import aubo_simple_env
import aubo_env
from gym.envs.registration import register
import numpy as np

max_episode_steps = 1000 # Can be any Value

register(
        id='ReachSim-v0',
        entry_point='reach_sim:ReachSimEnv',
        max_episode_steps=max_episode_steps,
    )


# class ReachSimEnv(aubo_simple_env.AuboSimpleEnv):
#     def __init__(self):
#         rospy.loginfo("Entered CubePush Env")
        
#         initial_gripper_pos = [0., 0., 1.2, 0.8]


#         aubo_simple_env.AuboSimpleEnv.__init__(self, gripper_block = True, object_name = "block", 
#                                                 has_object = False, reward_type = 'dense', initial_gripper_pos = initial_gripper_pos, 
#                                                target_range = 0.2,  target_in_the_air = False, height_offset= 0, distance_threshold = 0.05, target_offset = None)


class ReachSimEnv(aubo_env.AuboEnv):
    def __init__(self):
        rospy.loginfo("Entered CubePush Env")
        
        initial_gripper_pos = [0., 0., 1.2, 0.8]
        

        aubo_env.AuboEnv.__init__(self, gripper_block = True, object_name = "block", 
                                                has_object = False, reward_type = 'sparse', initial_gripper_pos = initial_gripper_pos, 
                                               target_range = 0.2,  target_in_the_air = False, height_offset= 0, distance_threshold = 0.05, target_offset = None)