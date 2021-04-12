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
        id='PushCubeSim-v0',
        entry_point='push_cube_sim:PushCubeSimEnv',
        max_episode_steps=max_episode_steps,
    )


# class PushCubeSimEnv(aubo_simple_env.AuboSimpleEnv):
#     def __init__(self):
#         rospy.loginfo("Entered CubePush Env")
        
#         initial_gripper_pos = [0., 0., 1.1, 0.8]
#         target_offset = [0, -0.2, 0]

#         aubo_simple_env.AuboSimpleEnv.__init__(self, gripper_block = True, object_name = "block", 
#                                                 has_object = True, reward_type = 'sparse', initial_gripper_pos = initial_gripper_pos, 
#                                                target_range = 0.1,  target_in_the_air = False, height_offset= 0.7725, distance_threshold = 0.05, target_offset = target_offset)

    

class PushCubeSimEnv(aubo_env.AuboEnv):
    def __init__(self):
        rospy.loginfo("Entered CubePush Env")
        
        initial_gripper_pos = [0., 0., 1.1, 0.8]
        target_offset = [0, -0.2, 0]

        aubo_env.AuboEnv.__init__(self, gripper_block = True, object_name = "block", 
                                                has_object = True, reward_type = 'dense', initial_gripper_pos = initial_gripper_pos, 
                                               target_range = 0.1,  target_in_the_air = False, height_offset= 0.7725, distance_threshold = 0.05, target_offset = target_offset)

    
