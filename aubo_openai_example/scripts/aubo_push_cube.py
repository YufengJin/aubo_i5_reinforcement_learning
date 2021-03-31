#! /usr/bin/env python

from gym import utils
import math
import rospy
import aubo_env
from gym.envs.registration import register
import numpy as np

max_episode_steps = 1000 # Can be any Value

register(
        id='AuboPushCube-v0',
        entry_point='aubo_push_cube:AuboPushCubeEnv',
        max_episode_steps=max_episode_steps,
    )


class AuboPushCubeEnv(aubo_env.AuboEnv):
    def __init__(self, object_name = 'block'):
        initial_pos = {
            'end_effector' : [-0.1, 0., 1.1, 0.8],
            object_name : [0., 0., 0.8, 0., 0., 0., 0.],
        }
        aubo_env.AuboEnv.__init__(self, object_name = object_name, initial_pos = initial_pos, block_gripper = True,
                                    has_object = True, target_in_the_air = False, target_range = 0.15, 
                                    distance_threshold = 0.05, reward_type = 'sparse')

        rospy.loginfo("Entered AuboPushCube Env")
