#! /usr/bin/env python

from gym import utils
import math
import rospy
from gym import spaces
import aubo_env
from gym.envs.registration import register
import numpy as np

max_episode_steps = 1000 # Can be any Value

register(
        id='AuboPushCube-v0',
        entry_point='aubo_push:AuboPushCubeEnv',
        max_episode_steps=max_episode_steps,
    )


class AuboPushCubeEnv(aubo_env.AuboEnv):
    def __init__(self):
        
        aubo_env.AuboEnv.__init__(self, gripper_block = True, action_type = "ee_control")

        rospy.loginfo("Entered CubePush Env")

        self.gazebo.unpauseSim()
        
        obs = self._get_obs()


    def calc_dist(self,p1,p2):
        """
        d = ((2 - 1)2 + (1 - 1)2 + (2 - 0)2)1/2
        """
        x_d = math.pow(p1[0] - p2[0],2)
        y_d = math.pow(p1[1] - p2[1],2)
        z_d = math.pow(p1[2] - p2[2],2)
        d = math.sqrt(x_d + y_d + z_d)

        return d


    
    def get_elapsed_time(self):
        """
        Returns the elapsed time since the beginning of the simulation
        Then maintains the current time as "previous time" to calculate the elapsed time again
        """
        current_time = rospy.get_time()
        dt = self.sim_time - current_time
        self.sim_time = current_time
        return dt

    def _is_done(self, observations):
        """
        If the latest Action didnt succeed, it means that tha position asked was imposible therefore the episode must end.
        It will also end if it reaches its goal.
        """

        distance = observations[0]
        speed = observations[1]

        # Did the movement fail in set action?
        done_fail = not(self.movement_result)

        done_success = speed >= self.max_speed

        print(">>>>>>>>>>>>>>>>done_fail="+str(done_fail)+",done_sucess="+str(done_success))
        # If it moved or the arm couldnt reach a position asced for it stops
        done = done_fail or done_success

        return done

    def _compute_reward(self, observations, done):
        """
        Reward moving the cube
        Punish movint to unreachable positions
        Calculate the reward: binary => 1 for success, 0 for failure
        """
        distance = observations[0]
        speed = observations[1]
        ee_z_pos = observations[2]

        # Did the movement fail in set action?
        exec_fail = not(self.movement_result)

        done_sucess = speed >= self.max_speed

        if exec_fail:
            # We punish that it trie sto move where moveit cant reach
            reward = self.impossible_movement_punishement
        else:
            if done_sucess:
                #It moved the cube
                reward = -1*self.impossible_movement_punishement
            else:
                if ee_z_pos < self.ee_z_min or ee_z_pos >= self.ee_z_max:
                    print("Punish, ee z too low or high..."+str(ee_z_pos))
                    reward = self.impossible_movement_punishement / 4.0
                else:
                    # It didnt move the cube. We reward it by getting closser
                    print("Reward for getting closser")
                    reward = 1.0 / abs(distance-0.13)

        return reward