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
    def __init__(self):
        
        aubo_env.AuboEnv.__init__(self, gripper_block = True, action_type = "ee_control",object_name = "block")

        rospy.loginfo("Entered CubePush Env")

        self.gazebo.unpauseSim()
        
        obs = self._get_obs()
        
        self.cube_desired_goal = {'x': 0.75, 'y':0.1}

        self.distance_threshold = 0.05

        self.cube_move_threshold = 0.01

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

    def _is_success(self, achived_goal, goal):

        done = False

        distance = self.calc_dist(goal,achived_goal)

        if distance < self.distance_threshold:
            done = True

        return done

    def _is_done(self, observations):
        """
        if movement planning fail, it done. and the cube reach the desired position it     
        """

        cube_pos = observations[7:10]

        # Did the movement fail in set action?
        mov_fail = not(self.movement_succees)

        goal = [self.cube_desired_goal['x'],self.cube_desired_goal['y'],cube_pos[2]]

        done_success = self._is_success(cube_pos, goal)

        print(">>>>>>>>>>>>>>>> Movement planning fails: "+str(mov_fail)+", Mission completed: "+str(done_success))
        # If it moved or the arm couldnt reach a position asced for it stops
        done = mov_fail or done_success

        return done

    def _compute_reward(self, observations, done):
        """
        Reward moving the cube
        Punish movint to unreachable positions
        Calculate the reward: binary => 1 for success, 0 for failure
        """

        ee_pos = observations[:3]
 
        cube_pos = observations[7:10]

        cube_vel = np.linalg.norm(observations[-3:])

        goal = [self.cube_desired_goal['x'], self.cube_desired_goal['y'],cube_pos[2]]
        # Did the movement fail in set action?
        exec_fail = not(self.movement_succees)

        done_sucess = self._is_success(cube_pos, goal)

        ee_clo_to_cube = self._is_success(ee_pos,cube_pos)

        if exec_fail:
            # We punish that it trie sto move where moveit cant reach
            reward = self.impossible_movement_punishement
        else:
            if done_sucess:
                #It moved the cube
                reward = self.done_reward
            else:
                if ee_clo_to_cube and cube_vel > self.cube_move_threshold:
                    # ee close to cube
                    reward = 0
                else:
                    # ee didnt get close to cube
                    reward = -1

        return reward