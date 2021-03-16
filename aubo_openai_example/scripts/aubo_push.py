#! /usr/bin/env python

from gym import utils
import math
import rospy
from gym import spaces
import aubo_env
from gym.envs.registration import register
import numpy as np

from cube_positions import Obj_Pos

max_episode_steps = 1000 # Can be any Value

register(
        id='AuboPush-v0',
        entry_point='aubo_push:AuboPushEnv',
        max_episode_steps=max_episode_steps,
    )


#class AuboPushEnv(aubo_env.AuboEnv, utils.EzPickle):
class AuboPushEnv(aubo_env.AuboEnv):
    def __init__(self):
        
        super(AuboPushEnv, self).__init__()
        #super(utils.EzPickle,self).__init__()
        
        rospy.loginfo("Entered Push Env")
        self.obj_positions = Obj_Pos(object_name="block")

        self.get_params()


        self.gazebo.unpauseSim()

      # self.action_space = spaces.Discrete(self.n_actions)
        self.action_space = spaces.Box(
            low=self.position_joints_min,
            high=self.position_joints_max, shape=(self.n_actions,),
            dtype=np.float32
        )

        # distance between ee and block
        observations_high_dist = np.array([self.max_distance])
        observations_low_dist = np.array([0.0])
        # speed of block
        observations_high_speed = np.array([self.max_speed])
        observations_low_speed = np.array([0.0])

        observations_ee_z_max = np.array([self.ee_z_max])
        observations_ee_z_min = np.array([self.ee_z_min])

        high = np.concatenate([observations_high_dist, observations_high_speed, observations_ee_z_max])
        low = np.concatenate([observations_low_dist, observations_low_speed, observations_ee_z_min])

        self.observation_space = spaces.Box(low, high)

        obs = self._get_obs()


        

    def get_params(self):
        """
        get configuration parameters

        """
        # set limits for joint
        self.position_joints_max = 2.16
        self.position_joints_min = 2.16


        self.sim_time = rospy.get_time()
        self.n_actions = 6
        self.n_observations = 3
        self.position_ee_max = 10.0
        self.position_ee_min = -10.0


        self.init_pos = {   "shoulder_joint": 0.0,
                            "upperArm_joint": 0.0,
                            "foreArm_joint": 0.6,
                            "wrist1_joint": 0.0,
                            "wrist2_joint": 1.53,
                            "wrist3_joint": 0.0,
                            "gripper":0.8}
        """
        self.setup_ee_pos = {"x": 0.598,
                            "y": 0.005,
                            "z": 0.9}
        """

        self.position_delta = 0.1
        self.step_punishment = -1
        self.closer_reward = 10
        self.impossible_movement_punishement = -100
        self.reached_goal_reward = 100

        self.max_distance = 3.0
        self.max_speed = 1.0
        # 1.5 aubo can reach highest height
        self.ee_z_max = 1.2
        # the height of table 
        self.ee_z_min = 0.7725



    def _set_init_pose(self):
        """
        Sets the Robot in its init pose
        The Simulation will be unpaused for this purpose.
        """
        self.gazebo.unpauseSim()
        if not (self.set_trajectory_joints(self.init_pos) and self.set_ee(self.init_pos["gripper"])):
            assert False, "Initialisation is failed...."

    def _init_env_variables(self):
        """
        Inits variables needed to be initialised each time we reset at the start
        :return:
        """
        rospy.logdebug("Init Env Variables...")
        rospy.logdebug("Init Env Variables...END")

    def _set_action(self, action):

        self.new_pos = {"shoulder_joint": action[0],
                        "upperArm_joint": action[1],
                        "foreArm_joint": action[2],
                        "wrist1_joint": action[3],
                        "wrist2_joint": action[4],
                        "wrist3_joint": action[5]}


        self.movement_result = self.set_trajectory_joints(self.new_pos)


    def _get_obs(self):
        """
        It returns the Position of the TCP/EndEffector as observation.
        And the speed of cube
        Orientation for the moment is not considered
        """
        self.gazebo.unpauseSim()

        (grip_trans, grip_rot) = self.get_tf("world", "robotiq_gripper_center")
        # ee postion in np array
        gripper_trans = np.array([grip_trans[0], grip_trans[1], grip_trans[2]])
        gripper_rot = np.array([grip_rot[0], grip_rot[1], grip_rot[2]], grip_rot[3])

        # the pose of the cube/box on a table        
        object_data = self.obj_positions.get_states()

        # postion of block
        object_pos = object_data[:3]
        #vector of block
        object_vect = object_data[3:]

        distance_from_block = self.calc_dist(object_pos,gripper_trans)

        speed = np.linalg.norm(object_vect)

        # We state as observations the distance form cube, the speed of cube and the z postion of the end effector
        observations_obj = np.array([distance_from_block,
                             speed, gripper_trans[2]])

        return  observations_obj
    
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
        exec_fail = not(self.movement_result)

        done_success = speed >= self.max_speed

        print(">>>>>>>>>>>>>>>>done_fail="+str(done_fail)+",done_sucess="+str(done_sucess))
        # If it moved or the arm couldnt reach a position asced for it stops
        done = exec_fail or done_success

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
                    reward = 1.0 / distance

        return reward