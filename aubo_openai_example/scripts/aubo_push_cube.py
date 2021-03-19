#! /usr/bin/env python

from gym import utils
import math
import rospy
from gym import spaces
import aubo_env
from gym.envs.registration import register
import numpy as np

from obj_positions import Obj_Pos

max_episode_steps = 1000 # Can be any Value

register(
        id='AuboPushCube-v0',
        entry_point='aubo_push:AuboPushCubeEnv',
        max_episode_steps=max_episode_steps,
    )


#class AuboPushCubeEnv(aubo_env.AuboEnv, utils.EzPickle):
class AuboPushCubeEnv(aubo_env.AuboEnv):
    def __init__(self):
        
        super(AuboPushCubeEnv, self).__init__()
        #super(utils.EzPickle,self).__init__()
        
        self.gripper_block = True

        rospy.loginfo("Entered CubePush Env")
        self.obj_positions = Obj_Pos(object_name="block")

        self.get_params()

        self.gazebo.unpauseSim()

        # define working space for aciton
        x_low = np.array(self.x_min)
        x_high = np.array(self.x_max)
        y_low = np.array(self.y_min)
        y_high = np.array(self.y_max)
        z_low = np.array(self.z_min)
        z_high = np.array(self.z_max)
        ee_low = np.array(self.ee_open)
        ee_high= np.array(self.ee_close)

        pos_low = np.concatenate([x_low, y_low, z_low, ee_low])
        pos_high = np.concatenate([x_high, y_high, z_high, ee_high])
        
        self.action_space = spaces.Box(
            low=pos_low,
            high=pos_high, shape = (self.n_actions,), dtype=np.float32)

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
        # set limits for end_effector(working space above the table)
        self.x_max = 0.75 + 0.91/2
        self.x_min = 0.75 - 0.91/2
        self.y_max = 0.91/2
        self.y_min = - 0.91/2
        self.z_max = 1.3
        self.z_min = 0.77

        # gripper maximum and minimum 
        self.ee_close = 0.8
        self.ee_open = 0.0

        self.sim_time = rospy.get_time()
        self.n_actions = 4
        self.n_observations = 3
 


        # ee postion and gripper (x,y,z,ee)
        self.setup_ee_pos = [0.5, 0, 1.1, 0.8]

        self.position_delta = 0.1
        self.step_punishment = -1
        self.closer_reward = 10
        self.impossible_movement_punishement = -100
        self.reached_goal_reward = 100

        self.max_distance = 3.0
        self.max_speed = 1.0
        # 1.5 aubo can reach highest height
        self.ee_z_max = 1.0
        # the height of table 
        self.ee_z_min = 0.8



    def _set_init_pose(self):
        """
        Sets the Robot in its init pose
        The Simulation will be unpaused for this purpose.
        """
        self.gazebo.unpauseSim()
        ee_postion = self.init_pos[:3]
        ee_value = self.init_pos[-1]
        if not (self.set_trajectory_ee(ee_postion) and self.set_ee(ee_value)):
            assert False, "Initialisation is failed...."

    def _init_env_variables(self):
        """
        Inits variables needed to be initialised each time we reset at the start
        :return:
        """
        rospy.logdebug("Init Env Variables...")
        rospy.logdebug("Init Env Variables...END")

    def _set_action(self, action):

        if self.gripper_block:
            # if gripper block, action spaces should be 3
            assert len(action) == 3, 'Action should be 3 dimensions, gripper in this task not working'
            self.movement_result = self.set_trajectory_ee(action)

        else:
            # if gripper not block, action spaces should be 4, 
            assert len(action) == 4, 'Action should be 4 dimensions, gripper in this task not working'
            self.movement_result = self.set_trajectory_ee(action[:3]) and self.set_ee[-1]



    def _get_obs(self):
        """
        It returns the Position of the TCP/EndEffector as observation.
        And the speed of cube
        Orientation for the moment is not considered
        """
        self.gazebo.unpauseSim()

        grip_pose = self.get_ee_pose()
        ee_array_pose = [grip_pose.position.x, grip_pose.position.y, grip_pose.position.z]

        # the pose of the cube/box on a table        
        object_data = self.obj_positions.get_states()

        # speed cube
        object_pos = object_data[3:]

        distance_from_cube = self.calc_dist(object_pos,ee_array_pose)


        object_velp = object_data[-3:]
        speed = np.linalg.norm(object_velp)

        # We state as observations the distance form cube, the speed of cube and the z postion of the end effector
        observations_obj = np.array([distance_from_cube,
                             speed, ee_array_pose[2]])

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