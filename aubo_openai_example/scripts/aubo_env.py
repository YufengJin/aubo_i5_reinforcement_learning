#! /usr/bin/env python

import numpy as np
import rospy
import tf
from std_msgs.msg import Float64
from sensor_msgs.msg import JointState, Image
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped, Pose
from aubo_moveit_config.aubo_commander import AuboCommander
from openai_ros import robot_gazebo_env
from obj_positions import Obj_Pos
from gym import spaces

def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)


def gen_sphere_action(last_action, action, semidia = 0.05):
    delta = action[:3] - last_action[:3]
    result = np.zeros(4)
    if np.linalg.norm(delta) < semidia: 
        result = action
    else:
        scalar = semidia/np.linalg.norm(delta)
        result[:3] = last_action[:3] + scalar * delta
        result[3] = action[3]
    return result

class AuboEnv(robot_gazebo_env.RobotGazeboEnv):

    def __init__(self, object_name, block_gripper, has_object, target_in_the_air, target_range, distance_threshold, initial_pos, reward_type):
        """
        Initializes a new aubo environment.
        
        To check any topic we need to have the simulations running, we need to do two things:
        1) Unpause the simulation: without that th stream of data doesnt flow. This is for simulations
        that are pause for whatever the reason
        2) If the simulation was running already for some reason, we need to reset the controlers.
        This has to do with the fact that some plugins with tf, dont understand the reset of the simulation
        and need to be reseted to work properly.
        
        The Sensors: The sensors accesible are the ones considered usefull for AI learning.
        
        Sensor Topic List:
        * /aubo_i5/camera/image
        
        Actuators Topic List: 
        * /aubo_i5/arm_controller/follow_joint_trajectory
        * /aubo_i5/gripper_controller/gripper_cmd

        

       Args:
            model_path (string): path to the environments XML file
            block_gripper (boolean): whether or not the gripper is blocked (i.e. not movable) or not
            has_object (boolean): whether or not the environment has an object
            target_in_the_air (boolean): whether or not the target should be in the air above the table or on the table surface
            target_range (float): range of a uniform distribution for sampling a target
            distance_threshold (float): the threshold after which a goal is considered achieved
            initial_qpos (dict): a dictionary of joint names and values that define the initial configuration
            reward_type ('sparse' or 'dense'): the reward type, i.e. sparse or dense
        """
        self.block_gripper = block_gripper
        self.has_object = has_object
        self.target_in_the_air = target_in_the_air
        self.target_range = target_range
        self.distance_threshold = distance_threshold
        self.reward_type = reward_type
        self.object_name = object_name
        self.initial_gripper_pos = initial_pos['end_effector']
        self.initial_object_pos = initial_pos[object_name]

        self.controllers_list = []

        self.robot_name_space = ""
        
        self.aubo_commander = AuboCommander()
        # add collision with table mesh
        self.setup_planning_scene()
        # It doesnt use namespace

        self.listener = tf.TransformListener()
        # create object class
        if self.has_object:
            self.obj = Obj_Pos(object_name = object_name)

        self.height_offset = 0.7725
        # We launch the init function of the Parent Class robot_gazebo_env.RobotGazeboEnv
        super(AuboEnv, self).__init__(  controllers_list=self.controllers_list,
	                                    robot_name_space=self.robot_name_space,
	                                    reset_controls=False,
	                                    start_init_physics_parameters=False,
	                                    reset_world_or_sim="WORLD")





        rospy.logdebug("Start AuboEnv INIT...")

        GIPPER_IMAGE_SUBSCRIBER = '/camera/image_raw'
       


        self.grippper_camera_image_raw = rospy.Subscriber(GIPPER_IMAGE_SUBSCRIBER, Image, self.gripper_camera_callback)
        self.grippper_camera_image_raw = Image()

        action_space_low = np.array([-0.25,-0.25,0.95,0.])
        action_space_high = np.array([0.25,0.25,1.2,0.8])
        obs = self._get_obs()

        self.action_space = spaces.Box(action_space_low, action_space_high, shape=(4,), dtype='float32')
        self.observation_space = spaces.Dict(dict(
            desired_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
            achieved_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
            observation=spaces.Box(-np.inf, np.inf, shape=obs['observation'].shape, dtype='float32'),
        ))

    def get_params(self):
        """
        get configuration parameters

        """
        # set limits for end_effector(working space above the table)
        self.x_max = 0.75
        self.x_min = 0.75 - 0.91/2
        self.y_max = 0.9/2
        self.y_min = - 0.9/2
        self.z_max = 1.1
        self.z_min = 0.95

        # gripper maximum and minimum 
        self.ee_close = 0.8
        self.ee_open = 0.0

        self.done_reward = 100

    def set_action_observation_space(self):

        # define working space for aciton
        x_low = np.array([self.x_min])
        x_high = np.array([self.x_max])
        y_low = np.array([self.y_min])
        y_high = np.array([self.y_max])
        z_low = np.array([self.z_min])
        z_high = np.array([self.z_max])
        ee_low = np.array([self.ee_open])
        ee_high= np.array([self.ee_close])

        pos_low = np.concatenate([x_low, y_low, z_low, ee_low])
        pos_high = np.concatenate([x_high, y_high, z_high, ee_high])
        
        self.action_space = spaces.Box(
            low=pos_low,
            high=pos_high, shape=(4,), dtype='float32')

        self.observation_space = spaces.Box(-np.inf, np.inf, shape = (self.n_observations,), dtype ='float32')

      

    def setup_planning_scene(self):
        # add table mesh in scene planning, avoiding to collosion
        rospy.sleep(2)

        p = PoseStamped()
        p.header.frame_id = self.aubo_commander.robot.get_planning_frame()
        p.pose.position.x = 0.225
        p.pose.position.y = 0.
        p.pose.position.z = 0.386

        self.aubo_commander.scene.add_box("table",p,(0.91,0.91,0.77))



    def gripper_camera_callback(self, data):
        #get camera raw
        self.grippper_camera_image_raw = data


    
    def _check_all_systems_ready(self):
        """
        Checks joint_state_publisher and camera topic , publishers and other simulation systems are
        operational.
        """
        self._check_all_sensors_ready()
        return True

    def _check_all_sensors_ready(self):
        self._check_gripper_camera_image_ready()      
        rospy.logdebug("ALL SENSORS READY")


    def _check_gripper_camera_image_ready(self):
        self.grippper_camera_image_raw = None
        while self.grippper_camera_image_raw is None and not rospy.is_shutdown():
            try:
                self.grippper_camera_image_raw = rospy.wait_for_message("/camera/image_raw", Image , timeout=1.0)
                rospy.logdebug("Current /camera/image_raw READY" )

            except:
                rospy.logerr("Current /camera/image_raw not ready yet, retrying for getting image_raw")
    

    def _set_init_pose(self):
        """
        Sets the Robot in its init pose
        The Simulation will be unpaused for this purpose.
        """
        obj_init_pose = Pose()
        obj_init_pose.position.x = np.random.uniform(-self.target_range,self.target_range)
        obj_init_pose.position.y = np.random.uniform(-self.target_range,self.target_range)
        obj_init_pose.position.z = self.initial_object_pos[2]
        obj_init_pose.orientation.x = 0
        obj_init_pose.orientation.y = 0
        obj_init_pose.orientation.z = 0
        obj_init_pose.orientation.w = 0
        rospy.loginfo('Reset object position:\n\tx: {}\n\ty:{}\n\tz:{}'.format(obj_init_pose.position.x ,  obj_init_pose.position.y,  obj_init_pose.position.z ))
        self.gazebo.set_model_state(self.object_name, obj_init_pose)
        self.gazebo.unpauseSim()

        self.last_action = np.array(self.initial_gripper_pos, dtype = 'float32')
        
        assert self.set_ee_movement(self.last_action.tolist()), "Initializing failed"



    def _init_env_variables(self):
        """
        Inits variables needed to be initialised each time we reset at the start
        :return:
        """
        rospy.logdebug("Init Env Variables...")
        rospy.logdebug("Init Env Variables...END")
        

    def set_ee_movement(self,action):
        ee_pose = Pose()
        ee_pose.position.x = action[0]
        ee_pose.position.y = action[1]
        ee_pose.position.z = action[2]
        ee_pose.orientation.y = 1.0
        ee_pose.orientation.z = 0.0
        ee_pose.orientation.w = 0.0
        ee_pose.orientation.x = 0.0

        ee_value = action[3]
        assert self.aubo_commander.move_ee_to_pose(ee_pose), 'arm execuation fails'
        assert self.aubo_commander.execut_ee([ee_value]), 'gripper execuation fails'
        return self.aubo_commander.move_ee_to_pose(ee_pose) and self.aubo_commander.execut_ee([ee_value])


        
    def _set_action(self, action):
        """Applies the given action to the simulation.
        Args:
        	orig_actions(list): 7 or 4 action spaces
        """
        assert len(action) == 4, "Action should be 4 dimensions"




        if isinstance(action,list) : action = np.array(action, dtype = 'float32')
        new_action = gen_sphere_action(self.last_action, action)

        if self.block_gripper:
            new_action[3] = 0.8
        
        self.last_action = new_action

        action = new_action.tolist()

        self.movement_succees = self.set_ee_movement(action)

        return self.movement_succees


    def _get_obs(self):
        """
        It returns the Position of the TCP/EndEffector as observation.
        And the speed of cube
        Orientation for the moment is not considered
        """
        self.gazebo.unpauseSim()

        self.listener.waitForTransform("/world","/robotiq_gripper_center", rospy.Time(), rospy.Duration(4.0))
        try:
            now = rospy.Time.now()
            self.listener.waitForTransform("/world","/robotiq_gripper_center", now, rospy.Duration(4.0))
            (trans, rot) = self.listener.lookupTransform("/world", "/robotiq_gripper_center", now)
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
            raise e
        # using moveit_commander to get end effector pose
        # grip_pose = self.get_ee_pose()
        # ee_array_pose = [grip_pose.position.x, grip_pose.position.y, grip_pose.position.z]

        grip_pos = np.array(trans).reshape(3,)

        if self.has_object:
            # the pose of the cube/box on a table        
            obj_data = self.obj.get_states().copy()

            # postion of object
            object_pos = obj_data[:3]

            # orientation of object
            object_rot = obj_data[3:7]

            # speed of object
            object_velp = obj_data[-3:]

            object_rel_pos = object_pos - grip_pos
        else:
            object_pos = object_rot = object_velp = object_rel_pos = np.zeros(0)

        if not self.has_object:
            achieved_goal = grip_pos.copy()
        else:
            achieved_goal = np.squeeze(object_pos.copy())
        obs = np.concatenate([
            grip_pos, object_pos.ravel(), object_rel_pos.ravel(), object_rot.ravel(),
            object_velp.ravel()
        ])

        return {
            'observation': obs.copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': self.goal.copy(),
        }

    def _sample_goal(self):
        if self.has_object:
            goal = self.initial_object_pos[:3] + self.np_random.uniform(-self.target_range, self.target_range, size=3)
            goal[2] = self.height_offset
            if self.target_in_the_air and self.np_random.uniform() < 0.5:
                goal[2] += self.np_random.uniform(0, 0.35)
        else:
            goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(-self.target_range, self.target_range, size=3)
        return goal.copy()

    def _compute_reward(self, achieved_goal, goal, info):
        # Compute distance between goal and the achieved goal.
        d = goal_distance(achieved_goal, goal)
        if self.reward_type == 'sparse':
            return -1 if d > self.distance_threshold else 0
        else:
            return -d


    def _is_success(self, achieved_goal, desired_goal):
        d = goal_distance(achieved_goal, desired_goal)
        return d < self.distance_threshold
