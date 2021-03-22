#! /usr/bin/env python

import numpy
import rospy
import tf
from std_msgs.msg import Float64
from sensor_msgs.msg import JointState, Image
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped, Pose
from aubo_moveit_config.aubo_commander import AuboCommander
from openai_ros import robot_gazebo_env
from obj_positions import Obj_Pos


class AuboEnv(robot_gazebo_env.RobotGazeboEnv):

    def __init__(self, gripper_block, action_type, object_name):
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
        gripper_block(bool) : whether or not gripper execuated
        action_type(string): "joints_control" and "ee_control"
        """

        self.gripper_block = gripper_block
        self.action_type = action_type
        self.obj = Obj_Pos(object_name = object_name)

        rospy.logdebug("Start AuboEnv INIT...")

        JOINT_STATES_SUBSCRIBER = '/joint_states'
        GIPPER_IMAGE_SUBSCRIBER = '/camera/image_raw'
        self.joint_states_sub = rospy.Subscriber(JOINT_STATES_SUBSCRIBER, JointState, self.joints_callback)
        self.joints = JointState()
        self.listener = tf.TransformListener()

        self.grippper_camera_image_raw = rospy.Subscriber(GIPPER_IMAGE_SUBSCRIBER, Image, self.gripper_camera_callback)
        self.grippper_camera_image_raw = Image()

        self.controllers_list = []

        self.aubo_commander = AuboCommander()

        self.setup_planning_scene()
        # It doesnt use namespace
        self.robot_name_space = ""

        # We launch the init function of the Parent Class robot_gazebo_env.RobotGazeboEnv
        super(AuboEnv, self).__init__(controllers_list=self.controllers_list,
	                                    robot_name_space=self.robot_name_space,
	                                    reset_controls=False,
	                                    start_init_physics_parameters=False,
	                                    reset_world_or_sim="WORLD")
        self.get_params()

        self.set_action_observation_space()


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
        self.n_observations = 17

        # joint_state_control(joint_states, ee)
        self.position_joints_max = 2.16
        self.position_joints_min = -2.16

        self.init_pos = [0, 0.6, 0, 0, 1.53, 0, 0.0]
        # ee postion and gripper (x,y,z,ee)
        self.setup_ee_pos = [0.5, 0, 1.1, 0.8]


    def set_action_observation_space(self):

    	if self.action_type == "ee_control":
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
	            high=pos_high, shape=(4,), dtype='float32')

	    elif self.action_type == "joints_control":

	    	joint_low = np.ones(6)*self.position_joints_min
	    	joint_high = np.ones(6)*self.position_joints_max

	    	ee_low = np.array(self.ee_open)
	        ee_high= np.array(self.ee_close)

	        joints_low = np.concatenate([joint_low, ee_low])
	        joints_high = np.concatenate([joint_high, ee_high])
	        
			self.action_space = spaces.Box(
            low= joints_low,
            high= joints_high, shape=(7),
            dtype='float32')


            self.observation_space = spaces.Box(-np.inf, np.inf, shape = (self.n_observations,), dtype ='float32')


    def setup_planning_scene(self):
        # add table mesh in scene planning, avoiding to collosion
        rospy.sleep(2)

        p = PoseStamped()
        p.header.frame_id = self.aubo_commander.robot.get_planning_frame()
        p.pose.position.x = 0.75
        p.pose.position.y = 0.
        p.pose.position.z = 0.386

        self.aubo_commander.scene.add_box("table",p,(0.91,0.91,0.77))


    def joints_callback(self, data):
        # get joint_states
        self.joints = data

    def gripper_camera_callback(self, data):
        #get camera raw
    	self.grippper_camera_image_raw = data

    def get_joints(self):
    	return self.joints


    def get_ee_pose(self):

        gripper_pose = self.aubo_commander.get_ee_pose()
        
        return gripper_pose
        
    def get_ee_rpy(self):
        
        gripper_rpy = self.aubo_commander.get_ee_rpy()
        
        return gripper_rpy

    
    def _check_all_systems_ready(self):
        """
        Checks joint_state_publisher and camera topic , publishers and other simulation systems are
        operational.
        """
        self._check_all_sensors_ready()
        return True

    def _check_all_sensors_ready(self):
        self._check_joint_states_ready()
        self._check_gripper_camera_image_ready()
        
        rospy.logdebug("ALL SENSORS READY")

    def _check_joint_states_ready(self):
        self.joints = None
        while self.joints is None and not rospy.is_shutdown():
            try:
                self.joints = rospy.wait_for_message("/joint_states", JointState, timeout=1.0)
                rospy.logdebug("Current /joint_states READY=>" + str(self.joints))

            except:
                rospy.logerr("Current /joint_states not ready yet, retrying for getting joint_states")
        return self.joints

    def _check_gripper_camera_image_ready(self):
        self.grippper_camera_image_raw = None
        while self.grippper_camera_image_raw is None and not rospy.is_shutdown():
            try:
                self.grippper_camera_image_raw = rospy.wait_for_message("/camera/image_raw", Image , timeout=1.0)
                rospy.logdebug("Current /camera/image_raw READY" )

            except:
                rospy.logerr("Current /camera/image_raw not ready yet, retrying for getting image_raw")
        return self.grippper_camera_image_raw
    

    def _set_init_pose(self):
        """
        Sets the Robot in its init pose
        The Simulation will be unpaused for this purpose.
        """
        self.gazebo.unpauseSim()
        if self.action_type == "ee_control":
			assert self._set_action(self.setup_ee_pos), "Initializing failed"

	    elif self.action_type == "joints_control":
	    	assert self._set_action(self.init_pos), "Initializing failed"


    def _init_env_variables(self):
        """
        Inits variables needed to be initialised each time we reset at the start
        :return:
        """
        rospy.logdebug("Init Env Variables...")
        rospy.logdebug("Init Env Variables...END")
        


    def _set_action(self, action):
        """Applies the given action to the simulation.
        Args:
        	actions(list): 7 or 4 action spaces
        """

        # action should be list, but i want to clip the list
        # action = np.clip(action.copy(),self.action_space.low, self.action_space.high)

        # joint_state control
        if self.action_type == "joints_control":
        	assert len(action) == 7, "Action spaces should be 7 dimensions"
        	joint_states, ee_value = action[:6], action[7]

        	# gripper always close
        	if self.gripper_block == True:
	        	print("Gripper Block")
        		ee_value = 0.8

        	self.movement_succees = self.aubo_commander.move_joints_traj(joint_states) and self.aubo_commander.execut_ee([ee_value])
        
        # end effector control , and ee always with fixed rpy
        elif self.action_type == "ee_control":
        	assert len(action) == 4, "Action should be 4 dimensions"

        	ee_pose = Pose()
	        ee_pose.position.x = action[0]
	        ee_pose.position.z = action[1]
	        ee_pose.position.y = action[2]
	        ee_pose.orientation.y = 1.0
	        ee_pose.orientation.z = 0.0
	        ee_pose.orientation.w = 0.0
	        ee_pose.orientation.x = 0.0

	        ee_value = action[3]

	        if self.gripper_block:
	        	print("Gripper Block")
	            ee_value = 0.8

	        self.movement_succees = self.aubo_commander.move_ee_to_pose(ee_pose) and self.aubo_commander.execut_ee([ee_value])



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

        ee_trans = np.array(trans).reshape(3,)
        # rotation in quaterion
        ee_rot = np.array(rot).reshape(4,)


        # the pose of the cube/box on a table        
        obj_data = self.obj.get_states().copy()

        # postion of object
        obj_trans = object_data[3:]

        # orientation of object
        obj_rot = object_data[4:7]
        # speed of object
        obj_vel = object_data[-3:]

        # observation spaces 17
        obs = np.concatenate([ee_trans, ee_rot, obj_trans, obj_rot, obj_vel])
        
        return  obs


    def _compute_reward(self, observations, done):
        """Calculates the reward to give based on the observations given.
        """
        raise NotImplementedError()

    def _is_done(self, observations):
        """Calculates the reward to give based on the observations given.
        """
        raise NotImplementedError()
