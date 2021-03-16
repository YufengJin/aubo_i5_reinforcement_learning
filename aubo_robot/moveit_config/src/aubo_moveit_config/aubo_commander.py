#! /usr/bin/env python

import sys
import copy
import rospy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
import trajectory_msgs.msg
import math
from moveit_commander.conversions import pose_to_list

def euler_from_quaternion(x, y, z, w):
        """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
        """
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = math.atan2(t0, t1)
     
        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = math.asin(t2)
     
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)
     
        return roll_x, pitch_y, yaw_z # in radians



class AuboCommander(object):
    
    def __init__(self):

        moveit_commander.roscpp_initialize(sys.argv)
        
        self.robot = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface()
        print("###### Setting Group Moveit with 30 seconds wait...")
        self.arm_group = moveit_commander.MoveGroupCommander("arm")
        #self.arm_group.set_end_effector_link()
        self.gripper_group = moveit_commander.MoveGroupCommander("gripper")


        
    def check_close(goal,actual,tolerance=0.01):
        """
        Convenience method for testing if a list of values are within a tolerance of their counterparts in another list
        param: goal       A list of floats, a Pose or a PoseStamped
        param: actual     A list of floats, a Pose or a PoseStamped
        param: tolerance  A float
        returns: bool
        """
        done = True
        if type(goal) is list:
            for index in range(len(goal)):
              if abs(actual[index] - goal[index]) > tolerance:
                done = False


        elif type(goal) is geometry_msgs.msg.PoseStamped:
            return self.check_close(goal.pose, actual.pose, tolerance)

        elif type(goal) is geometry_msgs.msg.Pose:
            return self.check_close(pose_to_list(goal), pose_to_list(actual), tolerance)

        return done

    def move_ee_to_pose(self, target_pose):
        """
        ee_pose : a geometry_msgs.msg.Pose() mesaage
        """
        done = None
        self.arm_group.set_pose_target(target_pose)

        if self.execute_trajectory():     
            # check the ending point closed to target point within tolerance   
            current_pose = self.arm_group.get_current_pose()
            done = self.check_close(target_pose, current_pose)
        else:
            # false if not execuate
            done = False

        return done

    def move_joints_traj(self, joint_positions):
        """
        joint_positions : a list of joint state
        """
        done = None
        joint_state_goal = self.arm_group.get_current_joint_values()
        joint_state_goal[0] = joint_positions[0]
        joint_state_goal[1] = joint_positions[1]
        joint_state_goal[2] = joint_positions[2]
        joint_state_goal[3] = joint_positions[3]
        joint_state_goal[4] = joint_positions[4]
        joint_state_goal[5] = joint_positions[5]

        self.arm_group.set_joint_value_target(joint_state_goal)

        if self.execute_trajectory():
            # check if succeed
            current_joint_state = self.arm_group.get_current_joint_values()
            done = self.check_close(joint_state_goal, current_joint_state)
        else:
            done = False
        return done

    def execute_trajectory(self):
        
        #self.plan = self.arm_group.plan()
        flag = self.arm_group.go(wait=True)
        self.arm_group.stop()
        return flag

    def get_ee_pose(self):
        # pose of wrist3_link
        gripper_pose = self.arm_group.get_current_pose()

        return gripper_pose.pose
        
    def get_ee_rpy(self):
        # row, pitch and yaw of wrist3_link
        gripper_rpy = self.arm_group.get_current_rpy()
        roll = gripper_rpy[0]
        pitch = gripper_rpy[1]
        yaw = gripper_rpy[2]
        
        return [roll,pitch,yaw]

    def execut_ee(self, target_value):
        done = self.gripper_group.go(target_value, wait = True)

        if done:     
            # check the ending point closed to target point within tolerance   
            current_value = self.gripper_group.get_current_joint_values()
            done = self.check_close(target_value, current_value)
        else:
            # false if not execuate
            done = False

        return done



# if __name__ == "__main__":
    
#     rospy.init_node('move_group_python_interface_tutorial', anonymous=True)
#     traj_serv_object = AuboCommander()

#     print("Initializing the joint states....")
#     print(traj_serv_object.move_joints_traj([0.0, 0.0, 1.53, 0.0, 1.53, 0.0]))
#     pose_goal = traj_serv_object.get_ee_pose()
#     print(pose_goal)
 
#     pose_goal.position.x += 0.1
#     pose_goal.position.y = 0
#     pose_goal.position.z += 0.2
#     pose_goal.orientation.x = 0
#     pose_goal.orientation.y = 1
#     pose_goal.orientation.z = 0
#     pose_goal.orientation.w = 0



#     print("if succeed? ", traj_serv_object.move_ee_to_pose(pose_goal))
#     print(traj_serv_object.get_ee_pose())
#     traj_serv_object.execut_ee([0.8])
#     print(traj_serv_object.arm_group.get_end_effector_link())
#     print(traj_serv_object.gripper_group.get_end_effector_link())

    
