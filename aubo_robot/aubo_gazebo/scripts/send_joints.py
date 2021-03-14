#!/usr/bin/python

from trajectory_msgs.msg import JointTrajectory
from std_msgs.msg import Header
from trajectory_msgs.msg import JointTrajectoryPoint
import rospy
import actionlib
from actionlib_msgs.msg import GoalID
from control_msgs.msg import GripperCommandAction, GripperCommandGoal, FollowJointTrajectoryAction, FollowJointTrajectoryGoal

class AuboCommand:
    def __init__(self):
        rospy.init_node('send_control_command')
        self.joint_names = ['shoulder_joint', 'upperArm_joint', 'foreArm_joint', 'wrist1_joint', 'wrist2_joint', 'wrist3_joint' ]
        # create a publish for arm_controller
        #self.pub_arm = rospy.Publisher('/aubo_i5/arm_controller/command', JointTrajectory, queue_size=10)
        # create a action client
        self.gripper_client = actionlib.SimpleActionClient('/aubo_i5/gripper_controller/gripper_cmd', GripperCommandAction)
        self.arm_client = actionlib.SimpleActionClient('/aubo_i5/arm_controller/follow_joint_trajectory', FollowJointTrajectoryAction )
        # add topic check



    def set_arm(self, joint_position,duration=2.0):
        # traj = JointTrajectory()
        # traj.header = Header()
        # # Joint names for Aubo
        # traj.joint_names = self.joint_names
        # while not rospy.is_shutdown():
        #     traj.header.stamp = rospy.Time.now()
        #     pts = JointTrajectoryPoint()
        #     pts.positions = joint_position
        #     pts.time_from_start = rospy.Duration(1.0)
        #     traj.points = []
        #     traj.points.append(pts)
        #     self.pub_arm.publish(traj)
                # check the server working correctly
        self.arm_client.wait_for_server()

        goal = FollowJointTrajectoryGoal()
        # Joint names for Aubo
        goal.trajectory.joint_names = self.joint_names

        goal.trajectory.header.stamp = rospy.Time.now()
        pts = JointTrajectoryPoint()
        pts.positions = joint_position
        pts.time_from_start = rospy.Duration(duration)
        goal.trajectory.points = []
        goal.trajectory.points.append(pts)
        self.arm_client.send_goal(goal)
        self.arm_client.wait_for_result()      
        return self.arm_client.get_result()


    def set_ee(self,value):
        # check the server working correctly
        self.gripper_client.wait_for_server()

        goal = GripperCommandGoal()
        goal.command.position = value   # From 0.0 to 0.8
        goal.command.max_effort = -1.0  # Do not limit the effort
        self.gripper_client.send_goal(goal)
        
        self.gripper_client.wait_for_result()
        
        return self.gripper_client.get_result()

def main():

    aubo_i5 = AuboCommand()
    aubo_i5.set_ee(0.0)
    aubo_i5.set_arm([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    aubo_i5.set_arm([0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    aubo_i5.set_arm([0.0, -1.0, 0.0, 0.0, 0.0, 0.0])
    aubo_i5.set_arm([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    aubo_i5.set_ee(0.8)


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        print ("Program interrupted before completion")
