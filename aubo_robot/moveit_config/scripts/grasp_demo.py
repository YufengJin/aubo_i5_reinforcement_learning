#! /usr/bin/env python
import sys
import rospy
import moveit_commander
import geometry_msgs

moveit_commander.roscpp_initializer.roscpp_initialize(sys.argv)
rospy.init_node('move_group_grasp', anonymous=True)
robot = moveit_commander.robot.RobotCommander()

arm_group = moveit_commander.move_group.MoveGroupCommander("arm")
hand_group = moveit_commander.move_group.MoveGroupCommander("gripper")
arm_group.set_named_target("standby")
plan = arm_group.go()
hand_group.set_named_target("open")
plan = hand_group.go()

pose_target = geometry_msgs.msg.Pose()
pose_target.orientation.w = 0.0
pose_target.orientation.x = 0.0
pose_target.orientation.y = 0
pose_target.orientation.z = 0
pose_target.position.x = 0.3
pose_target.position.y = 0
pose_target.position.z = 1.16
arm_group.set_pose_target(pose_target)
plan = arm_group.go()

pose_target.position.z = 1.08
arm_group.set_pose_target(pose_target)
plan = arm_group.go()

hand_group.set_named_target("close")
plan = hand_group.go()

pose_target.position.z = 1.5
arm_group.set_pose_target(pose_target)
plan = arm_group.go()

hand_group.set_named_target("open")
plan = hand_group.go()

rospy.sleep(5)
moveit_commander.roscpp_initializer.roscpp_shutdown()