#!/usr/bin/env python

import numpy as np
import rospy
from gazebo_msgs.srv import GetWorldProperties, GetModelState
import sys

class Obj_Pos(object):
    """
    This object maintains the pose and rotation of the cube in a simulation through Gazebo Service

    """

    def __init__(self, object_name):
        self._object_name = object_name
        world_specs = rospy.ServiceProxy(
            '/gazebo/get_world_properties', GetWorldProperties)()
        self.time = 0
        self.model_names = world_specs.model_names
        self.get_model_state = rospy.ServiceProxy(
            '/gazebo/get_model_state', GetModelState)

    def get_states(self):
        """
        Returns the ndarray of pose&rotation of the cube
        """
        for model_name in self.model_names:
            if model_name == self._object_name:
                data = self.get_model_state(
                    model_name, "world")  # gazebo service client
                return np.array([
                    data.pose.position.x,
                    data.pose.position.y,
                    data.pose.position.z,
                    data.twist.linear.x,
                    data.twist.linear.y,
                    data.twist.linear.z
                ])


if __name__ == "__main__":
    obj_positions = Obj_Pos(object_name="block")
    st = obj_positions.get_states()
    r = rospy.Rate(5.0)
    while not rospy.is_shutdown():
        st = obj_positions.get_states()
        print(st)
        r.sleep()
