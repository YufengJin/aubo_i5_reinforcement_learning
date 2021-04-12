import gym
import numpy
import time
from gym import wrappers
# ROS packages required
import rospy
import rospkg
# import our training environment
import reach_sim
import push_cube_sim


if __name__ == '__main__':

    rospy.init_node('aubo_learn', anonymous=True, log_level=rospy.WARN)
        # Create the Gym environment
    env = gym.make('ReachSim-v0')
    rospy.loginfo("Gym environment done")

    nepisodes = 1000
    
    for x in range(nepisodes):
    	rospy.logdebug("start episode >> ", str(x))
    	observation = env.reset()

    	done = False
    	for i in range(100):
            #print(env.action_space.high)
            #print(env.action_space.low)
            action = env.action_space.sample()
            print(env.action_space.shape[0])
            observation, reward, done, info = env.step(action)
            if done:
                print("Episode finished after {} timesteps".format(i+1))
                break