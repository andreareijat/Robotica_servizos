#!/usr/bin/env python

import rospy
from geometry_msgs.msg import Twist 
from sensor_msgs.msg import LaserScan


def main():

    rospy.init_node("neural_robot_controller", anonymous=True)
    


if __name__ == "__main__":
    main()
