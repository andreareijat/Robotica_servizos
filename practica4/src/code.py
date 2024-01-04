#!/usr/bin/env python

import rospy
from geometry_msgs.msg import Twist 
from sensor_msgs.msg import LaserScan
import torch
import torch.nn as nn

class NeuralNetwork(nn.Modele):
    def __init__(self, num_sensors):
        super(NeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(num_sensors, 10)
        self.layer2 = nn.Linear(10,10)
        self.output_layer = nn.Linear(10, 2)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = self.output_layer(x)
        return x


class Robot: 

    def __init__(self, name, neural_network):
        
        self.name = name
        self.neural_network = neural_network

    def move(self):
        
        data = None #METER AQUI LASER
        action = self.neural_network(data)

        return action


def initialize_population(size):
    pass

def calculate_fitness(robot):
    pass

def evolve_popu(population):
    #PSEUDOCOGIDO AQUI
    # commit test
    pass

def main():

    rospy.init_node("neural_robot_controller", anonymous=True)

    num_generations = 20

    size = 10
    population = initialize_population(size)

    robots = [None for r in population]

    for robot in robots: 
        action = robot.move()
        fitness = calculate_fitness(robot)

    population = evolve_popu(population)



if __name__ == "__main__":
    main()
