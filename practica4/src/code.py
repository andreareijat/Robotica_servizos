#!/usr/bin/env python

import rospy
from geometry_msgs.msg import Twist 
from sensor_msgs.msg import LaserScan
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


def initialize_population(size, num_sensors):
    
    population = []

    for _ in range(size):   #una red por robot de la poblacion
        neural_network = NeuralNetwork(num_sensors)

        for param in neural_network.parameters():
            nn.init.uniform_(param, -1, 1) #init de los pesos

        population.append(neural_network)
    
    return population

def selection(population, fitnesses):
    #implementar ruleta
    pass

def crossover(parents):
    #implementar aleatorio
    pass

def mutate(children):
    #TODO: escoller metodo
    pass

def calculate_fitness():
    #fundion de Lois
    pass


def main():

    rospy.init_node("neural_robot_controller", anonymous=True)

    num_generations = 20
    size = 3
    t = 0
    n_parents = 2

    population = initialize_population(size)
    fitnesses = [calculate_fitness(robot) for robot in population]

    while t < num_generations:
        t+=1    

        #SELECCION P' desde P(t-1)
        parents = selection(population, fitnesses, n_parents) 

        #CRUCE P'
        children = crossover(parents)

        #MUTACION P'
        mutate_children = mutate(children)

        #SUSTITUIR P a partir de P(t-1) e P'
        #TODO: definir cales son os compoñentes da nova poboacion

        #AVALIAR P
        fitnesses = [calculate_fitness(robot) for robot in population]



if __name__ == "__main__":
    main()
