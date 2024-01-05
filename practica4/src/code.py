#!/usr/bin/env python
from math import sqrt
import numpy as np

import rospy
from geometry_msgs.msg import Twist 
from sensor_msgs.msg import LaserScan

import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

exit()

class NeuralNetwork(tf.keras.Model):
    def __init__(self, num_sensors):
        super(NeuralNetwork, self).__init__()
        self.layer = Dense(8, activation='relu', input_shape=(num_sensors,))
        self.output_layer = Dense(2)

    def forward(self, x):
        x = self.layer(x)
        return self.output_layer(x)
       


class Robot:
    def __init__(self, name, neural_network):
        self.name = name
        self.neural_network = neural_network
        self.laser_data = None
        self.vel_pub = rospy.Subscriber(name + '/cmd_vel', Twist, queue_size=10)
        self.vel_histoty = []
        self.sensor_histoty = []

        rospy.Subscriber(name + '/front_scan', LaserScan, self.laser_callback)

    def laser_callback(self, data):
        # !PREGUNTA por que reemplazas esto?
        self.laser_data = data.ranges #TODO: comprobar esto pero creo que e un array de 8 valores
        
        # save data to compute fitness. check size
        self.sensor_histoty.append(self.laser_data)

    def compute_action(self):
        sensor_input = np.array(self.laser_data)
        action = self.neural_network(sensor_input)
        
        # save data to compute fitness. check size
        self.vel_histoty.append(action)

        return action



def initialize_population(size, num_sensors):
    
    population = []

    for _ in range(size):   #una red por robot de la poblacion
        neural_network = NeuralNetwork(num_sensors)
        neural_network.build((None, num_sensors))

        for layer in neural_network.layers():
            layer.set_weights([np.random.uniform(-1, 1, size = w.shape) for w in layer.get_weights()]) #init de los pesos

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

def calculate_fitness(robot, max_sensor_read):
    #fundion de Lois | fundion jaja 
    
    # TODO CHECK WHOLE FUNCTION
    vels = np.asarray(robot.vel_history)
    mean_vel = np.average(vels, axis=0) # 1x2
    lin_vel = np.sum(mean_vel)/2
    
    sensor_history = np.asarray(robot.sensor_history)
    max_read = np.amax(sensor_history, axis=1)
    mean_max_sensor = np.average(max_read) # Compressible to a single line

    phi = lin_vel * (1 - sqrt(np.abs(mean_vel[0] - mean_vel[1]))) * mean_max_sensor
    
    return phi


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

        #SUSTITUIR P a partir de P(t-1) e P_prima || cambiado pq daba erro de utf-8 (codificacion)
        #TODO: definir cales son os componhentes da nova poboacion || idem que arriba. utf-8

        #AVALIAR P
        fitnesses = [calculate_fitness(robot) for robot in population]



if __name__ == "__main__":
    main()
