#!/usr/bin/env python
# evolutionaryRobotics.py (c) Inaki Rano 2023
#
# This script will train a neural network using
# a genetic algorithm for a robot simulated in Gazebo.

import time
from math import sqrt
import numpy as np
import rospy
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from std_srvs.srv import Empty
from rosgraph_msgs.msg import Clock


# Initialise the ROS node 
rospy.init_node('evolutionary_robotics')

# Create the publisher object to send velocity commands to the
# robot
pub_vel = rospy.Publisher('/robot/cmd_vel', Twist, queue_size=0)

# Multi-Layer Perceptron Class to implement a neural network.
# This class allows:
# - To save and retrieve network data to/from a file
# - To obtain the output given an input
# - To obtain the weights of all the neurons as a vector
# - To set all the weights of the network given a vector
class MLP:
    # Constructor takes an array of units which include input and output.
    # If no units are provided it creates an empty network
    def __init__(self, units = None):
        self.weights = []
        self.activation = []
        if units is not None:
            self.units = units
            self.layers = len(units) - 1
            for i in range(self.layers):
                W = np.random.rand(self.units[i+1], self.units[i] + 1)
                self.weights.append(0.05 * (W - 0.5))
                # Here is where the activation function for each layer
                # is set. If you want to try different activation functions
                # you need to modify this (and possibly the save()/load()
                # methods too)
                if i == self.layers - 1:
                    self.activation.append(lambda x : x)
                else:
                    self.activation.append(np.tanh)

        else:
            self.units = 0
            self.layers = 0

    # Method to compute the output of the network for a given input
    def __call__(self, x):
        y = np.copy(x)
        for l in range(self.layers):
            yHat = np.insert(y, 0, 1.0)
            y = self.activation[l](np.matmul(self.weights[l], yHat))

        return y

    # This method stores the network in a file with a text format
    def save(self, filename):
        fd = open(filename, "w")
        fd.write('MLPv0\n')
        fd.write('units:\n' + str(self.units) + '\n')
        for i in range(self.layers):
            fd.write('W{' + str(i) + '}:\n')
            np.savetxt(fd, self.weights[i])
            fd.write('A{' + str(i) + '}:\n')
            if self.activation[i] == np.tanh:
                fd.write('tanh\n')
            else:
                fd.write('lin\n')
        fd.close()

    # This method reads a file saved with the save() method and sets the
    # values of weights, units... ideally to an empty network
    def load(self, filename):
        fd = open(filename, "r")
        L = fd.readlines()
        if L[0] != 'MLPv0\n':
            print('Error reading ' + filename)
            return None
        if L[1] != 'units:\n':
            print('Malformed file. Missing: units')
            return None

        aux = L[2].replace('[','').replace(',','').split(']')[0].split()
        self.units = np.array(aux, dtype=int)

        self.layers = len(self.units) - 1
        idx = 3
        self.weights = []
        self.activation = []
        for i in range(self.layers):
            if L[idx].strip() == 'W{' + str(i) + '}:\n'.strip():
                w = np.array([])
                for j in range(self.units[i+1]):
                    idx = idx + 1
                    aux = np.array(L[idx].split(), dtype=np.float32)
                    w = np.append([w], aux)
                w.shape = (self.units[i+1], self.units[i]+1)
                self.weights.append(w)
                idx = idx + 1
                if L[idx] != 'A{' + str(i) + '}:\n':
                    print('Malformed file. Missing: A{'+str(i)+'}:')
                    return None
                else:
                    idx = idx + 1
                    if L[idx] == 'tanh\n':
                        self.activation.append(np.tanh)
                    elif L[idx] == 'lin\n':
                        self.activation.append(lambda x: x)
                    else:
                        print('Malformed file. Missing: A{'+str(i)+'} value')
                        return None
                    idx = idx + 1
            else:
                print('Malformed file. Missing: W{'+str(i)+'}:')
                return None
                
    # This method prints the network as text for debugging but it works
    # for python >= 3.7
    # def __str__(self):
    #     s = f'Network\n Inputs: {self.units[0]}\n Outputs {self.units[-1]}\n'
    #     s += f' Layers {self.layers} layers\n'
    #     for i in range(self.layers):
    #         s += f'Layer {i} weights:\n'
    #         s += self.weights[i].__str__()
    #         s += '\n'
    #
    #    return s

    # This method returns the totak number of weights of the network, which is
    # useful when vectorising/de-vectorising the weights
    def number_of_weights(self):
        n = 0
        for i in range(self.layers):
            n += (self.units[i] + 1) * self.units[i + 1]

        return n

    # This method sets the weights of the network based on the vector given
    # as a parameter. The dimension of the vector should match the total
    # number of weights otherwise an error messages is printed on the
    # screen.
    def vector_as_weights(self, w):
        size = 0
        for l in range(self.layers):
            size += (self.units[l] + 1) * self.units[l + 1]
        if len(w) == size:
            idx0 = 0
            for l in range(self.layers):
                idx1 = idx0 + (self.units[l] + 1) * self.units[l + 1]
                self.weights[l] = w[idx0:idx1]
                self.weights[l].shape = (self.units[l+1],self.units[l]+1)
                idx0 = idx1
        else:
            print('Error setting weights')

    # This method returns all the weights of the network as a single
    # vector to be used as individuals of the GA
    def weights_as_vector(self):
        w = np.array([],dtype=np.float32)
        for l in range(self.layers):
            w = np.append(w, np.reshape(self.weights[l], (-1,)))

        return w

# ==================================================================
# Definition of global variables required to interface the GA
# with ROS
# ==================================================================

# ------------------------------------------------------------------
# Definition of the NLP for the GA. Because of how ROS works in
# python the network object needs to be global so that ROS callbacks
# can access it.
units = [6, 5, 8, 2]    # TODO: Change the network architecure HERE.
# TODO comprobar que esto non rompe cando metemos os 8 sensores do scan. quitar comentario
nn = MLP(units)


# These global variables are necessary for the interaction
# between the optimisation process of the GA with ROS

# ------------------------------------------------------------------
# scan_front_new and scan_back_new hold boolena values for the
# case when the network to learn uses front and back values. The
# network is only used when we have fresh values for the front and
# back readings. Since the callback is the same for both sensors
# the readings will be stored in another global variable called
# scan (see below). This has to be so regardless of whether the
# callback functions are the same or not.
scan_front_new = False
scan_back_new = False

# ------------------------------------------------------------------
# Numpy array to store the distances scanned by the front and rear
# sensors.
scan = np.zeros((8,))

# ------------------------------------------------------------------
# This crash variable stores whether the robot has crashed against a wall
# (actually whether the reading of one of the sensors is smaller than
# crash_distance). This boolean variable needs to be global since it
# is updated in the sensor callback function and used in the clock()
# function to stop the simulation. To run the simulation for the maximum
# amount of time regardless of whether the robot crashes or not the "crash"
# variable should be kept to False all time, but this will affect the
# evaluation time of each individual on the GA.
crash = False
crash_distance = 0.1

# ------------------------------------------------------------------
# fitness variable holds the value of the fitness for one simulation.
# Since the simulation (and the fitness calculation) must be done in
# the callback function for the sensors (ROS dictates this), while the
# fitness value is returned by the fitness() method of the GA (see
# below) this variable is the way of communicating the ROS sensor callback
# function with the GA object
fness = 0.0

# ------------------------------------------------------------------
# Tinit stores the time a new simulation of an individual started. Since the
# individuals of the GA must run for a certain amount of time and then return
# the fitness of the network we need to store the starting time of the
# simulation. This starting time is used to switch states in the execution 
# of the whole program (see variable clock_state). The clock() callback
# is executed periodically and it only receives the simulated time. It could be
# a static variable inside the clock() callback funtion, but how t.f.
# are static variables defined in python? well, let's make it global
Tinit = Clock()

# ------------------------------------------------------------------
# This variable holds the state of the running program and it affects the
# program:
# state = 0: the program is not doing anything (i.e. waiting for the simulation
#            of an individual to start)
# state = 1: The simulation just started and the simulation starting time
#            is stored in Tinit
# state = 2: The simulation is running. Transitions to state=0 is simulation
#            time passed or the robot crashed
clock_state = 0


# ==================================================================
# End of global variable definition 
# 
# ==================================================================


# Callback definition for the sensor readings and simulation time

# CB: Depending on your network architecture you might use the front
# sensors only or the front+back sensors, which means you must ensure
# a full reading is available before calculating the output of the
# neural network. It is *VERY IMPORTANT* to limit the velocity commands
# sent to the motors as very large values *MIGHT MAKE THE SIMULATION
# FAIL*. Finally, the fitness function must be computed here (the only
# way I found of interfacing ROS with the GA)
def laser_cb(L):
    # You might want to use these global variables here
    global scan_front_new, scan_back_new, crash, fness, scan
    # TODO

    scan = L.ranges

    cmd_vel = Twist()
    vel = nn(scan)
    cmd_vel.linear.x = vel[0]
    cmd_vel.linear.y = vel[1]
    # kosas
    pub_vel.publish(cmd_vel)

    scan_max = max(scan)
    r_max = 3 # parameter to chech sensor activation. Might lower later
    if scan_max > r_max:
        scan_max = r_max
    i = scan_max/r_max # activation

    fness += vel *(1-sqrt(abs(vel[0]-vel[1])) * (1-i)) 
    # cambio de = a += para 1.premiar simulacions longas, 2.ter un valor pseudo-medio

    # Might delete later idk
    if min(scan) < crash_distance:
        crash = True

# Clock callback function. This function is executed periodically
# and controls whether the simulation is running to 
def clock_cb(t):
    global Tinit, clock_state, crash
    simulation_time = 90   # Change this to set your simulation time
                         # to compute the fitness
    if clock_state == 0:
        return
    elif clock_state == 1:
        Tinit = t.clock
        clock_state = 2
    elif clock_state == 2:
        if t.clock.secs - Tinit.secs > simulation_time or crash:
            clock_state = 0

# ROS subscribers to the sensor topics to use in the simulation. Keep in mind
# that the callback function is the same for both sensors. That helps
# interfacing ROS with the GA implementation, but care must be taken in the
# implementation because the sensors received as an argument to the CB can
# be the front or rear sensors
# TODO comprobar 
sub_front = rospy.Subscriber('/robot/laser_front/scan', LaserScan, laser_cb)
# sub_back = rospy.Subscriber('/robot/laser_back/scan', LaserScan, laser_cb) 

# ROS subscriber to the clock topic to launch simulations for a specific
# period of time (note: the callback functions are running as separate
# threads independent of what the GA is doing and therefore what these
# functions do must depend on the state of the clock_state variable)
sub_clock = rospy.Subscriber('/clock', Clock, clock_cb)
            
# Example of my parameters for the GA. TODO: Tune it so that the optimisation works
# for the case study. 
GAParams = {'dim' : 3,                  # Dimension of the search/parameter space
            'pop_size' : 2,             # Population size
            'max_iter' : 1,             # Maximum number of iterations
            'mutation_rate' : 0.025,    # Mutation rate in the child population
            'mutation_sigma' : 2,       # Variance of the normal for mutation
            'performace_stop' : 1e-3,   # Stop criterion, minimum performance increase
            'similarity_stop' : 0.8}    # Percentage of population similatiry over sigma score

# Class to implement a genetic algorithm
class GeneticAlgorithm:
    # My constructor for the GA
    def __init__(self, params = None):
        if params is not None:
            self.dim = params['dim']
            self.pop_size = params['pop_size']
            self.iter = 0
            self.max_iter = params['max_iter']
            self.mutation_rate = params['mutation_rate']
            self.mutation_sigma = params['mutation_sigma']
            self.performace_stop = params['performace_stop']
            self.similarity_stop = params['similarity_stop']
            self.population = []
            for i in range(self.pop_size):
                self.population.append(self.random_individual())

            self.score = np.array(self.pop_size * [-np.inf])
            self.score_prev = np.array(self.pop_size * [0])
            
    # Generate random individual to initialise the population. This
    # method should return a random individual for the GA
    def random_individual(self):
        num_weights = nn.number_of_weights()
        return np.random.uniform(-1, 1, size=num_weights)
        
    
    # Method to select and return two parents from the population. This
    # method should return two individuals of the current population
    # to me crossed in the crossover operator. The probability of selecting
    # an individual should depend on its fitness (score).
    def parents(self, ns):

        # Metodo de ruleta
        total = np.sum(self.score)
        parent_results =[]

        for _ in range(ns):
            # r = np.random.uniform(0, total) esto da un erro, np de por que
            r = np.random.random() * total
            cp = 0

            for choice, prob in zip(self.population, self.score):
                cp += prob
                if r<= cp:
                    parent_results.append(choice)
                    break

        return parent_results


        
    # Method to check if the optimisation can stop
    # This method should return True if the termination condition(s) is(are) fulfilled.
    def stop_condition(self): 
        if self.iter >= self.max_iter: 
            return True
        return False

    # Method to cross the two parents
    # This method takes as argument two individuals of the popularion and
    # it must return two new individuals (potentially) for the next population.
    def crossover(self, p1, p2):
        crossover_point = np.random.randint(1, len(p1))
        child1 = np.hstack([p1[:crossover_point], p2[crossover_point:]])
        child2 = np.hstack([p2[:crossover_point], p1[:crossover_point]])

        return child1, child2


    
    # Mutation method for one individual.
    # This method takes as input one individual and returns a (possibly) mutated
    # version of the individual
    def mutation(self, c):
        # TODO

        for i in range(len(c)):
            if np.random.rand() < self.mutation_rate:
                c[i] += np.random.normal(0, self.mutation_sigma)
        return c


    
    # Because of the way the program must intectact with ROS this funcion only
    # affects the ROS callback functions. I don't see a need to modify this method.
    def fitness(self, w):
        # Reset the state of the robot in Gazebo
        rospy.wait_for_service('/gazebo/reset_world')
        reset_world = rospy.ServiceProxy('/gazebo/reset_world', Empty)
        reset_world()

        global clock_state, crash, fness, nn
        # Set the weights of the network to the individual
        # being evaluated currently
        nn.vector_as_weights(w)
        # Make the simulation start
        clock_state = 1
        crash = False
        fness = 0.0
        # Wait until the simulation stops
        while clock_state != 0:
            time.sleep(2)

        # Return the fitness
        return fness

    # Method which implements the main loop of the GA. It must return the best
    # solution after the optimisation process, i.e. a vector representing the
    # fitesst individual
    def optimize(self):

        while not self.stop_condition():
            self.iter += 1

            # selection
            p1, p2 = self.parents(2)
            # crossover
            cross = self.crossover(p1, p2)

            # mutation
            # m1 = self.mutation(c1)
            # m2 = self.mutation(c2)

            for ind in cross:
                mi = self.mutation(ind)
                fitness = self.fitness(mi) # simulate the final child to obtain a fitness value
                self.population.append(mi) # Revisar estas duas linhas, pq non esta habendo remplazo poblacional
                self.score = np.hstack(self.score, fitness) # Revisar estas duas linhas, pq non esta habendo remplazo poblacional

        
        return self.population[np.argmax(self.score)]

    
# Main program
if __name__ == '__main__':
    
    # Set the number of weights as the dimension of the GA
    GAParams['dim'] = nn.number_of_weights()
    # Create a Genetic Algorithm object
    ga = GeneticAlgorithm(GAParams)
    # Run the GA and set the weights of the network to the best individual
    global nn
    nn.vector_as_weights(ga.optimize())
    # Save the neural network to be able to load it later and simulate
    # the behaviour
    nn.save("best-individual.net")
