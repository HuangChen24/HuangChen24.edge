"""
This module contains the tool functions about Communication and Computing details in edge computing systems.

Author:
    Hailiang Zhao
"""
import math
import numpy as np
from geopy.distance import geodesic
import random


class ToolFunction:
    """
    This class defines the tool functions about Communication and Computing.
    """

    def __init__(self):
        pass

    @staticmethod
    def obtain_achieve_rate(bandwidth, channel_power_gain, transmit_power, noise):
        """
        Calculate the achievable rate under Shannon-Hartley Theorem, with both inter-cell & intra-cell
        interference not considered.

        :param bandwidth: transmission bandwidth
        :param channel_power_gain: channel power gain from sender to receiver
        :param transmit_power: transmitting power from sender to receiver
        :param noise: background noise at the receiver
        :return: the achievable rate of the transmission
        """
        return bandwidth * math.log2(1 + channel_power_gain * transmit_power / noise)

    @staticmethod
    def obtain_channel_power_gain(path_loss_const, distance):
        """
        Calculate the channel power gain from source to destination. Currently we assume that it is exponentially
        distributed with mean $g_0 d^{-4}$, where $g_0$ is the path-loss constant, $d$ is the distance from sender
        to receiver.

        :param path_loss_const: path-loss constant
        :param distance: distance from sender to receiver
        :return: the channel power gain under exponential distribution
        """
        mu = path_loss_const * (distance ** -4)
        return np.random.exponential(scale=mu)

    @staticmethod
    def log(algo, content):
        """
        Output logs for computation offloading problem.

        :param algo: the algorithm chosen
        :param content: the text content
        :return: no return value
        """
        print('==>' + algo + '<==:', content)

    @staticmethod
    def obtain_geo_distance(user_pos, server_pos):
        """
        Calculate the geography distance between a particular mobile device and a particular edge server.

        :param user_pos: user's position (latitude, longitude), described in numpy array
        :param server_pos: edge server's position (latitude, longitude), described in numpy array
        :return: geography distance between user and server
        """
        return geodesic(tuple(user_pos), tuple(server_pos)).m

    @staticmethod
    def generate_harvestable_energy(parameter):
        """
        Generate harvestable energy $E_i^H$.

        :param parameter:
        :return:
        """
        return random.uniform(0, parameter.get_max_harvest_energy())

    @staticmethod
    def obtain_transmit_times(edge_selection, parameter, distances):
        """
        Calculate the transmission time of one mobile device to its every chosen 'connectable' edge sites,
        described in numpy array.

        :param edge_selection: the edge selection decision, such as [0,1,0,...,1] (numpy array)
        :param parameter: the instance of Parameter
        :param distances: the distance from a mobile device to every connectable edge sites, described in numpy array
        :return: the transmit times from a user to chosen edge sites (numpy array)
        """
        division = sum(edge_selection)
        offload_data_size = parameter.get_edge_input_size() / division
        channel_power_gains = list(map(ToolFunction.obtain_channel_power_gain,
                                       [parameter.get_path_loss_const()] * division, distances))
        achieve_rates = list(map(ToolFunction.obtain_achieve_rate, [parameter.get_bandwidth()] * division,
                                 channel_power_gains, [parameter.get_transmit_power()] * division,
                                 [parameter.get_noise()] * division))
        transmit_times = np.repeat(offload_data_size, division) / achieve_rates
        return transmit_times

    @staticmethod
    def obtain_transmit_energys(edge_selection, parameter, distances):
        """
        Calculate the transmission energy consumption of one mobile device

        :param edge_selection: the edge selection decision, such as [0,1,0,...,1] (numpy array)
        :param parameter: the instance of Parameter
        :param distances: the distance from a mobile device to every connectable edge sites, described in numpy array
        :return: the transmit energy consumption
        """
        return sum(ToolFunction.obtain_transmit_times(edge_selection, parameter, distances) *
                   parameter.get_transmit_power())

    @staticmethod
    def obtain_edge_exe_times(edge_selection, parameter):
        """
        Calculate the edge execution time of a mobile device on every chosen edge sites.

        :param edge_selection: the edge selection decision, such as [0,1,0,...,1] (numpy array)
        :param parameter: the instance of Parameter
        :return: the edge execution times (numpy array)
        """
        division = sum(edge_selection)
        # CPU-cycle required in every chosen edge sites
        cpu_cycle_required = parameter.get_unit_cpu_num() * parameter.get_edge_input_size() / division
        # notice that we set CPU-cycle frequency of every edge sites the same,
        # thus the edge execution times are the same
        edge_exe_times = np.repeat(cpu_cycle_required / parameter.get_edge_cpu_freq(), division)
        return edge_exe_times

    @staticmethod
    def sample_from_bernoulli(trials, parameter):
        """
        Sample from Bernoulli Distribution with probability $\rho$.

        :param trials: number of trials
        :param parameter: the instance of class Parameter
        :return: sampling results in numpy array ([0, 1, 1, ..., 0, 1])
        """
        samples = np.repeat(0, trials)
        for i in range(trials):
            samples[i] = 1 if random.random() <= parameter.get_task_request_prob() else 0
        return samples
