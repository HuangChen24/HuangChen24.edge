"""
This module contains the class OffloadingCommon, which is the base class of all algorithms (benchmarks, cco and decor).
OffloadingCommon defines several points in a computation offloading problem.

Author:
    Hailiang Zhao, Cheng Zhang
"""
from cross_edge_offloading.parameter import Parameter
from cross_edge_offloading.utils.tool_function import ToolFunction
import numpy as np
import random


class OffloadingCommon:
    """
    This class contains several points in a computation offloading problem, including:
    (1) the objective function of the cross-edge computation offloading problem;
    (2) the solution of the problem (edge_selection, harvested_energy).
    """

    def __init__(self, parameter):
        """
        Initialize key parameters in offloading problems.

        :param parameter: the instance of class Parameter
        """
        # use initial user info and server info to obtain the initial connectable info
        # because of mobility, user info will change in different time slots
        self.__user_num = parameter.get_user_num()
        self.__server_num = parameter.get_server_num()
        self.__user_info = parameter.get_user_info()
        self.__connectable_servers, self.__connectable_users, self.__connectable_distances, self.__global_distances = \
            Parameter.obtain_wireless_signal_coverage(parameter.get_user_info(), parameter.get_edge_info())

        self.__battery_energy_levels = np.zeros(parameter.get_user_num())
        self.__virtual_energy_levels = self.__battery_energy_levels - \
                                       np.repeat(parameter.get_perturbation_para(), self.get_user_num())

        # edge_selections is a list with every element (edge_selection) being a numpy array,
        # which is the feasible solution (independent var) of the problem $\mathcal{P}_2^{es}$
        self.edge_selections = self.obtain_edge_selections(parameter)

        # assignable numbers of edge sites
        self.assignable_nums = np.repeat(parameter.get_max_assign(), self.__user_num)

        # task requests of mobile devices
        self.__task_requests = self.generate_task_request(parameter)

        # mobility management
        self.__latitude_drv, self.__longitude_drv = self.obtain_derivation()

        # add parameter as its property
        self.__parameter = parameter

    def obtain_overall_costs(self, parameter):
        """
        Calculate the overall costs, which is the sum of cost of each mobile device.

        :param parameter: the instance of class Parameter
        :return: overall costs
        """
        overall_costs = 0
        for i in range(self.__user_num):
            transmit_times = ToolFunction.obtain_transmit_times(
                self.edge_selections[i], parameter, self.__connectable_distances[i])
            edge_exe_times = ToolFunction.obtain_edge_exe_times(self.edge_selections[i], parameter)
            edge_times = transmit_times + edge_exe_times
            division = sum(self.edge_selections[i])
            is_dropped = False if division else True
            overall_cost = max(edge_times) + parameter.get_local_exe_time() + parameter.get_coordinate_cost() * \
                           sum(self.edge_selections[i]) + is_dropped * parameter.get_drop_penalty()
            overall_costs += overall_cost
        return overall_costs

    def obtain_edge_selections(self, parameter):
        """
        Initialize the feasible solution with random policy.

        :param parameter: the instance of class Parameter
        :return: initial edge_selections, every row denotes a mobile device who has task request
        (size: $\sum_{i \in \mathcal{N}'} \mathcal{M}_i$, where $N' \leq N$.
        """
        # first initialize with zero
        edge_selections = []
        for i in range(self.__user_num):
            edge_selection = np.repeat(0, len(self.__connectable_servers[i]))
            edge_selections.append(edge_selection)

        # for every edge site, generate a random integer with [0, max_assign], and distribute connections to
        # connectable mobile devices
        for j in range(self.__server_num):
            assign_num = random.randint(0, parameter.get_max_assign())
            connectable_user_num = len(self.__connectable_users[j])
            if assign_num >= connectable_user_num:
                # every mobile device in it can be chosen
                for i in range(connectable_user_num):
                    user_index = self.__connectable_users[j][i]
                    edge_selections[user_index][j] = 1
            else:
                # randomly choose assign_num users to distribute j's computation capacity
                user_indices = random.sample(self.__connectable_users[j], assign_num)
                for i in range(len(user_indices)):
                    user_index = user_indices[i]
                    edge_selections[user_index][j] = 1

        # set those mobile devices who do not have task request to [0, 0, ..., 0]
        # we can not delete them from the list because every row is the index of the corresponding mobile device
        for i in range(self.__user_num):
            if self.__task_requests[i] == 0:
                edge_selections[i] = np.array(0, len(edge_selections[i]))
        return edge_selections

    def obtain_harvested_energy(self, parameter):
        """
        Randomly choose energy between $[0, E_i^H]$.

        :param parameter: the instance of class Parameter
        :return: actually harvested energy, $\alpha_i$
        """
        return random.uniform(0, ToolFunction.generate_harvestable_energy(parameter))

    def update_energy_levels(self, parameter):
        """
        Update the battery & virtual energy level according to the involution expression \eqref{10}.

        :param parameter: the instance of class Parameter
        :return: no return
        """
        for i in range(self.__user_num):
            self.__battery_energy_levels[i] = self.__battery_energy_levels[i] - ToolFunction.obtain_transmit_energys(
                self.edge_selections[i], parameter, self.__connectable_distances[i]) - \
                                              parameter.get_local_exe_energy() + self.obtain_harvested_energy(parameter)
            self.__virtual_energy_levels[i] = self.__battery_energy_levels[i] - parameter.get_perturbation_para()

    def update_assignable_nums(self):
        """
        Update assignable number of connectable mobile devices for each edge site.

        :return: no return
        """
        for i in range(self.__user_num):
            for j in range(len(self.edge_selections[i])):

                server_index = self.get_connectable_servers()[i][j]
                if self.edge_selections[i][j] == 1:
                    self.assignable_nums[server_index] -= 1

    def generate_task_request(self, parameter):
        """
        Generate task request from Bernoulli Distribution.

        :param parameter: the instance of class Parameter
        :return: a numpy array denoted task request, presented in [0, 1, ..., 1]
        """
        return ToolFunction.sample_from_bernoulli(self.get_user_num(), parameter)

    def update_users_pos(self):
        """
        Generate users position with random policy, and then update user_info.
        !! How to confine users within this specific area? !!

        :return: no return
        """
        positions = self.__user_info.T
        positions[0] += np.random.uniform(-self.__latitude_drv, self.__latitude_drv, size=self.__user_num)
        positions[1] += np.random.uniform(-self.__longitude_drv, self.__longitude_drv, size=self.__user_num)
        self.__user_info = positions.T

    def obtain_derivation(self):
        positions = self.__user_info.T
        latitude_drv = np.mean(positions[0] - np.mean(positions[0]))
        longitude_drv = np.mean(positions[1] - np.mean(positions[1]))
        return latitude_drv, longitude_drv

    def get_user_num(self):
        return self.__user_num

    def get_server_num(self):
        return self.__server_num

    def get_user_info(self):
        return self.__user_info

    def get_connectable_servers(self):
        return self.__connectable_servers

    def get_connectable_users(self):
        return self.__connectable_users

    def get_connectable_distances(self):
        return self.__connectable_distances

    def get_global_distances(self):
        return self.__global_distances

    def get_battery_energy_levels(self):
        return self.__battery_energy_levels

    def get_virtual_energy_levels(self):
        return self.__virtual_energy_levels

    def get_task_requests(self):
        return self.__task_requests

    def get_latitude_drv(self):
        return self.__latitude_drv

    def get_longitude_drv(self):
        return self.__longitude_drv

    def get_parameter(self):
        return self.__parameter
