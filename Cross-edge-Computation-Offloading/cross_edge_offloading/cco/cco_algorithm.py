"""
This module defines the class CcoAlgorithm, which is the main algorithm proposed in this paper.
It inherits from class OffloadingCommon.

Author:
    Hailiang Zhao
"""
from cross_edge_offloading.offloading_common import OffloadingCommon
from cross_edge_offloading.utils.tool_function import ToolFunction
import numpy as np
import random
from zoopt.zoopt.dimension import Dimension
from zoopt.zoopt.objective import Objective
from zoopt.zoopt.solution import Solution
from zoopt.zoopt.parameter import Parameter as ZooptParameter
from zoopt.zoopt.opt import Opt


class CcoAlgorithm(OffloadingCommon):
    """
    This class implements the algorithm named CCO proposed in the paper.
    """

    def __init__(self, parameter):
        """
        Super from base class OffloadingCommon, then update the way CCO initializes feasible solution.

        :param parameter: the instance of class Parameter
        """
        super().__init__(parameter)
        # update the way CCO initializes feasible solution
        self.edge_selections = self.obtain_edge_selections(parameter)

    def obtain_harvested_energy(self, parameter):
        """
        Obtain the optimal harvested energy by solving the 'optimal energy harvesting' sub-problem
        according to \eqref{18}.

        :param parameter: the instance of class Parameter
        :return: the optimal harvested energy
        """
        if self.get_virtual_energy_levels() <= 0:
            return ToolFunction.generate_harvestable_energy(parameter)
        else:
            return 0

    def obtain_edge_selections(self, parameter):
        """
        Override base method with greedy policy.

        :param parameter: the instance of class Parameter
        :return: initial edge_selections, which has the same shape with connectable_servers
        """
        assignable_nums = np.repeat(parameter.get_max_assign(), len(self.get_connectable_servers()))
        return self.greedy_sample(assignable_nums)

    def greedy_sample(self, assignable_nums):
        """
        According to the assignable numbers of edge sites, sample solution with greedy policy.

        :param assignable_nums: the assignable numbers of edge sites, which is <= max_assign
        :return: a feasible solution edge_selections, which has the same shape with connectable_servers
        """
        # first initialize with zero
        edge_selections = []
        for i in range(len(self.__connectable_servers)):
            edge_selection = np.repeat(0, len(self.__connectable_servers[i]))
            edge_selections.append(edge_selection)

        # for every edge site, directly distribute max_assign connections to connectable mobile devices
        for j in range(len(self.__connectable_users)):
            assign_num = assignable_nums[j]
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

    def sub_problem_es(self, parameter):
        """
        Calculate the optimization goal of sub-problem $\mathcal{P}_2^{es}$.

        :param parameter: the instance of class Parameter
        :return: the optimization goal of $\mathcal{P}_2^{es}$
        """
        distances = self.get_connectable_distances()
        optimization_goals = 0
        for i in range(len(self.edge_selections)):
            edge_selection = self.edge_selections[i]
            distance = distances[i]
            optimization_goal = parameter.get_v() * self.obtain_overall_costs(parameter) - \
                                parameter.get_local_exe_energy() - ToolFunction.obtain_transmit_energys(
                                edge_selection, parameter, distance)
            optimization_goals += optimization_goal
        return optimization_goals

    def construct_solution_space(self):
        """
        Construct feasible solution space for SAC (racos) algorithm (edge_selections -> solution).

        :return: an instance of class Dimension
        """
        task_requests = self.get_task_requests()
        dim_size = 0
        for i in range(len(self.edge_selections)):
            if task_requests[i] == 1:
                dim_size += len(self.edge_selections[i])
        dim_regs = [[0, 1]] * dim_size
        dim_tys = [False] * dim_size
        return Dimension(size=dim_size, regs=dim_regs, tys=dim_tys)

    def solution_to_edge_selections(self, solution):
        """
        Convert an instance of class Solution to edge_selections, and then update self.edge_selections.

        :param solution: the instance of class Solution
        :return: no return
        """
        chosen_sites = solution.get_x()
        edge_selections = []
        task_requests = self.get_task_requests()
        for i in range(self.get_user_num()):
            if task_requests[i] == 1:
                edge_selection = np.array(chosen_sites[0: len(self.get_connectable_servers()[i]) + 1])
                edge_selections.append(edge_selection)
                del chosen_sites[0: len(self.get_connectable_servers()[i]) + 1]
            else:
                edge_selections.append(np.repeat(0, len(self.get_connectable_servers()[i])))
        self.edge_selections = edge_selections

    def obtain_overall_costs(self, parameter):
        """
        Calculate 'V * overall_costs + Lyapunov_drift', which is the sum of it of each mobile device.

        :param parameter: the instance of class Parameter
        :return: V * overall costs + \Delta(\Theta)
        """
        overall_costs = 0
        for i in range(len(self.edge_selections)):
            transmit_times = ToolFunction.obtain_transmit_times(
                self.edge_selections[i], parameter, self.__connectable_distances[i])
            edge_exe_times = ToolFunction.obtain_edge_exe_times(self.edge_selections[i], parameter)
            edge_times = transmit_times + edge_exe_times
            division = sum(self.edge_selections[i])
            is_dropped = False if division else True
            overall_cost = max(edge_times) + parameter.get_local_exe_time() + parameter.get_coordinate_cost() * \
                           sum(self.edge_selections[i]) + is_dropped * parameter.get_drop_penalty()
            neg_lyapunov_drift = (parameter.get_local_exe_energy() + ToolFunction.obtain_transmit_energys(
                self.edge_selections[i], parameter, self.get_connectable_distances()[i])) * \
                             self.get_virtual_energy_levels()[i]
            overall_costs += overall_cost - neg_lyapunov_drift
        return overall_costs

    def object_func(self, solution):
        """
        According to the input solution, calculate the value of objective function.

        :param solution: the instance of class Parameter
        :return: the value of objective function, i.e., V * overall_costs + Lyapunov_drift
        """
        self.solution_to_edge_selections(solution)
        return self.obtain_overall_costs(self.get_parameter())

    def sal_framework(self):
        dim_space = self.construct_solution_space()
        optimization_goal = Objective(self.object_func, dim_space)
        # budget is the number of calls to object function, it decides training_size
        budget = 5 * dim_space.get_size()
        sal_para = ZooptParameter(algorithm='racos', budget=budget, autoset=True, sequential=False)
        optimal_solution = Opt.min(objective=optimization_goal, parameter=sal_para)
        return self.solution_to_edge_selections(optimal_solution)

    # Need to update random sample in Dimension!
