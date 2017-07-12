"""
Abstract classes for solvers

Author: Jeff Mahler
"""
from abc import ABCMeta, abstractmethod

import numpy as np

from dexnet.constants import DEF_MAX_ITER
from dexnet.learning import MaxIterTerminationCondition
import IPython

class Solver:
    __metaclass__ = ABCMeta

    def __init__(self, objective):
        self.objective_ = objective

    @abstractmethod
    def solve(self, termination_condition = MaxIterTerminationCondition(DEF_MAX_ITER),
              snapshot_rate = 1):
        '''
        Solves for the maximal / minimal point
        '''
        pass

class TopKSolver(Solver):
    def __init__(self, objective):
        Solver.__init__(self, objective)

    @abstractmethod
    def top_K_solve(self, K, termination_condition = MaxIterTerminationCondition(DEF_MAX_ITER),
                    snapshot_rate = 1):
        '''
        Solves for the top K maximal / minimal points
        '''
        pass

class SamplingSolver(TopKSolver):
    """ Optimization methods based on a sampling strategy"""
    __metaclass__ = ABCMeta

class DiscreteSamplingSolver(SamplingSolver):
    __metaclass__ = ABCMeta
    def __init__(self, objective, candidates):
        """
        Initialize a solver with a discrete set of candidate points
        specified in a list object
        """
        self.candidates_ = candidates # discrete candidates
        self.num_candidates_ = len(candidates)
        TopKSolver.__init__(self, objective)

    @abstractmethod
    def discrete_maximize(self, candidates, termination_condition, snapshot_rate):
        """
        Main loop for sampling-based solvers
        """
        pass

    def partition(self, K):
        """
        Partition the input space into K bins uniformly at random
        """
        candidate_bins = []
        indices = np.linspace(0, self.num_candidates_)
        indices_shuff = np.random.shuffle(indices) 
        candidates_per_bin = np.floor(float(self.num_candidates_) / float(K))

        # loop through bins, adding candidates at random
        start_i = 0
        end_i = min(start_i + candidates_per_bin, self.num_candidates_ - 1)
        for k in range(K-1):
            candidate_bins.push_back(self.candidates_[indices_shuff[start_i:end_i]])

            start_i = start_i + candidates_per_bin
            end_i = min(start_i + candidates_per_bin, self.num_candidates_ - 1)
            
        candidate_bins.push_back(self.candidates_[indices_shuff[start_i:end_i]])
        return candidate_bins

    def solve(self, termination_condition = MaxIterTerminationCondition(DEF_MAX_ITER),
              snapshot_rate = 1):
        """ Call discrete maxmization function with all candidates """
        return self.discrete_maximize(self.candidates_, termination_condition, snapshot_rate)

    def top_K_solve(self, K, termination_condition = MaxIterTerminationCondition(DEF_MAX_ITER),
                    snapshot_rate = 1):
        """ Solves for the top K maximal / minimal points """
        # partition the input space
        if K == 1:
            candidate_bins = [self.candidates_]
        else:
            candidate_bins = self.partition(K)

        # maximize over each bin
        top_K_results = []
        for k in range(K):
            top_K_results.append(self.discrete_maximize(candidate_bins[k], termination_condition, snapshot_rate))
        return top_K_results


class OptimizationSolver(Solver):
    def __init__(self, objective, ineq_constraints = None, eq_constraints = None, eps_i = 1e-2, eps_e = 1e-2):
        """
        Inequality constraints: g_i(x) <= 0
        Equality constraints: h_i(x) <= 0
        """
        self.ineq_constraints_ = ineq_constraints
        self.eq_constraints_ = eq_constraints        
        self.eps_i_ = eps_i
        self.eps_e_ = eps_e
        Solver.__init__(self, objective)

    def is_feasible(self, x):
        """ Check feasibility of a given point """
        try:
            self.objective_.check_valid_input(x)
        except ValueError as e:
            return False

        if self.ineq_constraints_ is not None:
            for g in self.ineq_constraints_:
                if np.sum(g(x) > eps_i * np.ones(g.num_outputs())) > 0:
                    return False

        if self.eq_constraints_ is not None:
            for h in self.eq_constraints_:
                if np.sum(np.abs(h(x)) > eps_e * np.ones(h.num_outputs())) > 0:
                    return False            
        return True
