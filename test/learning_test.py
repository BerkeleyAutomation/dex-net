"""
Tests learning module basic functionality
Author: Jeff Mahler
"""
from abc import ABCMeta, abstractmethod

import copy
import IPython
import logging
import numpy as np
import os
import sys
import time
from unittest import TestCase, TestSuite, TextTestRunner

from autolab_core import RigidTransform, YamlConfig, BernoulliRV, GaussianRV
from dexnet.learning import RandomBinaryObjective, RandomContinuousObjective, UniformAllocationMean, ThompsonSampling, GaussianUniformAllocationMean, MaxIterTerminationCondition

from constants import *

CONFIG = YamlConfig(TEST_CONFIG_NAME)

class LearningTest(TestCase):
    def test_uniform_alloc(self, num_candidates=NUM_CANDIDATES):
        # get candidates
        np.random.seed(1000)
        pred_means = np.random.rand(num_candidates)
        candidates = []
        for i in range(num_candidates):
            candidates.append(BernoulliRV(pred_means[i]))

        # get true maximum
        true_max = np.max(pred_means)
        true_max_indices = np.where(pred_means == true_max)
        
        # solve using uniform allocation
        obj = RandomBinaryObjective()
        ua = UniformAllocationMean(obj, candidates)

        result = ua.solve(termination_condition = MaxIterTerminationCondition(MAX_ITERS), snapshot_rate = SNAPSHOT_RATE)
        
        # check result (not guaranteed to work in finite iterations but whatever)
        self.assertTrue(len(result.best_candidates) == 1)
        self.assertTrue(np.abs(result.best_candidates[0].p - true_max) < 1e-4)
        self.assertTrue(result.best_pred_ind[-1] == true_max_indices[0])

    def test_thompson_sampling(self, num_candidates=NUM_CANDIDATES):
        # get candidates
        np.random.seed(1000)
        pred_means = np.random.rand(num_candidates)

        candidates = []
        for i in range(num_candidates):
            candidates.append(BernoulliRV(pred_means[i]))

        # get true maximum
        true_max = np.max(pred_means)
        true_max_indices = np.where(pred_means == true_max)
        
        # solve using uniform allocation
        obj = RandomBinaryObjective()
        ts = ThompsonSampling(obj, candidates)

        result = ts.solve(termination_condition = MaxIterTerminationCondition(MAX_ITERS), snapshot_rate = SNAPSHOT_RATE)

        # check result (not guaranteed to work in finite iterations but whatever)
        self.assertTrue(len(result.best_candidates) == 1)
        self.assertTrue(np.abs(result.best_candidates[0].p - true_max) < 1e-4)
        self.assertTrue(result.best_pred_ind[-1] == true_max_indices[0])

    def test_gaussian_uniform_alloc(self, num_candidates=25):
        # get candidates
        np.random.seed(1000)
        pred_means = np.random.rand(num_candidates)
        pred_covs = np.random.rand(num_candidates)

        candidates = []
        for i in range(num_candidates):
            candidates.append(GaussianRV(pred_means[i], pred_covs[i]))

        # get true maximum
        true_max = np.max(pred_means)
        true_max_indices = np.where(pred_means == true_max)
        
        # solve using uniform allocation
        obj = RandomContinuousObjective()
        ua = GaussianUniformAllocationMean(obj, candidates)

        result = ua.solve(termination_condition = MaxIterTerminationCondition(MAX_ITERS), snapshot_rate = SNAPSHOT_RATE)

        # check result (not guaranteed to work in finite iterations but whatever)
        self.assertTrue(len(result.best_candidates) == 1)
        self.assertTrue(np.abs(result.best_candidates[0].mu - true_max) < 1e-4)
        self.assertTrue(result.best_pred_ind[-1] == true_max_indices[0])        

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.DEBUG)
    test_suite = TestSuite()
    test_suite.addTest(LearningTest('test_uniform_alloc'))
    test_suite.addTest(LearningTest('test_thompson_sampling'))    
    test_suite.addTest(LearningTest('test_gaussian_uniform_alloc'))    
    TextTestRunner(verbosity=2).run(test_suite)
        
