# -*- coding: utf-8 -*-
"""
Copyright ©2017. The Regents of the University of California (Regents). All Rights Reserved.
Permission to use, copy, modify, and distribute this software and its documentation for educational,
research, and not-for-profit purposes, without fee and without a signed licensing agreement, is
hereby granted, provided that the above copyright notice, this paragraph and the following two
paragraphs appear in all copies, modifications, and distributions. Contact The Office of Technology
Licensing, UC Berkeley, 2150 Shattuck Avenue, Suite 510, Berkeley, CA 94720-1620, (510) 643-
7201, otl@berkeley.edu, http://ipira.berkeley.edu/industry-info for commercial licensing opportunities.

IN NO EVENT SHALL REGENTS BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL,
INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF
THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF REGENTS HAS BEEN
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

REGENTS SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED
HEREUNDER IS PROVIDED "AS IS". REGENTS HAS NO OBLIGATION TO PROVIDE
MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
"""
"""
Classes for sampling grasps.
Author: Jeff Mahler
"""
from abc import ABCMeta, abstractmethod
import copy
import IPython
import logging
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
import random
import sys
import time

USE_OPENRAVE = True
try:
    import openravepy as rave
except:
    logging.warning('Failed to import OpenRAVE')
    USE_OPENRAVE = False

import scipy.stats as stats

from dexnet.grasping import Contact3D, ParallelJawPtGrasp3D, PointGraspMetrics3D, GraspableObject3D

class GraspSampler:
    """ Base class for various methods to sample a number of grasps on an object.
    Should not be instantiated directly.

    Attributes
    ----------
    gripper : :obj:`RobotGripper`
        the gripper to compute grasps for
    config : :obj:`YamlConfig`
        configuration for the grasp sampler
    """
    __metaclass__ = ABCMeta

    def __init__(self, gripper, config):
        self.gripper = gripper
        self._configure(config)

    def _configure(self, config):
        """ Configures the grasp generator."""
        self.friction_coef = config['sampling_friction_coef']
        self.num_cone_faces = config['num_cone_faces']
        self.num_samples = config['grasp_samples_per_surface_point']
        self.target_num_grasps = config['target_num_grasps']
        if self.target_num_grasps is None:
            self.target_num_grasps = config['min_num_grasps']

        self.min_contact_dist = config['min_contact_dist']
        self.num_grasp_rots = config['num_grasp_rots']
        if 'max_num_surface_points' in config.keys():
            self.max_num_surface_points_ = config['max_num_surface_points']
        else:
            self.max_num_surface_points_ = 100
        if 'grasp_dist_thresh' in config.keys():
            self.grasp_dist_thresh_ = config['grasp_dist_thresh']
        else:
            self.grasp_dist_thresh_ = 0

    @abstractmethod
    def sample_grasps(self, graspable):
        """
        Create a list of candidate grasps for a given object.
        Must be implemented for all grasp sampler classes.

        Parameters
        ---------
        graspable : :obj:`GraspableObject3D`
            object to sample grasps on
        """
        pass

    def generate_grasps_stable_poses(self, graspable, stable_poses, target_num_grasps=None, grasp_gen_mult=5, max_iter=3,
                        sample_approach_angles=False, vis=False, **kwargs):
        """Samples a set of grasps for an object, aligning the approach angles to the object stable poses.

        Parameters
        ----------
        graspable : :obj:`GraspableObject3D`
            the object to grasp
        stable_poses : :obj:`list` of :obj:`meshpy.StablePose`
            list of stable poses for the object with ids read from the database
        target_num_grasps : int
            number of grasps to return, defualts to self.target_num_grasps
        grasp_gen_mult : int
            number of additional grasps to generate
        max_iter : int
            number of attempts to return an exact number of grasps before giving up
        sample_approach_angles : bool
            whether or not to sample approach angles

        Return
        ------
        :obj:`list` of :obj:`ParallelJawPtGrasp3D`
            list of generated grasps
        """
        # sample dense grasps
        unaligned_grasps = self.generate_grasps(graspable, target_num_grasps=target_num_grasps,
                                                grasp_gen_mult=grasp_gen_mult,
                                                max_iter=max_iter, vis=vis)
        
        # align for each stable pose
        grasps = {}
        for stable_pose in stable_poses:
            grasps[stable_pose.id] = []
            for grasp in unaligned_grasps:
                aligned_grasp = grasp.perpendicular_table(grasp)
                grasps[stable_pose.id].append(copy.deepcopy(aligned_grasp))
        return grasps
        
    def generate_grasps(self, graspable, target_num_grasps=None, grasp_gen_mult=5, max_iter=3,
                        sample_approach_angles=False, vis=False, **kwargs):
        """Samples a set of grasps for an object.

        Parameters
        ----------
        graspable : :obj:`GraspableObject3D`
            the object to grasp
        target_num_grasps : int
            number of grasps to return, defualts to self.target_num_grasps
        grasp_gen_mult : int
            number of additional grasps to generate
        max_iter : int
            number of attempts to return an exact number of grasps before giving up
        sample_approach_angles : bool
            whether or not to sample approach angles

        Return
        ------
        :obj:`list` of :obj:`ParallelJawPtGrasp3D`
            list of generated grasps
        """
        # get num grasps 
        if target_num_grasps is None:
            target_num_grasps = self.target_num_grasps
        num_grasps_remaining = target_num_grasps

        grasps = []
        k = 1
        while num_grasps_remaining > 0 and k <= max_iter:
            # SAMPLING: generate more than we need
            num_grasps_generate = grasp_gen_mult * num_grasps_remaining
            new_grasps = self.sample_grasps(graspable, num_grasps_generate,
                                               vis, **kwargs)

            # COVERAGE REJECTION: prune grasps by distance
            pruned_grasps = []
            for grasp in new_grasps:
                min_dist = np.inf
                for cur_grasp in grasps:
                    dist = ParallelJawPtGrasp3D.distance(cur_grasp, grasp)
                    if dist < min_dist:
                        min_dist = dist
                for cur_grasp in pruned_grasps:
                    dist = ParallelJawPtGrasp3D.distance(cur_grasp, grasp)
                    if dist < min_dist:
                        min_dist = dist
                if min_dist >= self.grasp_dist_thresh_:
                    pruned_grasps.append(grasp)            

            # ANGLE EXPANSION sample grasp rotations around the axis
            candidate_grasps = []
            if sample_approach_angles:
                for grasp in pruned_grasps:
                    # construct a set of rotated grasps
                    for i in range(self.num_grasp_rots):
                        rotated_grasp = copy.copy(grasp)
                        rotated_grasp.set_approach_angle(i * delta_theta)
                        candidate_grasps.append(rotated_grasp)                
            else:
                candidate_grasps = pruned_grasps

            # add to the current grasp set
            grasps += candidate_grasps
            logging.info('%d/%d grasps found after iteration %d.',
                         len(grasps), target_num_grasps, k)

            grasp_gen_mult *= 2
            num_grasps_remaining = target_num_grasps - len(grasps)
            k += 1

        # shuffle computed grasps
        random.shuffle(grasps)
        if len(grasps) > target_num_grasps:
            logging.info('Truncating %d grasps to %d.',
                         len(grasps), target_num_grasps)
            grasps = grasps[:target_num_grasps]
        logging.info('Found %d grasps.', len(grasps))

        return grasps

class UniformGraspSampler(GraspSampler):
    """ Sample grasps by sampling pairs of points on the object surface uniformly at random.    
    """
    def sample_grasps(self, graspable, num_grasps,
                         vis=False, max_num_samples=1000):
        """
        Returns a list of candidate grasps for graspable object using uniform point pairs from the SDF

        Parameters
        ----------
        graspable : :obj:`GraspableObject3D`
            the object to grasp
        num_grasps : int
            the number of grasps to generate

        Returns
        -------
        :obj:`list` of :obj:`ParallelJawPtGrasp3D`
           list of generated grasps
        """
        # get all surface points
        surface_points, _ = graspable.sdf.surface_points(grid_basis=False)
        num_surface = surface_points.shape[0]
        i = 0
        grasps = []

        # get all grasps
        while len(grasps) < num_grasps and i < max_num_samples:
            # get candidate contacts
            indices = np.random.choice(num_surface, size=2, replace=False)
            c0 = surface_points[indices[0], :]
            c1 = surface_points[indices[1], :]

            if np.linalg.norm(c1 - c0) > self.gripper.min_width and np.linalg.norm(c1 - c0) < self.gripper.max_width:
                # compute centers and axes
                grasp_center = ParallelJawPtGrasp3D.center_from_endpoints(c0, c1)
                grasp_axis = ParallelJawPtGrasp3D.axis_from_endpoints(c0, c1)
                g = ParallelJawPtGrasp3D(ParallelJawPtGrasp3D.configuration_from_params(grasp_center,
                                                                                        grasp_axis,
                                                                                        self.gripper.max_width))
                # keep grasps if the fingers close
                success, contacts = g.close_fingers(graspable)
                if success:
                    grasps.append(g)
            i += 1

        return grasps

class GaussianGraspSampler(GraspSampler):
    """ Sample grasps by sampling a center from a gaussian with mean at the object center of mass
    and grasp axis by sampling the spherical angles uniformly at random. 
    """
    def sample_grasps(self, graspable, num_grasps,
                         vis=False,
                         sigma_scale=2.5):
        """
        Returns a list of candidate grasps for graspable object by Gaussian with
        variance specified by principal dimensions.

        Parameters
        ----------
        graspable : :obj:`GraspableObject3D`
            the object to grasp
        num_grasps : int
            the number of grasps to generate
        sigma_scale : float
            the number of sigmas on the tails of the Gaussian for each dimension

        Returns
        -------
        :obj:`list` of obj:`ParallelJawPtGrasp3D`
           list of generated grasps            
        """
        # get object principal axes
        center_of_mass = graspable.mesh.center_of_mass
        principal_dims = graspable.mesh.principal_dims()
        sigma_dims = principal_dims / (2 * sigma_scale)

        # sample centers
        grasp_centers = stats.multivariate_normal.rvs(
            mean=center_of_mass, cov=sigma_dims**2, size=num_grasps)

        # samples angles uniformly from sphere
        u = stats.uniform.rvs(size=num_grasps)
        v = stats.uniform.rvs(size=num_grasps)
        thetas = 2 * np.pi * u
        phis = np.arccos(2 * v - 1.0)
        grasp_dirs = np.array([np.sin(phis) * np.cos(thetas), np.sin(phis) * np.sin(thetas), np.cos(phis)])
        grasp_dirs = grasp_dirs.T

        # convert to grasp objects
        grasps = []
        for i in range(num_grasps):
            grasp = ParallelJawPtGrasp3D(ParallelJawPtGrasp3D.configuration_from_params(grasp_centers[i,:], grasp_dirs[i,:], self.gripper.max_width))
            contacts_found, contacts = grasp.close_fingers(graspable)

            # add grasp if it has valid contacts
            if contacts_found and np.linalg.norm(contacts[0].point - contacts[1].point) > self.min_contact_dist:
                grasps.append(grasp)

        # visualize
        if vis:
            for grasp in grasps:
                plt.clf()
                h = plt.gcf()
                plt.ion()
                grasp.close_fingers(graspable, vis=vis)
                plt.show(block=False)
                time.sleep(0.5)

            grasp_centers_grid = graspable.sdf.transform_pt_obj_to_grid(grasp_centers.T)
            grasp_centers_grid = grasp_centers_grid.T
            com_grid = graspable.sdf.transform_pt_obj_to_grid(center_of_mass)

            plt.clf()
            ax = plt.gca(projection = '3d')
            graspable.sdf.scatter()
            ax.scatter(grasp_centers_grid[:,0], grasp_centers_grid[:,1], grasp_centers_grid[:,2], s=60, c=u'm')
            ax.scatter(com_grid[0], com_grid[1], com_grid[2], s=120, c=u'y')
            ax.set_xlim3d(0, graspable.sdf.dims_[0])
            ax.set_ylim3d(0, graspable.sdf.dims_[1])
            ax.set_zlim3d(0, graspable.sdf.dims_[2])
            plt.show()

        return grasps

class AntipodalGraspSampler(GraspSampler):
    """ Samples antipodal pairs using rejection sampling.
    The proposal sampling ditribution is to choose a random point on
    the object surface, then sample random directions within the friction cone, then form a grasp axis along the direction,
    close the fingers, and keep the grasp if the other contact point is also in the friction cone.
    """
    def sample_from_cone(self, n, tx, ty, num_samples=1):
        """ Samples directoins from within the friction cone using uniform sampling.
        
        Parameters
        ----------
        n : 3x1 normalized :obj:`numpy.ndarray`
            surface normal
        tx : 3x1 normalized :obj:`numpy.ndarray`
            tangent x vector
        ty : 3x1 normalized :obj:`numpy.ndarray`
            tangent y vector
        num_samples : int
            number of directions to sample

        Returns
        -------
        v_samples : :obj:`list` of 3x1 :obj:`numpy.ndarray`
            sampled directions in the friction cone
       """
        v_samples = []
        for i in range(num_samples):
            theta = 2 * np.pi * np.random.rand()
            r = self.friction_coef * np.random.rand()
            v = n + r * np.cos(theta) * tx + r * np.sin(theta) * ty
            v = -v / np.linalg.norm(v)
            v_samples.append(v)
        return v_samples

    def within_cone(self, cone, n, v):
        """
        Checks whether or not a direction is in the friction cone.
        This is equivalent to whether a grasp will slip using a point contact model.

        Parameters
        ----------
        cone : 3xN :obj:`numpy.ndarray`
            supporting vectors of the friction cone
        n : 3x1 :obj:`numpy.ndarray`
            outward pointing surface normal vector at c1
        v : 3x1 :obj:`numpy.ndarray`
            direction vector

        Returns
        -------
        in_cone : bool
            True if alpha is within the cone
        alpha : float
            the angle between the normal and v
        """
        if (v.dot(cone) < 0).any(): # v should point in same direction as cone
            v = -v # don't worry about sign, we don't know it anyway...
        f = -n / np.linalg.norm(n)
        alpha = np.arccos(f.T.dot(v) / np.linalg.norm(v))
        return alpha <= np.arctan(self.friction_coef), alpha

    def perturb_point(self, x, scale):
        """ Uniform random perturbations to a point """
        x_samp = x + (scale / 2.0) * (np.random.rand(3) - 0.5)
        return x_samp

    def sample_grasps(self, graspable, num_grasps,
                         vis=False):
        """Returns a list of candidate grasps for graspable object.

        Parameters
        ----------
        graspable : :obj:`GraspableObject3D`
            the object to grasp
        num_grasps : int
            number of grasps to sample
        vis : bool
            whether or not to visualize progress, for debugging

        Returns
        -------
        :obj:`list` of :obj:`ParallelJawPtGrasp3D`
            the sampled grasps
        """
        # get surface points
        grasps = []
        surface_points, _ = graspable.sdf.surface_points(grid_basis=False)
        np.random.shuffle(surface_points)
        shuffled_surface_points = surface_points[:min(self.max_num_surface_points_, len(surface_points))]
        logging.info('Num surface: %d' %(len(surface_points)))

        for k, x_surf in enumerate(shuffled_surface_points):
            start_time = time.clock()

            # perturb grasp for num samples
            for i in range(self.num_samples):
                # perturb contact (TODO: sample in tangent plane to surface)
                x1 = self.perturb_point(x_surf, graspable.sdf.resolution)

                # compute friction cone faces
                c1 = Contact3D(graspable, x1, in_direction=None)
                _, tx1, ty1 = c1.tangents()
                cone_succeeded, cone1, n1 = c1.friction_cone(self.num_cone_faces, self.friction_coef)
                if not cone_succeeded:
                    continue
                cone_time = time.clock()

                # sample grasp axes from friction cone
                v_samples = self.sample_from_cone(n1, tx1, ty1, num_samples=1)
                sample_time = time.clock()

                for v in v_samples:
                    if vis:
                        x1_grid = graspable.sdf.transform_pt_obj_to_grid(x1)
                        cone1_grid = graspable.sdf.transform_pt_obj_to_grid(cone1, direction=True)
                        plt.clf()
                        h = plt.gcf()
                        plt.ion()
                        ax = plt.gca(projection = '3d')
                        for i in range(cone1.shape[1]):
                            ax.scatter(x1_grid[0] - cone1_grid[0], x1_grid[1] - cone1_grid[1], x1_grid[2] - cone1_grid[2], s = 50, c = u'm')

                    # random axis flips since we don't have guarantees on surface normal directoins
                    if random.random() > 0.5:
                        v = -v

                    # start searching for contacts
                    grasp, c1, c2 = ParallelJawPtGrasp3D.grasp_from_contact_and_axis_on_grid(graspable, x1, v, self.gripper.max_width,
                                                                                             min_grasp_width_world=self.gripper.min_width,
                                                                    
                 vis=vis)
                    
                    if grasp is None or c2 is None:
                        continue

                    # get true contacts (previous is subject to variation)
                    success, c = grasp.close_fingers(graspable)
                    if not success:
                        continue
                    c1 = c[0]
                    c2 = c[1]

                    # make sure grasp is wide enough
                    x2 = c2.point
                    if np.linalg.norm(x1 - x2) < self.min_contact_dist:
                        continue

                    v_true = grasp.axis
                    # compute friction cone for contact 2
                    cone_succeeded, cone2, n2 = c2.friction_cone(self.num_cone_faces, self.friction_coef)
                    if not cone_succeeded:
                        continue

                    if vis:
                        plt.figure()
                        ax = plt.gca(projection='3d')
                        c1_proxy = c1.plot_friction_cone(color='m')
                        c2_proxy = c2.plot_friction_cone(color='y')
                        ax.view_init(elev=5.0, azim=0)
                        plt.show(block=False)
                        time.sleep(0.5)
                        plt.close() # lol

                    # check friction cone
                    if PointGraspMetrics3D.force_closure(c1, c2, self.friction_coef):
                        grasps.append(grasp)

        # randomly sample max num grasps from total list
        random.shuffle(grasps)
        return grasps









