"""
Class to encapsulate robot grippers
Author: Jeff
"""
import json
import numpy as np
import os
import sys

import IPython

import meshpy.obj_file as obj_file

from autolab_core import RigidTransform

GRIPPER_MESH_FILENAME = 'gripper.obj'
GRIPPER_PARAMS_FILENAME = 'params.json'
T_MESH_GRIPPER_FILENAME = 'T_mesh_gripper.tf' 
T_GRASP_GRIPPER_FILENAME = 'T_grasp_gripper.tf' 

class RobotGripper(object):
    """ Robot gripper wrapper for collision checking and encapsulation of grasp parameters (e.g. width, finger radius, etc)
    Note: The gripper frame should be the frame used to command the physical robot
    
    Attributes
    ----------
    name : :obj:`str`
        name of gripper
    mesh : :obj:`Mesh3D`
        3D triangular mesh specifying the geometry of the gripper
    params : :obj:`dict`
        set of parameters for the gripper, at minimum (finger_radius and grasp_width)
    T_mesh_gripper : :obj:`RigidTransform`
        transform from mesh frame to gripper frame (for rendering)
    T_grasp_gripper : :obj:`RigidTransform`
        transform from gripper frame to the grasp canonical frame (y-axis = grasp axis, x-axis = palm axis)
    """

    def __init__(self, name, mesh, mesh_filename, params, T_mesh_gripper, T_grasp_gripper):
        self.name = name
        self.mesh = mesh
        self.mesh_filename = mesh_filename
        self.T_mesh_gripper = T_mesh_gripper
        self.T_grasp_gripper = T_grasp_gripper
        for key, value in params.iteritems():
            setattr(self, key, value)

    def collides_with_table(self, grasp, stable_pose, clearance=0.0):
        """ Checks whether or not the gripper collides with the table in the stable pose.
        No longer necessary with CollisionChecker.

        Parameters
        ----------
        grasp : :obj:`ParallelJawPtGrasp3D`
            grasp parameterizing the pose of the gripper
        stable_pose : :obj:`StablePose`
            specifies the pose of the table
        clearance : float
            min distance from the table

        Returns
        -------
        bool
            True if collision, False otherwise
        """
        # transform mesh into object pose to check collisions with table
        T_obj_gripper = grasp.gripper_pose(self)

        T_obj_mesh = T_obj_gripper * self.T_mesh_gripper.inverse()
        mesh_tf = self.mesh.transform(T_obj_mesh.inverse())
        
        # extract table
        n = stable_pose.r[2,:]
        x0 = stable_pose.x0

        # check all vertices for intersection with table
        collision = False
        for vertex in mesh_tf.vertices():
            v = np.array(vertex)
            if n.dot(v - x0) < clearance:
                collision = True
        return collision

    @staticmethod
    def load(gripper_name, gripper_dir='data/grippers'):
        """ Load the gripper specified by gripper_name.

        Parameters
        ----------
        gripper_name : :obj:`str`
            name of the gripper to load
        gripper_dir : :obj:`str`
            directory where the gripper files are stored

        Returns
        -------
        :obj:`RobotGripper`
            loaded gripper objects
        """
        mesh_filename = os.path.join(gripper_dir, gripper_name, GRIPPER_MESH_FILENAME)
        mesh = obj_file.ObjFile(mesh_filename).read()
        
        f = open(os.path.join(os.path.join(gripper_dir, gripper_name, GRIPPER_PARAMS_FILENAME)), 'r')
        params = json.load(f)

        T_mesh_gripper = RigidTransform.load(os.path.join(gripper_dir, gripper_name, T_MESH_GRIPPER_FILENAME)) 
        T_grasp_gripper = RigidTransform.load(os.path.join(gripper_dir, gripper_name, T_GRASP_GRIPPER_FILENAME)) 
        return RobotGripper(gripper_name, mesh, mesh_filename, params, T_mesh_gripper, T_grasp_gripper)
