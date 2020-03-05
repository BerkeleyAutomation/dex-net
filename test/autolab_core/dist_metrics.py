"""
Custom distance metrics.
Author: Jeff Mahler
"""
import numpy as np

from .rigid_transformations import RigidTransform

def abs_angle_diff(v_i, v_j):
    """ 
    Returns the absolute value of the angle between two 3D vectors.
    
    Parameters
    ----------
    v_i : :obj:`numpy.ndarray`
        the first 3D array
    v_j : :obj:`numpy.ndarray`
        the second 3D array
    """
    # compute angle distance
    dot_prod = min(max(v_i.dot(v_j), -1), 1)
    angle_diff = np.arccos(dot_prod)
    return np.abs(angle_diff)

# dictionary of distance functions
DistMetrics = {
    'abs_angle_diff': abs_angle_diff
}
