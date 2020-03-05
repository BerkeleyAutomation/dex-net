"""
Lean rigid transformation class
Author: Jeff
"""
import logging
import os

import numpy as np
import scipy.linalg

from . import utils
from . import transformations
from .points import BagOfPoints, BagOfVectors, Point, PointCloud, Direction, NormalCloud
from .dual_quaternion import DualQuaternion

try:
    from geometry_msgs import msg
except:
    logging.warning('Failed to import geometry msgs in rigid_transformations.py.')
    
try:
    import rospy
    import rosservice
except ImportError:
    logging.warning("Failed to import ros dependencies in rigid_transforms.py")
    
try:
    from autolab_core.srv import *
except ImportError:
    logging.warning("autolab_core not installed as catkin package, RigidTransform ros methods will be unavailable")
    
import subprocess

TF_EXTENSION = '.tf'
STF_EXTENSION = '.stf'

class RigidTransform(object):
    """A Rigid Transformation from one frame to another.
    """

    def __init__(self, rotation=np.eye(3), translation=np.zeros(3),
                 from_frame='unassigned', to_frame='world'):
        """Initialize a RigidTransform.

        Parameters
        ----------
        rotation : :obj:`numpy.ndarray` of float
            A 3x3 rotation matrix (should be unitary).

        translation : :obj:`numpy.ndarray` of float
            A 3-entry translation vector.

        from_frame : :obj:`str`
            A name for the frame of reference on which this transform
            operates. This and to_frame are used for checking compositions
            of RigidTransforms, which is useful for debugging and catching
            errors.

        to_frame : :obj:`str`
            A name for the frame of reference to which this transform
            moves objects.

        Raises
        ------
        ValueError
            If any of the arguments are invalid. The frames must be strings or
            unicode, the translations and rotations must be ndarrays, have the
            correct shape, and the determinant of the rotation matrix should be
            1.0.
        """
        if not isinstance(from_frame, str) and not isinstance(from_frame, unicode):
            raise ValueError('Must provide string name of input frame of data')
        if not isinstance(to_frame, str) and not isinstance(to_frame, unicode):
            raise ValueError('Must provide string name of output frame of data')

        self.rotation = rotation
        self.translation = translation
        self._from_frame = str(from_frame)
        self._to_frame = str(to_frame)

    def copy(self):
        """Returns a copy of the RigidTransform.

        Returns
        -------
        :obj:`RigidTransform`
            A deep copy of the RigidTransform.
        """
        return RigidTransform(np.copy(self.rotation), np.copy(self.translation), self.from_frame, self.to_frame)

    def _check_valid_rotation(self, rotation):
        """Checks that the given rotation matrix is valid.
        """
        if not isinstance(rotation, np.ndarray) or not np.issubdtype(rotation.dtype, np.number):
            raise ValueError('Rotation must be specified as numeric numpy array')

        if len(rotation.shape) != 2 or rotation.shape[0] != 3 or rotation.shape[1] != 3:
            raise ValueError('Rotation must be specified as a 3x3 ndarray')

        if np.abs(np.linalg.det(rotation) - 1.0) > 1e-3:
            raise ValueError('Illegal rotation. Must have determinant == 1.0')

    def _check_valid_translation(self, translation):
        """Checks that the translation vector is valid.
        """
        if not isinstance(translation, np.ndarray) or not np.issubdtype(translation.dtype, np.number):
            raise ValueError('Translation must be specified as numeric numpy array')

        t = translation.squeeze()
        if len(t.shape) != 1 or t.shape[0] != 3:
            raise ValueError('Translation must be specified as a 3-vector, 3x1 ndarray, or 1x3 ndarray')

    @property
    def rotation(self):
        """:obj:`numpy.ndarray` of float: A 3x3 rotation matrix.
        """
        return self._rotation

    @rotation.setter
    def rotation(self, rotation):
        # Convert quaternions
        if len(rotation) == 4:
            q = np.array([q for q in rotation])
            if np.abs(np.linalg.norm(q) - 1.0) > 1e-3:
                raise ValueError('Invalid quaternion. Must be norm 1.0')
            rotation = RigidTransform.rotation_from_quaternion(q)

        # Convert lists and tuples
        if type(rotation) in (list, tuple):
            rotation = np.array(rotation).astype(np.float32)

        self._check_valid_rotation(rotation)
        self._rotation = rotation * 1.

    @property
    def translation(self):
        """:obj:`numpy.ndarray` of float: A 3-ndarray that represents the
        transform's translation vector.
        """
        return self._translation

    @translation.setter
    def translation(self, translation):
        # Convert lists to translation arrays
        if type(translation) in (list, tuple) and len(translation) == 3:
            translation = np.array([t for t in translation]).astype(np.float32)

        self._check_valid_translation(translation)
        self._translation = translation.squeeze() * 1.

    @property
    def position(self):
        """:obj:`numpy.ndarray` of float: A 3-ndarray that represents the
        transform's translation vector (same as translation).
        """
        return self._translation

    @position.setter
    def position(self, position):
        self.translation = position

    @property
    def adjoint_tf(self):
        A = np.zeros([6,6])
        A[:3,:3] = self.rotation
        A[3:,:3] = utils.skew(self.translation).dot(self.rotation)
        A[3:,3:] = self.rotation
        return A
        
    @property
    def from_frame(self):
        """:obj:`str`: The identifier for the 'from' frame of reference.
        """
        return self._from_frame

    @from_frame.setter
    def from_frame(self, from_frame):
        self._from_frame = str(from_frame)

    @property
    def to_frame(self):
        """:obj:`str`: The identifier for the 'to' frame of reference.
        """
        return self._to_frame

    @to_frame.setter
    def to_frame(self, to_frame):
        self._to_frame = str(to_frame)

    @property
    def euler_angles(self):
        """:obj:`tuple` of float: The three euler angles for the rotation.
        """
        q_wxyz = self.quaternion
        q_xyzw = np.roll(q_wxyz, -1)
        return transformations.euler_from_quaternion(q_xyzw)

    @property
    def quaternion(self):
        """:obj:`numpy.ndarray` of float: A quaternion vector in wxyz layout.
        """
        q_xyzw = transformations.quaternion_from_matrix(self.matrix)
        q_wxyz = np.roll(q_xyzw, 1)
        return q_wxyz

    @property
    def dual_quaternion(self):
        """:obj:`DualQuaternion`: The DualQuaternion corresponding to this
        transform.
        """
        qr = self.quaternion
        qd = np.append([0], self.translation / 2.)
        return DualQuaternion(qr, qd)

    @property
    def axis_angle(self):
        """:obj:`numpy.ndarray` of float: The axis-angle representation for the rotation.
        """
        qw, qx, qy, qz = self.quaternion
        theta = 2 * np.arccos(qw)
        omega = np.array([1,0,0])
        if theta > 0:
            rx = qx / np.sqrt(1.0 - qw**2)
            ry = qy / np.sqrt(1.0 - qw**2)
            rz = qz / np.sqrt(1.0 - qw**2)
            omega = np.array([rx, ry, rz])
        return theta * omega
    
    @property
    def euler(self):
        """TODO DEPRECATE THIS?"""
        e_xyz = transformations.euler_from_matrix(self.rotation, 'sxyz')
        return np.array([180.0 / np.pi * a for a in e_xyz])

    @property
    def vec(self):
        return np.r_[self.translation, self.quaternion]
   
    @property
    def matrix(self):
        """:obj:`numpy.ndarray` of float: The canonical 4x4 matrix
        representation of this transform.

        The first three columns contain the columns of the rotation matrix
        followed by a zero, and the last column contains the translation vector
        followed by a one.
        """
        return np.r_[np.c_[self._rotation, self._translation], [[0,0,0,1]]]

    @property
    def x_axis(self):
        """:obj:`numpy.ndarray` of float: X axis of 'from' frame in 'to' basis.
        """
        return self.rotation[:,0]

    @property
    def y_axis(self):
        """:obj:`numpy.ndarray` of float: Y axis of 'from' frame in 'to' basis.
        """
        return self.rotation[:,1]

    @property
    def z_axis(self):
        """:obj:`numpy.ndarray` of float: Z axis of 'from' frame in 'to' basis.
        """
        return self.rotation[:,2]

    @property
    def pose_msg(self):
        """:obj:`geometry_msgs.msg.Pose` The rigid transform as a geometry_msg pose.
        """
        pose = msg.Pose()
        pose.orientation.w = float(self.quaternion[0])
        pose.orientation.x = float(self.quaternion[1])
        pose.orientation.y = float(self.quaternion[2])
        pose.orientation.z = float(self.quaternion[3])
        pose.position.x = float(self.translation[0])
        pose.position.y = float(self.translation[1])
        pose.position.z = float(self.translation[2])
        return pose

    @property
    def frames(self):
        """:obj:`str`: A string represeting the frame transform: from {} to {}.
        """
        return 'from {0} to {1}'.format(self.from_frame, self.to_frame)

    def interpolate_with(self, other_tf, t):
        """Interpolate with another rigid transformation.

        Parameters
        ----------
        other_tf : :obj:`RigidTransform`
            The transform to interpolate with.

        t : float
            The interpolation step in [0,1], where 0 favors this RigidTransform.

        Returns
        -------
        :obj:`RigidTransform`
            The interpolated RigidTransform.

        Raises
        ------
        ValueError
            If t isn't in [0,1].
        """
        if t < 0 or t > 1:
            raise ValueError('Must interpolate between 0 and 1')

        interp_translation = (1.0 - t) * self.translation + t * other_tf.translation
        interp_rotation = transformations.quaternion_slerp(self.quaternion, other_tf.quaternion, t)
        interp_tf = RigidTransform(rotation=interp_rotation, translation=interp_translation,
                                  from_frame = self.from_frame, to_frame = self.to_frame)
        return interp_tf

    def linear_trajectory_to(self, target_tf, traj_len):
        """Creates a trajectory of poses linearly interpolated from this tf to a target tf.

        Parameters
        ----------
        target_tf : :obj:`RigidTransform`
            The RigidTransform to interpolate to.
        traj_len : int
            The number of RigidTransforms in the returned trajectory.

        Returns
        -------
        :obj:`list` of :obj:`RigidTransform`
            A list of interpolated transforms from this transform to the target.
        """
        if traj_len < 0:
            raise ValueError('Traj len must at least 0')
        delta_t = 1.0 / (traj_len + 1)
        t = 0.0
        traj = []
        while t < 1.0:
            traj.append(self.interpolate_with(target_tf, t))
            t += delta_t
        traj.append(target_tf)
        return traj

    def apply(self, points):
        """Applies the rigid transformation to a set of 3D objects.

        Parameters
        ----------
        points : :obj:`BagOfPoints`
            A set of objects to transform. Could be any subclass of BagOfPoints.

        Returns
        -------
        :obj:`BagOfPoints`
            A transformed set of objects of the same type as the input.

        Raises
        ------
        ValueError
            If the input is not a Bag of 3D points or if the points are not in
            this transform's from_frame.
        """
        if not isinstance(points, BagOfPoints):
            raise ValueError('Rigid transformations can only be applied to bags of points')
        if points.dim != 3:
            raise ValueError('Rigid transformations can only be applied to 3-dimensional points')
        if points.frame != self._from_frame:
            raise ValueError('Cannot transform points in frame %s with rigid transformation from frame %s to frame %s' %(points.frame, self._from_frame, self._to_frame))

        if isinstance(points, BagOfVectors):
            # rotation only
            x = points.data
            x_tf = self.rotation.dot(x)
        else:
            # extract numpy data, homogenize, and transform
            x = points.data
            if len(x.shape) == 1:
                x = x[:,np.newaxis]
            x_homog = np.r_[x, np.ones([1, points.num_points])]
            x_homog_tf = self.matrix.dot(x_homog)
            x_tf = x_homog_tf[0:3,:]

        # output in BagOfPoints format
        if isinstance(points, PointCloud):
            return PointCloud(x_tf, frame=self._to_frame)
        elif isinstance(points, Point):
            return Point(x_tf, frame=self._to_frame)
        elif isinstance(points, Direction):
            return Direction(x_tf, frame=self._to_frame)
        elif isinstance(points, NormalCloud):
            return NormalCloud(x_tf, frame=self._to_frame)
        raise ValueError('Type %s not yet supported' %(type(points)))

    def dot(self, other_tf):
        """Compose this rigid transform with another.

        This transform is on the left-hand side of the composition.

        Parameters
        ----------
        other_tf : :obj:`RigidTransform`
            The other RigidTransform to compose with this one.

        Returns
        -------
        :obj:`RigidTransform`
            A RigidTransform that represents the composition.

        Raises
        ------
        ValueError
            If the to_frame of other_tf is not identical to this transform's
            from_frame.
        """
        if other_tf.to_frame != self.from_frame:
            raise ValueError('To frame of right hand side ({0}) must match from frame of left hand side ({1})'.format(other_tf.to_frame, self.from_frame))

        pose_tf = self.matrix.dot(other_tf.matrix)
        rotation, translation = RigidTransform.rotation_and_translation_from_matrix(pose_tf)

        if isinstance(other_tf, SimilarityTransform):
            return SimilarityTransform(self.rotation, self.translation, scale=1.0,
                                       from_frame=self.from_frame,
                                       to_frame=self.to_frame) * other_tf
        return RigidTransform(rotation, translation,
                              from_frame=other_tf.from_frame,
                              to_frame=self.to_frame)

    def __mul__(self, rigid_object):
        """Selects composition of rigid transforms based on input type.

        If the input is a BagOfPoints-type, it applies the transform.
        Otherwise, if it is another RigidTransform, it composes them.

        Parameters
        ----------
        rigid_object : :obj:`RigidTransform` or :obj:`BagOfPoints`
            The rigid object to multiply by this transform.

        Returns
        -------
        :obj:`RigidTransform` or :obj:`BagOfPoints`
            An object of the same type as the input.

        Raises
        ------
        ValueError
            If the input is not of one of the accepted types.
        """
        if isinstance(rigid_object, RigidTransform):
            return self.dot(rigid_object)
        if isinstance(rigid_object, BagOfPoints):
            return self.apply(rigid_object)
        raise ValueError('Cannot multiply rigid transform with object of type %s' %(type(rigid_object)))
                              
    def inverse(self):
        """Take the inverse of the rigid transform.

        Returns
        -------
        :obj:`RigidTransform`
            The inverse of this RigidTransform.
        """
        inv_rotation = self.rotation.T
        inv_translation = np.dot(-self.rotation.T, self.translation)
        return RigidTransform(inv_rotation, inv_translation,
                              from_frame=self._to_frame,
                              to_frame=self._from_frame)

    def save(self, filename):
        """Save the RigidTransform to a file.

        The file format is:
        from_frame
        to_frame
        translation (space separated)
        rotation_row_0 (space separated)
        rotation_row_1 (space separated)
        rotation_row_2 (space separated)

        Parameters
        ----------
        filename : :obj:`str`
            The file to save the transform to.

        Raises
        ------
        ValueError
            If filename's extension isn't .tf.
        """
        file_root, file_ext = os.path.splitext(filename)
        if file_ext.lower() != TF_EXTENSION:
            raise ValueError('Extension %s not supported for RigidTransform. Must be stored with extension %s' %(file_ext, TF_EXTENSION))

        f = open(filename, 'w')
        f.write('%s\n' %(self._from_frame))
        f.write('%s\n' %(self._to_frame))
        f.write('%f %f %f\n' %(self._translation[0], self._translation[1], self._translation[2]))
        f.write('%f %f %f\n' %(self._rotation[0, 0], self._rotation[0, 1], self._rotation[0, 2]))
        f.write('%f %f %f\n' %(self._rotation[1, 0], self._rotation[1, 1], self._rotation[1, 2]))
        f.write('%f %f %f\n' %(self._rotation[2, 0], self._rotation[2, 1], self._rotation[2, 2]))
        f.close()

    def as_frames(self, from_frame, to_frame='world'):
        """Return a shallow copy of this rigid transform with just the frames
        changed.

        Parameters
        ----------
        from_frame : :obj:`str`
            The new from_frame.

        to_frame : :obj:`str`
            The new to_frame.

        Returns
        -------
        :obj:`RigidTransform`
            The RigidTransform with new frames.
        """
        return RigidTransform(self.rotation, self.translation, from_frame, to_frame)
    
    def publish_to_ros(self, mode='transform', service_name='rigid_transforms/rigid_transform_publisher', namespace=None):
        """Publishes RigidTransform to ROS
        If a transform referencing the same frames already exists in the ROS publisher, it is updated instead.
        This checking is not order sensitive
        
        Requires ROS rigid_transform_publisher service to be running. Assuming autolab_core is installed as a catkin package,
        this can be done with: roslaunch autolab_core rigid_transforms.launch
        
        Parameters
        ----------
        mode : :obj:`str`
            Mode in which to publish. In {'transform', 'frame'}
            Defaults to 'transform'
        service_name : string, optional
            RigidTransformPublisher service to interface with. If the RigidTransformPublisher services are started through
            rigid_transforms.launch it will be called rigid_transform_publisher
        namespace : string, optional
            Namespace to prepend to transform_listener_service. If None, current namespace is prepended.
        
        Raises
        ------
        rospy.ServiceException
            If service call to rigid_transform_publisher fails
        """
        if namespace == None:
            service_name = rospy.get_namespace() + service_name
        else:
            service_name = namespace + service_name
        
        rospy.wait_for_service(service_name, timeout = 10)
        publisher = rospy.ServiceProxy(service_name, RigidTransformPublisher)
        
        trans = self.translation
        rot = self.quaternion
        
        publisher(trans[0], trans[1], trans[2], rot[0], rot[1], rot[2], rot[3], self.from_frame, self.to_frame, mode)
        
    def delete_from_ros(self, service_name='rigid_transforms/rigid_transform_publisher', namespace=None):
        """Removes RigidTransform referencing from_frame and to_frame from ROS publisher.
        Note that this may not be this exact transform, but may that references the same frames (order doesn't matter)
        
        Also, note that it may take quite a while for the transform to disappear from rigid_transform_publisher's cache 
        
        Requires ROS rigid_transform_publisher service to be running. Assuming autolab_core is installed as a catkin package,
        this can be done with: roslaunch autolab_core rigid_transforms.launch
        
        Parameters
        ----------
        service_name : string, optional
            RigidTransformPublisher service to interface with. If the RigidTransformPublisher services are started through
            rigid_transforms.launch it will be called rigid_transform_publisher
        namespace : string, optional
            Namespace to prepend to transform_listener_service. If None, current namespace is prepended.
        
        Raises
        ------
        rospy.ServiceException
            If service call to rigid_transform_publisher fails
        """
        if namespace == None:
            service_name = rospy.get_namespace() + service_name
        else:
            service_name = namespace + service_name
            
        rospy.wait_for_service(service_name, timeout = 10)
        publisher = rospy.ServiceProxy(service_name, RigidTransformPublisher)
        
        publisher(0, 0, 0, 0, 0, 0, 0, self.from_frame, self.to_frame, 'delete')

    def __str__(self):
        out = 'Tra: {0}\n Rot: {1}\n Qtn: {2}\n from {3} to {4}'.format(self.translation, self.rotation,
            self.quaternion, self.from_frame, self.to_frame)
        return out

    def __repr__(self):
        out = 'RigidTransform(rotation=np.{0}, translation=np.{1}, from_frame={2}, to_frame={3})'.format(repr(self.rotation),
                repr(self.translation), repr(self.from_frame), repr(self.to_frame))
        return out

    @staticmethod
    def ros_q_to_core_q(q_ros):
        """Converts a ROS quaternion vector to an autolab_core quaternion vector."""
        q_core = np.array([q_ros[3], q_ros[0], q_ros[1], q_ros[2]])
        return q_core

    @staticmethod
    def core_q_to_ros_q(q_core):
        """Converts a ROS quaternion vector to an autolab_core quaternion vector."""
        q_ros = np.array([q_core[1], q_core[2], q_core[3], q_core[0]])
        return q_ros

    @staticmethod
    def from_ros_pose_msg(pose_msg,
                          from_frame='unassigned',
                          to_frame='world'):
        """Creates a RigidTransform from a ROS pose msg. 
        
        Parameters
        ----------
        pose_msg : :obj:`geometry_msgs.msg.Pose`
            ROS pose message
        """
        quaternion = np.array([pose_msg.orientation.w,
                               pose_msg.orientation.x,
                               pose_msg.orientation.y,
                               pose_msg.orientation.z])
        position = np.array([pose_msg.position.x,
                             pose_msg.position.y,
                             pose_msg.position.z])
        pose = RigidTransform(rotation=quaternion,
                              translation=position,
                              from_frame=from_frame,
                              to_frame=to_frame)
        return pose        
        
    @staticmethod
    def from_vec(vec, from_frame='unassigned', to_frame='world'):
        return RigidTransform(rotation=vec[3:],
                              translation=vec[:3],
                              from_frame=from_frame,
                              to_frame=to_frame)

    @staticmethod
    def from_pose_msg(pose_msg, from_frame='unassigned', to_frame='world'):
        translation = np.array([pose_msg.position.x,
                                pose_msg.position.y,
                                pose_msg.position.z])
        rotation = np.array([pose_msg.orientation.w,
                             pose_msg.orientation.x,
                             pose_msg.orientation.y,
                             pose_msg.orientation.z])
        return RigidTransform(rotation=rotation,
                              translation=translation,
                              from_frame=from_frame,
                              to_frame=to_frame)

    
    @staticmethod
    def rigid_transform_from_ros(from_frame, to_frame, service_name='rigid_transforms/rigid_transform_listener', namespace=None):
        """Gets transform from ROS as a rigid transform
        
        Requires ROS rigid_transform_publisher service to be running. Assuming autolab_core is installed as a catkin package,
        this can be done with: roslaunch autolab_core rigid_transforms.launch
        
        Parameters
        ----------
        from_frame : :obj:`str`
        to_frame : :obj:`str`
        service_name : string, optional
            RigidTransformListener service to interface with. If the RigidTransformListener services are started through
            rigid_transforms.launch it will be called rigid_transform_listener
        namespace : string, optional
            Namespace to prepend to transform_listener_service. If None, current namespace is prepended.
        
        Raises
        ------
        rospy.ServiceException
            If service call to rigid_transform_listener fails
        """
        if namespace == None:
            service_name = rospy.get_namespace() + service_name
        else:
            service_name = namespace + service_name
        
        rospy.wait_for_service(service_name, timeout = 10)
        listener = rospy.ServiceProxy(service_name, RigidTransformListener)
        
        ret = listener(from_frame, to_frame)
        
        quat = np.asarray([ret.w_rot, ret.x_rot, ret.y_rot, ret.z_rot])
        trans = np.asarray([ret.x_trans, ret.y_trans, ret.z_trans])
        
        rot = RigidTransform.rotation_from_quaternion(quat)
        
        return RigidTransform(rotation=rot, translation=trans, from_frame=from_frame, to_frame=to_frame)
        
    @staticmethod
    def rotation_from_quaternion(q_wxyz):
        """Convert quaternion array to rotation matrix.

        Parameters
        ----------
        q_wxyz : :obj:`numpy.ndarray` of float
            A quaternion in wxyz order.

        Returns
        -------
        :obj:`numpy.ndarray` of float
            A 3x3 rotation matrix made from the quaternion.
        """
        q_xyzw = np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]])
        R = transformations.quaternion_matrix(q_xyzw)[:3,:3]
        return R


    @staticmethod
    def quaternion_from_axis_angle(v):
        """Convert axis-angle representation to a quaternion vector.

        Parameters
        ----------
        v : :obj:`numpy.ndarray` of float
            An axis-angle representation.

        Returns
        -------
        :obj:`numpy.ndarray` of float
            A quaternion vector from the axis-angle vector.
        """
        theta = np.linalg.norm(v)
        if theta > 0:
            v = v / np.linalg.norm(v)
        ax, ay, az = v    
        qx = ax * np.sin(0.5 * theta)
        qy = ay * np.sin(0.5 * theta)
        qz = az * np.sin(0.5 * theta)        
        qw = np.cos(0.5 * theta)
        q = np.array([qw, qx, qy, qz])
        return q
        
    @staticmethod
    def rotation_from_axis_angle(v):
        """Convert axis-angle representation to rotation matrix.

        Parameters
        ----------
        v : :obj:`numpy.ndarray` of float
            An axis-angle representation.

        Returns
        -------
        :obj:`numpy.ndarray` of float
            A 3x3 rotation matrix made from the axis-angle vector.
        """
        return RigidTransform.rotation_from_quaternion(RigidTransform.quaternion_from_axis_angle(v))
        
    @staticmethod
    def transform_from_dual_quaternion(dq, from_frame='unassigned', to_frame='world'):
        """Create a RigidTransform from a DualQuaternion.

        Parameters
        ----------
        dq : :obj:`DualQuaternion`
            The DualQuaternion to transform.

        from_frame : :obj:`str`
            A name for the frame of reference on which this transform
            operates.

        to_frame : :obj:`str`
            A name for the frame of reference to which this transform
            moves objects.

        Returns
        -------
        :obj:`RigidTransform`
            The RigidTransform made from the DualQuaternion.
        """
        quaternion = dq.qr
        translation = 2 * dq.qd[1:]
        return RigidTransform(rotation=quaternion, translation=translation, from_frame=from_frame, to_frame=to_frame)

    @staticmethod
    def rotation_and_translation_from_matrix(matrix):
        """Helper to convert 4x4 matrix to rotation matrix and translation vector.

        Parameters
        ----------
        matrix : :obj:`numpy.ndarray` of float
            4x4 rigid transformation matrix to be converted.

        Returns
        -------
        :obj:`tuple` of :obj:`numpy.ndarray` of float
            A 3x3 rotation matrix and a 3-entry translation vector.

        Raises
        ------
        ValueError
            If the incoming matrix isn't a 4x4 ndarray.
        """
        if not isinstance(matrix, np.ndarray) or \
                matrix.shape[0] != 4 or matrix.shape[1] != 4:
            raise ValueError('Matrix must be specified as a 4x4 ndarray')
        rotation = matrix[:3,:3]
        translation = matrix[:3,3]
        return rotation, translation

    @staticmethod
    def rotation_from_axis_and_origin(axis, origin, angle, to_frame='world'):
        """
        Returns a rotation matrix around some arbitrary axis, about the point origin, using Rodrigues Formula

        Parameters
        ----------
        axis : :obj:`numpy.ndarray` of float
            3x1 vector representing which axis we should be rotating about
        origin : :obj:`numpy.ndarray` of float
            3x1 vector representing where the rotation should be centered around
        angle : float
            how much to rotate (in radians)
        to_frame : :obj:`str`
            A name for the frame of reference to which this transform
            moves objects.
        """
        axis_hat = np.array([[0, -axis[2], axis[1]],
                             [axis[2], 0, -axis[0]],
                             [-axis[1], axis[0], 0]])
        # Rodrigues Formula
        R = RigidTransform(
            np.eye(3) + np.sin(angle) * axis_hat + (1 - np.cos(angle)) * axis_hat.dot(axis_hat),
            from_frame=to_frame,
            to_frame=to_frame
        )

        return RigidTransform(translation=origin, from_frame=to_frame, to_frame=to_frame) \
            .dot(R) \
            .dot(RigidTransform(translation=-origin, from_frame=to_frame, to_frame=to_frame))

    @staticmethod
    def x_axis_rotation(theta):
        """Generates a 3x3 rotation matrix for a rotation of angle
        theta about the x axis.

        Parameters
        ----------
        theta : float
            amount to rotate, in radians

        Returns
        -------
        :obj:`numpy.ndarray` of float
            A random 3x3 rotation matrix.
        """
        R = np.array([[1, 0, 0,],
                      [0, np.cos(theta), -np.sin(theta)],
                      [0, np.sin(theta), np.cos(theta)]])
        return R

    @staticmethod
    def y_axis_rotation(theta):
        """Generates a 3x3 rotation matrix for a rotation of angle
        theta about the y axis.

        Parameters
        ----------
        theta : float
            amount to rotate, in radians

        Returns
        -------
        :obj:`numpy.ndarray` of float
            A random 3x3 rotation matrix.
        """
        R = np.array([[np.cos(theta), 0, np.sin(theta)],
                      [0, 1, 0],
                      [-np.sin(theta), 0, np.cos(theta)]])
        return R

    @staticmethod
    def z_axis_rotation(theta):
        """Generates a 3x3 rotation matrix for a rotation of angle
        theta about the z axis.

        Parameters
        ----------
        theta : float
            amount to rotate, in radians

        Returns
        -------
        :obj:`numpy.ndarray` of float
            A random 3x3 rotation matrix.
        """
        R = np.array([[np.cos(theta), -np.sin(theta), 0],
                      [np.sin(theta), np.cos(theta), 0],
                      [0, 0, 1]])
        return R

    @staticmethod
    def random_rotation():
        """Generates a random 3x3 rotation matrix with SVD.

        Returns
        -------
        :obj:`numpy.ndarray` of float
            A random 3x3 rotation matrix.
        """
        rand_seed = np.random.rand(3, 3)
        U, S, V = np.linalg.svd(rand_seed)
        return U

    @staticmethod
    def random_translation():
        """Generates a random translation vector.

        Returns
        -------
        :obj:`numpy.ndarray` of float
            A 3-entry random translation vector.
        """
        return np.random.rand(3)

    @staticmethod
    def rotation_from_axes(x_axis, y_axis, z_axis):
        """Convert specification of axis in target frame to
        a rotation matrix from source to target frame.

        Parameters
        ----------
        x_axis : :obj:`numpy.ndarray` of float
            A normalized 3-vector for the target frame's x-axis.

        y_axis : :obj:`numpy.ndarray` of float
            A normalized 3-vector for the target frame's y-axis.

        z_axis : :obj:`numpy.ndarray` of float
            A normalized 3-vector for the target frame's z-axis.

        Returns
        -------
        :obj:`numpy.ndarray` of float
            A 3x3 rotation matrix that transforms from a source frame to the
            given target frame.
        """
        return np.hstack((x_axis[:,np.newaxis], y_axis[:,np.newaxis], z_axis[:,np.newaxis]))

    @staticmethod
    def sph_coords_to_pose(theta, psi):
        """ Convert spherical coordinates to a pose.
        
        Parameters
        ----------
        theta : float
            azimuth angle
        psi : float
            elevation angle

        Returns
        -------
        :obj:`RigidTransformation`
            rigid transformation corresponding to rotation with no translation
        """
        # rotate about the z and y axes individually
        rot_z = RigidTransform.z_axis_rotation(theta)
        rot_y = RigidTransform.y_axis_rotation(psi)
        R = rot_y.dot(rot_z)
        return RigidTransform(rotation=R)        

    @staticmethod
    def interpolate(T0, T1, t):
        """Return an interpolation of two RigidTransforms.

        Parameters
        ----------
        T0 : :obj:`RigidTransform`
            The first RigidTransform to interpolate.

        T1 : :obj:`RigidTransform`
            The second RigidTransform to interpolate.

        t : float
            The interpolation step in [0,1]. 0 favors T0, 1 favors T1.

        Returns
        -------
        :obj:`RigidTransform`
            The interpolated RigidTransform.

        Raises
        ------
        ValueError
            If the to_frame of the two RigidTransforms are not identical.
        """
        if T0.to_frame != T1.to_frame:
            raise ValueError('Cannot interpolate between 2 transforms with different to frames! Got T1 {0} and T2 {1}'.format(T0.to_frame, T1.to_frame))
        dq0 = T0.dual_quaternion
        dq1 = T1.dual_quaternion

        dqt = DualQuaternion.interpolate(dq0, dq1, t)
        from_frame = "{0}_{1}_{2}".format(T0.from_frame, T1.from_frame, t)
        return RigidTransform.transform_from_dual_quaternion(dqt, from_frame, T0.to_frame)

    @staticmethod
    def load(filename):
        """Load a RigidTransform from a file.

        The file format is:
        from_frame
        to_frame
        translation (space separated)
        rotation_row_0 (space separated)
        rotation_row_1 (space separated)
        rotation_row_2 (space separated)

        Parameters
        ----------
        filename : :obj:`str`
            The file to load the transform from.

        Returns
        -------
        :obj:`RigidTransform`
            The RigidTransform read from the file.

        Raises
        ------
        ValueError
            If filename's extension isn't .tf.
        """
        file_root, file_ext = os.path.splitext(filename)
        if file_ext.lower() != TF_EXTENSION:
            raise ValueError('Extension %s not supported for RigidTransform. Can only load extension %s' %(file_ext, TF_EXTENSION))

        f = open(filename, 'r')
        lines = list(f)
        from_frame = lines[0][:-1]
        to_frame = lines[1][:-1]

        t = np.zeros(3)
        t_tokens = lines[2][:-1].split()
        t[0] = float(t_tokens[0])
        t[1] = float(t_tokens[1])
        t[2] = float(t_tokens[2])

        R = np.zeros([3,3])
        r_tokens = lines[3][:-1].split()
        R[0, 0] = float(r_tokens[0])
        R[0, 1] = float(r_tokens[1])
        R[0, 2] = float(r_tokens[2])

        r_tokens = lines[4][:-1].split()
        R[1, 0] = float(r_tokens[0])
        R[1, 1] = float(r_tokens[1])
        R[1, 2] = float(r_tokens[2])

        r_tokens = lines[5][:-1].split()
        R[2, 0] = float(r_tokens[0])
        R[2, 1] = float(r_tokens[1])
        R[2, 2] = float(r_tokens[2])
        f.close()
        return RigidTransform(rotation=R, translation=t,
                              from_frame=from_frame,
                              to_frame=to_frame)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return hash(self) == hash(other)
        return False

    def __ne__(self, other):
        if isinstance(other, self.__class__):
            return not self.__eq__(other)
        return NotImplemented

    def __hash__(self):
        return hash(str(self.__dict__))

class SimilarityTransform(RigidTransform):
    """ A Similarity Transformation from one frame to another (rigid transformation + scaling)
    """
    def __init__(self, rotation=np.eye(3), translation=np.zeros(3), scale=1.0,
                 from_frame='unassigned', to_frame='world'):
        """Initialize a SimilarityTransform.

        Parameters
        ----------
        rotation : :obj:`numpy.ndarray` of float
            A 3x3 rotation matrix (should be unitary).

        translation : :obj:`numpy.ndarray` of float
            A 3-entry translation vector.

        scale : float
            Rescaling factor to the output reference frame

        from_frame : :obj:`str`
            A name for the frame of reference on which this transform
            operates. This and to_frame are used for checking compositions
            of RigidTransforms, which is useful for debugging and catching
            errors.

        to_frame : :obj:`str`
            A name for the frame of reference to which this transform
            moves objects.

        Raises
        ------
        ValueError
            If any of the arguments are invalid. The frames must be strings or
            unicode, the translations and rotations must be ndarrays, have the
            correct shape, and the determinant of the rotation matrix should be
            1.0.
        """
        self.scale = scale
        RigidTransform.__init__(self, rotation, translation, from_frame, to_frame)

    @property
    def scale(self):
        """ float : scaling factor """
        return self._scale

    @scale.setter
    def scale(self, scale):
        self._scale = scale

    @property
    def matrix(self):
        matrix = np.r_[np.c_[self._rotation, self._translation], [[0,0,0,1]]]
        scale_mat = np.eye(4)
        scale_mat[:3,:3] = np.diag(self.scale * np.ones(3))
        matrix = matrix.dot(scale_mat)
        return matrix

    def apply(self, points):
        """Applies the similarity transformation to a set of 3D objects.

        Parameters
        ----------
        points : :obj:`BagOfPoints`
            A set of objects to transform. Could be any subclass of BagOfPoints.

        Returns
        -------
        :obj:`BagOfPoints`
            A transformed set of objects of the same type as the input.

        Raises
        ------
        ValueError
            If the input is not a Bag of 3D points or if the points are not in
            this transform's from_frame.
        """
        if not isinstance(points, BagOfPoints):
            raise ValueError('Rigid transformations can only be applied to bags of points')
        if points.dim != 3:
            raise ValueError('Rigid transformations can only be applied to 3-dimensional points')
        if points.frame != self._from_frame:
            raise ValueError('Cannot transform points in frame %s with rigid transformation from frame %s to frame %s' %(points.frame, self._from_frame, self._to_frame))

        if isinstance(points, BagOfVectors):
            # rotation only
            x = points.data
            x_tf = self.rotation.dot(x)
        else:
            # extract numpy data, homogenize, and transform
            x = points.data
            if len(x.shape) == 1:
                x = x[:,np.newaxis]
            x_homog = np.r_[x, np.ones([1, points.num_points])]
            x_homog_tf = self.matrix.dot(x_homog)
            x_tf = x_homog_tf[0:3,:]

        # output in BagOfPoints format
        if isinstance(points, PointCloud):
            return PointCloud(x_tf, frame=self._to_frame)
        elif isinstance(points, Point):
            return Point(x_tf, frame=self._to_frame)
        elif isinstance(points, Direction):
            return Direction(x_tf, frame=self._to_frame)
        elif isinstance(points, NormalCloud):
            return NormalCloud(x_tf, frame=self._to_frame)
        raise ValueError('Type %s not yet supported' %(type(points)))

    def dot(self, other_tf):
        """Compose this simliarity transform with another.

        This transform is on the left-hand side of the composition.

        Parameters
        ----------
        other_tf : :obj:`SimilarityTransform`
            The other SimilarityTransform to compose with this one.

        Returns
        -------
        :obj:`SimilarityTransform`
            A SimilarityTransform that represents the composition.

        Raises
        ------
        ValueError
            If the to_frame of other_tf is not identical to this transform's
            from_frame.
        """
        if other_tf.to_frame != self.from_frame:
            raise ValueError('To frame of right hand side ({0}) must match from frame of left hand side ({1})'.format(other_tf.to_frame, self.from_frame))
        if not isinstance(other_tf, RigidTransform):
            raise ValueError('Can only compose with other RigidTransform classes')

        other_scale = 1.0
        if isinstance(other_tf, SimilarityTransform):
            other_scale = other_tf.scale

        rotation = self.rotation.dot(other_tf.rotation)
        translation = self.translation + self.scale * self.rotation.dot(other_tf.translation)
        scale = self.scale * other_scale
        return SimilarityTransform(rotation, translation, scale,
                                   from_frame=other_tf.from_frame,
                                   to_frame=self.to_frame)

    def inverse(self):
        """Take the inverse of the similarity transform.

        Returns
        -------
        :obj:`SimilarityTransform`
            The inverse of this SimilarityTransform.
        """
        inv_rot = np.linalg.inv(self.rotation)
        inv_scale = 1.0 / self.scale
        inv_trans = -inv_scale * inv_rot.dot(self.translation)
        return SimilarityTransform(inv_rot, inv_trans, inv_scale,
                                   from_frame=self._to_frame,
                                   to_frame=self._from_frame)

    def save(self, filename):
        """Save the SimliarityTransform to a file.

        The file format is:
        from_frame
        to_frame
        scale
        translation (space separated)
        rotation_row_0 (space separated)
        rotation_row_1 (space separated)
        rotation_row_2 (space separated)

        Parameters
        ----------
        filename : :obj:`str`
            The file to save the transform to.

        Raises
        ------
        ValueError
            If filename's extension isn't .stf.
        """
        file_root, file_ext = os.path.splitext(filename)
        if file_ext.lower() != STF_EXTENSION:
            raise ValueError('Extension %s not supported for SimilarityTransform. Must be stored with extension %s' %(file_ext, STF_EXTENSION))

        f = open(filename, 'w')
        f.write('%s\n' %(self._from_frame))
        f.write('%s\n' %(self._to_frame))
        f.write('%f\n' %(self._scale))
        f.write('%f %f %f\n' %(self._translation[0], self._translation[1], self._translation[2]))
        f.write('%f %f %f\n' %(self._rotation[0, 0], self._rotation[0, 1], self._rotation[0, 2]))
        f.write('%f %f %f\n' %(self._rotation[1, 0], self._rotation[1, 1], self._rotation[1, 2]))
        f.write('%f %f %f\n' %(self._rotation[2, 0], self._rotation[2, 1], self._rotation[2, 2]))
        f.close()

    def as_frames(self, from_frame, to_frame='world'):
        return SimilarityTransform(self.rotation, self.translation, self.scale, from_frame, to_frame)

    @staticmethod
    def load(filename):
        """Load a SimilarityTransform from a file.

        The file format is:
        from_frame
        to_frame
        scale
        translation (space separated)
        rotation_row_0 (space separated)
        rotation_row_1 (space separated)
        rotation_row_2 (space separated)

        Parameters
        ----------
        filename : :obj:`str`
            The file to load the transform from.

        Returns
        -------
        :obj:`SimilarityTransform`
            The SimilarityTransform read from the file.

        Raises
        ------
        ValueError
            If filename's extension isn't .stf.
        """
        file_root, file_ext = os.path.splitext(filename)
        if file_ext.lower() != STF_EXTENSION:
            raise ValueError('Extension %s not supported for SimilarityTransform. Can only load extension %s' %(file_ext, STF_EXTENSION))

        f = open(filename, 'r')
        lines = list(f)
        from_frame = lines[0][:-1]
        to_frame = lines[1][:-1]
        s = float(lines[2][:-1])

        t = np.zeros(3)
        t_tokens = lines[3][:-1].split()
        t[0] = float(t_tokens[0])
        t[1] = float(t_tokens[1])
        t[2] = float(t_tokens[2])

        R = np.zeros([3,3])
        r_tokens = lines[4][:-1].split()
        R[0, 0] = float(r_tokens[0])
        R[0, 1] = float(r_tokens[1])
        R[0, 2] = float(r_tokens[2])

        r_tokens = lines[5][:-1].split()
        R[1, 0] = float(r_tokens[0])
        R[1, 1] = float(r_tokens[1])
        R[1, 2] = float(r_tokens[2])

        r_tokens = lines[6][:-1].split()
        R[2, 0] = float(r_tokens[0])
        R[2, 1] = float(r_tokens[1])
        R[2, 2] = float(r_tokens[2])
        f.close()
        return SimilarityTransform(rotation=R, translation=t,
                                   scale=s,
                                   from_frame=from_frame,
                                   to_frame=to_frame)

    def __str__(self):
        out = 'Tra: {0}\n Rot: {1}\n Qtn: {2}\n Scale:{3}\n from {4} to {5}'.format(self.translation, self.rotation, self.scale,
            self.quaternion, self.from_frame, self.to_frame)
        return out

    def __repr__(self):
        out = "SimilarityTransform(rotation=np.{0}, translation=np.{1}, scale={2}, from_frame={3}, to_frame={4})".format(
                                    repr(self.rotation), repr(self.translation), repr(self.scale), repr(self.from_frame), repr(self.to_frame))
        return out
