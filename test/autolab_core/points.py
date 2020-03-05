"""
Common ops on lists / arrays of points
Author: Jeff Mahler
"""
from abc import ABCMeta, abstractmethod
import numbers

import numpy as np

from .primitives import Box

class BagOfPoints(object):
    """The abstract base class for collections of 3D point clouds.
    """
    __metaclass__ = ABCMeta

    def __init__(self, data, frame):
        """Initialize a BagOfPoints.

        Parameters
        ----------
        data : :obj:`numpy.ndarray` of float
            The data with which to initialize the collection. Usually is of
            shape (dim x #elements).
        frame : :obj:`str`
            The reference frame in which the collection of primitives
            resides.

        Raises
        ------
        ValueError
            If data is not a ndarray or frame is not a string.
        """
        if not isinstance(data, np.ndarray):
            raise ValueError('Must initialize bag of points with a numpy ndarray')
        if not isinstance(frame, str) and not isinstance(frame, unicode):
            raise ValueError('Must provide string name of frame of data')

        self._check_valid_data(data)
        self._data = self._preprocess_data(data)
        self._frame = frame

    @abstractmethod
    def _check_valid_data(self, data):
        """Checks that the data is valid for the appropriate class type.
        """
        pass

    def _preprocess_data(self, data):
        """Converts the data array to the preferred dim x #points structure.

        Parameters
        ----------
        data : :obj:`numpy.ndarray` of float
            The data to process.

        Returns
        -------
        :obj:`numpy.ndarray` of float
            The same data array, but reshapes lists to be dim x 1.
        """
        if len(data.shape) == 1: 
            data = data[:,np.newaxis]
        return data

    @property
    def shape(self):
        """:obj:`tuple` of int : The shape of the collection's data matrix.
        """
        return self._data.shape

    @property
    def frame(self):
        """:obj:`str` : The name of the reference frame in which the
        collection's objects reside.
        """
        return self._frame

    @property
    def data(self):
        """:obj:`numpy.ndarray` of float : The collection's data matrix.
        """
        return self._data.squeeze()

    @property
    def dim(self):
        """int : The number of entries in the data array along the first
        dimension. By convention, this is the dimension of the elements (usually
        3D).
        """
        return self._data.shape[0]

    @property
    def num_points(self):
        """int : The number of entries in the data array along the second
        dimenstion. By convention, this is the number of elements in the
        collection.
        """
        return self._data.shape[1]

    def copy(self):
        """Return a copy of the BagOfPoints object.

        Returns
        -------
        :obj:`BagOfPoints`
            An object of the same type as the original, with a copy
            of the data and the same frame.
        """
        return type(self)(self._data.copy(), self._frame)

    def save(self, filename):
        """Saves the collection to a file.

        Parameters
        ----------
        filename : :obj:`str`
            The file to save the collection to.

        Raises
        ------
        ValueError
            If the file extension is not .npy or .npz.
        """
        file_root, file_ext = os.path.splitext(filename)
        if file_ext == '.npy':
            np.save(filename, self._data)
        elif file_ext == '.npz':
            np.savez_compressed(filename, self._data)
        else:
            raise ValueError('Extension %s not supported for point saves.' %(file_ext))

    def load_data(filename):
        """Loads data from a file.

        Parameters
        ----------
        filename : :obj:`str`
            The file to load the collection from.

        Returns
        -------
        :obj:`numpy.ndarray` of float
            The data read from the file.

        Raises
        ------
        ValueError
            If the file extension is not .npy or .npz.
        """
        file_root, file_ext = os.path.splitext(filename)
        data = None
        if file_ext == '.npy':
            data = np.load(filename)
        elif file_ext == '.npz':
            data = np.load(filename)['arr_0']
        else:
            raise ValueError('Extension %s not supported for point reads' %(file_ext))
        return data

    def __getitem__(self, i):
        """Return a single element from the collection.

        Parameters
        ----------
        i : indexing-type (int or slice or list)
            The index of the desired element.

        Returns
        -------
        :obj:`Point` or :obj:`PointCloud`
            The returned element or group.
        """
        if isinstance(i, int):
            if i >= self.num_points:
                raise ValueError('Index %d is out of bounds' %(i))
            return Point(self._data[:,i], self._frame)
        if isinstance(i, list):
            i = np.array(i)
        if isinstance(i, np.ndarray):
            if np.max(i) >= self.num_points:
                raise ValueError('Index %d is out of bounds' %(np.max(i)))
            return PointCloud(self._data[:,i], self._frame)
        if isinstance(i, slice):
            return PointCloud(self._data[:,i], self._frame)
        raise ValueError('Type %s not supported for indexing' %(type(i)))

    def __str__(self):
        return str(self.data)

class BagOfVectors(BagOfPoints):
    """The base class for collections of 3D vectors.
    """
    pass

class Point(BagOfPoints):
    """A single 3D point.
    """

    def __init__(self, data, frame='unspecified'):
        """Initialize a Point.

        Parameters
        ----------
        data : :obj:`numpy.ndarray` of float
            An dim x 1 vector that represents the point's location.
        frame : :obj:`str`
            The reference frame in which the Point resides.
        """
        BagOfPoints.__init__(self, data, frame)

    def _check_valid_data(self, data):
        """Checks that the incoming data is a Nx1 ndarray.

        Parameters
        ----------
        data : :obj:`numpy.ndarray`
            The data to verify.

        Raises
        ------
        ValueError
            If the data is not of the correct shape.
        """
        if len(data.shape) == 2 and data.shape[1] != 1:
            raise ValueError('Can only initialize Point from a single Nx1 array') 

    @property
    def vector(self):
        """:obj:`numpy.ndarray` of float : The point's location relative
        to the origin.
        """
        return self._data.squeeze()

    @property
    def x(self):
        """float : The first element in the point's data array.
        """
        return self.vector[0]

    @property
    def y(self):
        """float : The second element in the point's data array.
        """
        return self.vector[1]

    @property
    def z(self):
        """float : The third element in the point's data array.
        """
        return self.vector[2]

    def __getitem__(self, dim):
        """float: The point value at the given dimension.
        """
        return self.vector[dim]
    
    def __add__(self, other_pt):
        """Add two Points together.

        Parameters
        ----------
        other_pt : :obj:`Point` or :obj:`numpy.ndarray` of float
            The other point to add to this one.

        Returns
        -------
        :obj:`Point`
            The result of adding the two points together.

        Raises
        ------
        ValueError
            If the shape and/or frames of the two points does not match.
        """
        # Handle point adds
        if isinstance(other_pt, Point) and other_pt.dim == self.dim:
            if self._frame != other_pt.frame:
                raise ValueError('Frames must be the same for addition')
            return Point(self.data + other_pt.data, frame=self._frame)
        # Handle numpy adds
        elif isinstance(other_pt, np.ndarray) and other_pt.shape == self.data.shape:
            return Point(self.data + other_pt, frame=self._frame)
        raise ValueError('Can only add to other Point objects or numpy ndarrays of the same dim')

    def __sub__(self, other_pt):
        """Subtract a point from this one.

        Parameters
        ----------
        other_pt : :obj:`Point` or :obj:`numpy.ndarray` of float
            The other point to subtract from this one.

        Returns
        -------
        :obj:`Point`
            The result of the subtraction.

        Raises
        ------
        ValueError
            If the shape and/or frames of the two points does not match.
        """
        return self + -1 * other_pt

    def __mul__(self, mult):
        """Multiply the point by a scalar.

        Parameters
        ----------
        mult : float
            The number by which to multiply the Point.

        Returns
        -------
        :obj:`Point3D`
            A 3D point created by the multiplication.

        Raises
        ------
        ValueError
            If mult is not a scalar value.
        """
        if isinstance(mult, numbers.Number):
            return Point(mult * self._data, self._frame)
        raise ValueError('Type %s not supported. Only scalar multiplication is supported' %(type(mult)))

    def __rmul__(self, mult):
        """Multiply the point by a scalar.

        Parameters
        ----------
        mult : float
            The number by which to multiply the Point.

        Returns
        -------
        :obj:`Point3D`
            A 3D point created by the multiplication.

        Raises
        ------
        ValueError
            If mult is not a scalar value.
        """
        return self.__mul__(mult)

    def __div__(self, div):
        """Divide the point by a scalar.

        Parameters
        ----------
        div : float
            The number by which to divide the Point.

        Returns
        -------
        :obj:`Point3D`
            A 3D point created by the division. 

        Raises
        ------
        ValueError
            If div is not a scalar value.
        """
        if not isinstance(div, numbers.Number):
            raise ValueError('Type %s not supported. Only scalar division is supported' %(type(div)))
        return self.__mul__(1.0 / div)

    @staticmethod
    def open(filename, frame='unspecified'):
        """Create a Point from data saved in a file.

        Parameters
        ----------
        filename : :obj:`str`
            The file to load data from.

        frame : :obj:`str`
            The frame to apply to the created point.

        Returns
        -------
        :obj:`Point`
            A point created from the data in the file.
        """
        data = BagOfPoints.load_data(filename)
        return Point(data, frame)

class Direction(BagOfVectors):
    """A single directional vector.
    """
    def __init__(self, data, frame):
        """Initialize a Direction.

        Parameters
        ----------
        data : :obj:`numpy.ndarray` of float
            An dim x 1 normalized vector.
        frame : :obj:`str`
            The reference frame in which the Direction resides.
        """
        BagOfPoints.__init__(self, data, frame)

    def _check_valid_data(self, data):
        """Checks that the incoming data is a Nx1 ndarray.

        Parameters
        ----------
        data : :obj:`numpy.ndarray`
            The data to verify.

        Raises
        ------
        ValueError
            If the data is not of the correct shape or if the vector is not
            normed.
        """
        if len(data.shape) == 2 and data.shape[1] != 1:
            raise ValueError('Can only initialize Direction from a single Nx1 array') 
        if np.abs(np.linalg.norm(data) - 1.0) > 1e-4:
            raise ValueError('Direction data must have norm=1.0')

    def orthogonal_basis(self):
        """Return an orthogonal basis to this direction.

        Note
        ----
            Only implemented in 3D.

        Returns
        -------
        :obj:`tuple` of :obj:`Direction`
            The pair of normalized Direction vectors that form a basis of
            this direction's orthogonal complement.

        Raises
        ------
        NotImplementedError
            If the vector is not 3D
        """
        if self.dim == 3:
            x_arr = np.array([-self.data[1], self.data[0], 0])
            if np.linalg.norm(x_arr) == 0:
                x_arr = np.array([self.data[2], 0, 0])
            x_arr = x_arr / np.linalg.norm(x_arr)
            y_arr = np.cross(self.data, x_arr)
            return Direction(x_arr, frame=self.frame), Direction(y_arr, frame=self.frame)
        raise NotImplementedError('Orthogonal basis only supported for 3 dimensions')

    @staticmethod
    def open(filename, frame='unspecified'):
        """Create a Direction from data saved in a file.

        Parameters
        ----------
        filename : :obj:`str`
            The file to load data from.

        frame : :obj:`str`
            The frame to apply to the created Direction.

        Returns
        -------
        :obj:`Direction`
            A Direction created from the data in the file.
        """
        data = BagOfPoints.load_data(filename)
        return Direction(data, frame)

class Plane3D(object):
    """A plane in three dimensions.
    """

    def __init__(self, n, x0):
        """Initialize a plane with a normal vector and a point.

        Parameters
        ----------
        n : :obj:`Direction`
            A 3D normal vector to the plane.

        x0 : :obj:`Point`
            A 3D point in the plane.

        Raises
        ------
        ValueError
            If the parameters are of the wrong type or are not of dimension 3.
        """
        if not isinstance(n, Direction) or n.dim != 3:
            raise ValueError('Plane normal must be a 3D direction')
        if not isinstance(x0, Point) or x0.dim != 3:
            raise ValueError('Plane offset must be a 3D point')
        self._n = n
        self._x0 = x0

    def split_points(self, point_cloud):
        """Split a point cloud into two along this plane.

        Parameters
        ----------
        point_cloud : :obj:`PointCloud`
            The PointCloud to divide in two.

        Returns
        -------
        :obj:`tuple` of :obj:`PointCloud`
            Two new PointCloud objects. The first contains points above the
            plane, and the second contains points below the plane.

        Raises
        ------
        ValueError
            If the input is not a PointCloud.
        """
        if not isinstance(point_cloud, PointCloud):
            raise ValueError('Can only split point clouds')
        # compute indices above and below
        above_plane = point_cloud._data - np.tile(self._x0.data, [1, point_cloud.num_points]).T.dot(self._n) > 0
        above_plane = point_cloud.z_coords > 0 & above_plane
        below_plane = point_cloud._data - np.tile(self._x0.data, [1, point_cloud.num_points]).T.dot(self._n) <= 0
        below_plane = point_cloud.z_coords > 0 & below_plane

        # split data
        above_data = point_cloud.data[:, above_plane]
        below_data = point_cloud.data[:, below_plane]
        return PointCloud(above_data, point_cloud.frame), PointCloud(below_data, point_cloud.frame)

class PointCloud(BagOfPoints):
    """A set of points.
    """

    def __init__(self, data, frame='unspecified'):
        """Initialize a PointCloud.

        Parameters
        ----------
        data : :obj:`numpy.ndarray` of float
            An dim x #points array that contains the points in the cloud.
        frame : :obj:`str`
            The reference frame in which the points reside.
        """
        BagOfPoints.__init__(self, data, frame)

    def _check_valid_data(self, data):
        """Checks that the incoming data is a 3 x #elements ndarray.

        Parameters
        ----------
        data : :obj:`numpy.ndarray`
            The data to verify.

        Raises
        ------
        ValueError
            If the data is not of the correct shape or type.
        """
        if data.dtype.type != np.float32 and data.dtype.type != np.float64:
            raise ValueError('Must initialize point clouds with a numpy float ndarray')
        if data.shape[0] != 3:
            raise ValueError('Illegal data array passed to point cloud. Must have 3 coordinates')
        if len(data.shape) > 2:
            raise ValueError('Illegal data array passed to point cloud. Must have 1 or 2 dimensions')

    @property
    def x_coords(self):
        """:obj:`numpy.ndarray` of float : An array containing all x coordinates
        in the cloud.
        """
        return self._data[0,:]

    @property
    def y_coords(self):
        """:obj:`numpy.ndarray` of float : An array containing all y coordinates
        in the cloud.
        """
        return self._data[1,:]

    @property
    def z_coords(self):
        """:obj:`numpy.ndarray` of float : An array containing all z coordinates
        in the cloud.
        """
        return self._data[2,:]

    def mean(self):
        """Returns the average point in the cloud.

        Returns
        -------
        :obj:`Point`
            The mean point in the PointCloud.
        """
        mean_point_data = np.mean(self._data, axis=1)
        return Point(mean_point_data, self._frame)

    def subsample(self, rate, random=False):
        """Returns a subsampled version of the PointCloud.

        Parameters
        ----------
        rate : int
            Only every rate-th element of the PointCloud is returned.

        Returns
        -------
        :obj:`PointCloud`
            A subsampled point cloud with N / rate total samples.

        Raises
        ------
        ValueError
            If rate is not a positive integer.
        """
        if type(rate) != int and rate < 1:
            raise ValueError('Can only subsample with strictly positive integer rate')
        indices = np.arange(self.num_points)
        if random:
            np.random.shuffle(indices)
        subsample_inds = indices[::rate]
        subsampled_data = self._data[:,subsample_inds]
        return PointCloud(subsampled_data, self._frame), subsample_inds

    def box_mask(self, box):
        """Return a PointCloud containing only points within the given Box.

        Parameters
        ----------
        box : :obj:`Box`
            A box whose boundaries are used to filter points.

        Returns
        -------
        :obj:`PointCloud`
            A filtered PointCloud whose points are all in the given box.
        :obj:`numpy.ndarray`
            Array of indices of the segmented points in the original cloud

        Raises
        ------
        ValueError
            If the input is not a box in the same frame as the PointCloud.
        """
        if not isinstance(box, Box):
            raise ValueError('Must provide Box object')
        if box.frame != self.frame:
            raise ValueError('Box must be in same frame as PointCloud')
        all_points = self.data.T
        cond1 = np.all(box.min_pt <= all_points, axis=1)
        cond2 = np.all(all_points <= box.max_pt, axis=1)
        valid_point_indices = np.where(np.logical_and(cond1, cond2))[0]
        valid_points = all_points[valid_point_indices]

        return PointCloud(valid_points.T, self.frame), valid_point_indices

    def best_fit_plane(self):
        """Fits a plane to the point cloud using least squares.

        Returns
        -------
        :obj:`tuple` of :obj:`numpy.ndarray` of float
            A normal vector to and point in the fitted plane.
        """
        X = np.c_[self.x_coords, self.y_coords, np.ones(self.num_points)]
        y = self.z_coords
        A = X.T.dot(X)
        b = X.T.dot(y)
        w = np.linalg.inv(A).dot(b)
        n = np.array([w[0], w[1], -1])
        n = n / np.linalg.norm(n)
        n = Direction(n, self._frame)
        x0 = self.mean()
        return n, x0

    def nonzero_indices(self):
        """ Returns the point indices corresponding to the zero points.
        
        Returns
        -------
        :obj:`numpy.ndarray`
            array of the nonzero indices
        """
        points_of_interest = np.where(self.z_coords != 0.0)[0]
        return points_of_interest
        
    def remove_zero_points(self):
        """Removes points with a zero in the z-axis.

        Note
        ----
        This returns nothing and updates the PointCloud in-place.
        """
        points_of_interest = np.where(self.z_coords != 0.0)[0]
        self._data = self.data[:, points_of_interest]

    def remove_infinite_points(self):
        """Removes infinite points.

        Note
        ----
        This returns nothing and updates the PointCloud in-place.
        """
        points_of_interest = np.where(np.all(np.isfinite(self.data), axis=0))[0]
        self._data = self.data[:, points_of_interest]
        
    def __add__(self, other_pc):
        """Add two PointClouds together element-wise.

        Parameters
        ----------
        other_pc : :obj:`PointCloud`
            The other PointCloud to add to this one.

        Returns
        -------
        :obj:`PointCloud`
            The result of adding the two PointClouds together.

        Raises
        ------
        ValueError
            If the shape and/or frames of the two PointClouds do not match.
        """
        if not isinstance(other_pc, PointCloud) or other_pc.num_points != self.num_points:
            raise ValueError('Can only add to other point clouds of same size')
        if self._frame != other_pc.frame:
            raise ValueError('Frames must be the same for addition')
        return PointCloud(self.data + other_pc.data, frame=self._frame)

    def __sub__(self, other_pc):
        """Subtract one PointCloud from another element-wise.

        Parameters
        ----------
        other_pc : :obj:`PointCloud`
            The other PointCloud to subtract from this one.

        Returns
        -------
        :obj:`PointCloud`
            The result of the subtraction.

        Raises
        ------
        ValueError
            If the shape and/or frames of the two PointClouds do not match.
        """
        return self + -1 * other_pc

    def __mul__(self, mult):
        """Multiply each point in the cloud by a scalar.

        Parameters
        ----------
        mult : float
            The number by which to multiply the PointCloud.

        Returns
        -------
        :obj:`PointCloud`
            A PointCloud created by the multiplication.

        Raises
        ------
        ValueError
            If mult is not a scalar value.
        """
        if isinstance(mult, numbers.Number):
            return PointCloud(mult * self._data, self._frame)
        raise ValueError('Type %s not supported. Only scalar multiplication is supported' %(type(mult)))

    def __rmul__(self, mult):
        """Multiply each point in the cloud by a scalar.

        Parameters
        ----------
        mult : float
            The number by which to multiply the PointCloud.

        Returns
        -------
        :obj:`PointCloud`
            A PointCloud created by the multiplication.

        Raises
        ------
        ValueError
            If mult is not a scalar value.
        """

        return self.__mul__(mult)

    def __div__(self, div):
        """Divide each point in the cloud by a scalar.

        Parameters
        ----------
        div : float
            The number by which to divide the PointCloud.

        Returns
        -------
        :obj:`PointCloud`
            A PointCloud created by the division.

        Raises
        ------
        ValueError
            If div is not a scalar value.
        """
        if not isinstance(div, numbers.Number):
            raise ValueError('Type %s not supported. Only scalar division is supported' %(type(div)))
        return self.__mul__(1.0 / div)

    @staticmethod
    def open(filename, frame='unspecified'):
        """Create a PointCloud from data saved in a file.

        Parameters
        ----------
        filename : :obj:`str`
            The file to load data from.

        frame : :obj:`str`
            The frame to apply to the created PointCloud.

        Returns
        -------
        :obj:`PointCloud`
            A PointCloud created from the data in the file.
        """
        data = BagOfPoints.load_data(filename)
        return PointCloud(data, frame)

class NormalCloud(BagOfVectors):
    """A set of normal vectors.
    """
    def __init__(self, data, frame='unspecified'):
        """Initialize a NormalCloud.

        Parameters
        ----------
        data : :obj:`numpy.ndarray` of float
            An dim x #points array that contains the normal vectors in the cloud.
            All of these vectors should be normalized.
        frame : :obj:`str`
            The reference frame in which the vectors reside.
        """
        BagOfPoints.__init__(self, data, frame)

    def _check_valid_data(self, data):
        """Checks that the incoming data is a 3 x #elements ndarray of normal
        vectors.

        Parameters
        ----------
        data : :obj:`numpy.ndarray`
            The data to verify.

        Raises
        ------
        ValueError
            If the data is not of the correct shape or type, or if the vectors
            therein are not normalized.
        """
        if data.dtype.type != np.float32 and data.dtype.type != np.float64:
            raise ValueError('Must initialize normals clouds with a numpy float ndarray')
        if data.shape[0] != 3:
            raise ValueError('Illegal data array passed to normal cloud. Must have 3 coordinates')
        if len(data.shape) > 2:
            raise ValueError('Illegal data array passed to normal cloud. Must have 1 or 2 dimensions')
        if np.any((np.abs(np.linalg.norm(data, axis=0) - 1) > 1e-4) & (np.linalg.norm(data, axis=0) != 0)):
            raise ValueError('Illegal data array passed to normal cloud. Must have norm=1.0 or norm=0.0')

    @property
    def x_coords(self):
        """:obj:`numpy.ndarray` of float : An array containing all x coordinates
        in the cloud.
        """
        return self._data[0,:]

    @property
    def y_coords(self):
        """:obj:`numpy.ndarray` of float : An array containing all y coordinates
        in the cloud.
        """
        return self._data[1,:]

    @property
    def z_coords(self):
        """:obj:`numpy.ndarray` of float : An array containing all z coordinates
        in the cloud.
        """
        return self._data[2,:]

    def subsample(self, rate):
        """Returns a subsampled version of the NormalCloud.

        Parameters
        ----------
        rate : int
            Only every rate-th element of the NormalCloud is returned.

        Returns
        -------
        :obj:`RateCloud`
            A subsampled point cloud with N / rate total samples.

        Raises
        ------
        ValueError
            If rate is not a positive integer.
        """
        if type(rate) != int and rate < 1:
            raise ValueError('Can only subsample with strictly positive integer rate')
        subsample_inds = np.arange(self.num_points)[::rate]
        subsampled_data = self._data[:,subsample_inds]
        return NormalCloud(subsampled_data, self._frame)

    def remove_zero_normals(self):
        """Removes normal vectors with a zero magnitude.

        Note
        ----
        This returns nothing and updates the NormalCloud in-place.
        """
        points_of_interest = np.where(np.linalg.norm(self._data, axis=0) != 0.0)[0]
        self._data = self._data[:, points_of_interest]

    def remove_nan_normals(self):
        """Removes normal vectors with nan magnitude.

        Note
        ----
        This returns nothing and updates the NormalCloud in-place.
        """
        points_of_interest = np.where(np.isfinite(np.linalg.norm(self._data, axis=0)))[0]
        self._data = self._data[:, points_of_interest]

    @staticmethod
    def open(filename, frame='unspecified'):
        """Create a NormalCloud from data saved in a file.

        Parameters
        ----------
        filename : :obj:`str`
            The file to load data from.

        frame : :obj:`str`
            The frame to apply to the created NormalCloud.

        Returns
        -------
        :obj:`NormalCloud`
            A NormalCloud created from the data in the file.
        """
        data = BagOfPoints.load_data(filename)
        return NormalCloud(data, frame)

class ImageCoords(BagOfPoints):
    """A set of 2D image coordinates.
    """

    def __init__(self, data, frame):
        """Initialize a set of image coodinates.

        Parameters
        ----------
        data : :obj:`numpy.ndarray` of int
            An 2 x #coords array that contains the image coordinates.
        frame : :obj:`str`
            The reference frame in which the points reside.
        """
        BagOfPoints.__init__(self, data, frame)

    def _check_valid_data(self, data):
        """Checks that the incoming data is a 2 x #elements ndarray of ints.

        Parameters
        ----------
        data : :obj:`numpy.ndarray`
            The data to verify.

        Raises
        ------
        ValueError
            If the data is not of the correct shape or type.
        """
        if data.dtype.type != np.int8 and data.dtype.type != np.int16 \
                and data.dtype.type != np.int32 and data.dtype.type != np.int64 \
                and data.dtype.type != np.uint8 and data.dtype.type != np.uint16 \
                and data.dtype.type != np.uint32 and data.dtype.type != np.uint64:
            raise ValueError('Must initialize image coords with a numpy int ndarray')
        if data.shape[0] != 2:
            raise ValueError('Illegal data array passed to image coords. Must have 2 coordinates')
        if len(data.shape) > 2:
            raise ValueError('Illegal data array passed to point cloud. Must have 1 or 2 dimensions')

    @property
    def i_coords(self):
        """:obj:`numpy.ndarray` of float : The set of i-coordinates
        (those in the second row of the data matrix).
        """
        return self._data[1,:]

    @property
    def j_coords(self):
        """:obj:`numpy.ndarray` of float : The set of j-coordinates
        (those in the first row of the data matrix).
        """
        return self._data[0,:]

    @staticmethod
    def open(filename, frame='unspecified'):
        """Create an ImageCoords from data saved in a file.

        Parameters
        ----------
        filename : :obj:`str`
            The file to load data from.

        frame : :obj:`str`
            The frame to apply to the created ImageCoords.

        Returns
        -------
        :obj:`ImageCoords`
            An ImageCoords created from the data in the file.
        """
        data = BagOfPoints.load_data(filename)
        return ImageCoords(data, frame)

class RgbCloud(BagOfPoints):
    """A set of colors.
    """

    def __init__(self, data, frame):
        """Initialize an RgbCloud.

        Parameters
        ----------
        data : :obj:`numpy.ndarray` of uint8
            An 3 x #elements array that contains the colors in the cloud.
            Elements each have a red, green, and blue coordinate.
        frame : :obj:`str`
            The reference frame in which the points reside.
        """
        BagOfPoints.__init__(self, data, frame)

    def _check_valid_data(self, data):
        """Checks that the incoming data is a 3 x #elements ndarray.

        Parameters
        ----------
        data : :obj:`numpy.ndarray`
            The data to verify.

        Raises
        ------
        ValueError
            If the data is not of the correct shape or type.
        """
        if data.dtype.type != np.uint8:
            raise ValueError('Must initialize rgb clouds with a numpy uint ndarray')
        if data.shape[0] != 3:
            raise ValueError('Illegal data array passed to rgb cloud. Must have 3 coordinates')
        if len(data.shape) > 2:
            raise ValueError('Illegal data array passed to rgb  cloud. Must have 1 or 2 dimensions')

    @property
    def red(self):
        """:obj:`numpy.ndarray` of uint8 : An array containing all red values
        in the cloud.
        """
        return self._data[0,:]

    @property
    def green(self):
        """:obj:`numpy.ndarray` of uint8 : An array containing all green values
        in the cloud.
        """
        return self._data[1,:]

    @property
    def blue(self):
        """:obj:`numpy.ndarray` of uint8 : An array containing all blue values
        in the cloud.
        """
        return self._data[2,:]

    @staticmethod
    def open(filename, frame='unspecified'):
        """Create a RgbCloud from data saved in a file.

        Parameters
        ----------
        filename : :obj:`str`
            The file to load data from.

        frame : :obj:`str`
            The frame to apply to the created RgbCloud.

        Returns
        -------
        :obj:`RgbCloud`
            A RgdCloud created from the data in the file.
        """
        data = BagOfPoints.load_data(filename)
        return RgbCloud(data, frame)

class RgbPointCloud(object):
    """A combined set of 3D points and RGB colors.
    """

    def __init__(point_data, rgb_data, frame):
        """Initialize a PointCloud + RgbCloud combination.

        Parameters
        ----------
        point_data : :obj:`numpy.ndarray` of float
            An dim x #elements array that contains the points in the cloud.
        rgb_data : :obj:`numpy.ndarray` of uint8
            A 3 x #elements array that contains the colors of the cloud.
        frame : :obj:`str`
            The reference frame in which the points reside.
        """
        self.point_cloud = PointCloud(point_data, frame)
        self.rgb_cloud = RgbCloud(rgb_data, frame)

    def __getitem__(self, i):
        """Returns the ith point and color.

        Returns
        -------
        :obj:`tuple` of :obj:`numpy.ndarray` of float and uint8
            The ith point and the ith color.
        """
        return self.point_cloud[i], self.rgb_cloud[i]

class PointNormalCloud(object):
    """A combined set of 3D points and normal vectors.
    """

    def __init__(self, point_data, normal_data, frame):
        """Initialize a PointCloud + NormalCloud combination.

        Parameters
        ----------
        point_data : :obj:`numpy.ndarray` of float
            An dim x #elements array that contains the points in the cloud.
        normal_data : :obj:`numpy.ndarray` of uint8
            A dim x #elements array that contains the normals of the cloud.
        frame : :obj:`str`
            The reference frame in which the points reside.

        Raises
        ------
        ValueError
            If the two datasets don't have the same number of elements.
        """
        self.point_cloud = PointCloud(point_data, frame)
        self.normal_cloud = NormalCloud(normal_data, frame)
        if self.point_cloud.num_points != self.normal_cloud.num_points:
            raise ValueError('PointCloud and NormalCloud must have the same number of points')

    @property
    def points(self):
        """:obj:`PointCloud` : The PointCloud in this set.
        """
        return self.point_cloud

    @property
    def normals(self):
        """:obj:`NormalCloud` : The NormalCloud in this set.
        """
        return self.normal_cloud

    @property
    def num_points(self):
        """int : The number of elements in the clouds.
        """
        return self.point_cloud.num_points

    @property
    def frame(self):
        """:obj:`str` : The frame in which these clouds exist.
        """
        return self.point_cloud.frame

    def __getitem__(self, i):
        """Returns the ith point and normals.

        Returns
        -------
        :obj:`tuple` of :obj:`numpy.ndarray` of float 
            The ith point and the ith normal.
        """
        return self.point_cloud[i], self.normal_cloud[i]

    def remove_zero_points(self):
        """Remove all elements where the norms and points are zero.

        Note
        ----
        This returns nothing and updates the NormalCloud in-place.
        """
        points_of_interest = np.where((np.linalg.norm(self.point_cloud.data, axis=0) != 0.0)  &
                                      (np.linalg.norm(self.normal_cloud.data, axis=0) != 0.0) &
                                      (np.isfinite(self.normal_cloud.data[0,:])))[0]
        self.point_cloud._data = self.point_cloud.data[:, points_of_interest]
        self.normal_cloud._data = self.normal_cloud.data[:, points_of_interest]

