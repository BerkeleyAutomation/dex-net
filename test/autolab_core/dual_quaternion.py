"""
Class to handle dual quaternions and their interpolations
Implementation details inspired by Ben Kenwright's "A Beginners Guide to Dual-Quaternions"
http://cs.gmu.edu/~jmlien/teaching/cs451/uploads/Main/dual-quaternion.pdf
Author: Jacky Liang
"""
from numbers import Number
import numpy as np

from .transformations import quaternion_multiply, quaternion_conjugate

class DualQuaternion(object):
    """Class for handling dual quaternions and their interpolations.

    Attributes
    ----------
    qr : :obj:`numpy.ndarray` of float
        A 4-entry quaternion in wxyz format.

    qd : :obj:`numpy.ndarray` of float
        A 4-entry quaternion in wxyz format.

    conjugate : :obj:`DualQuaternion`
        The conjugate of this DualQuaternion.

    norm : :obj:`tuple` of :obj:`numpy.ndarray`
        The normalized vectors for qr and qd, respectively.

    normalized : :obj:`DualQuaternion`
        This quaternion with qr normalized.
    """

    def __init__(self, qr=[1,0,0,0], qd=[0,0,0,0], enforce_unit_norm=True):
        """Initialize a dual quaternion.

        Parameters
        ----------
        qr : :obj:`numpy.ndarray` of float
            A 4-entry quaternion in wxyz format.

        qd : :obj:`numpy.ndarray` of float
            A 4-entry quaternion in wxyz format.

        enforce_unit_norm : bool
            If true, raises a ValueError when the quaternion is not normalized.

        Raises
        ------
        ValueError
            If enforce_unit_norm is True and the norm of qr is not 1.
        """
        self.qr = qr
        self.qd = qd

        if enforce_unit_norm:
            norm = self.norm
            if not np.allclose(norm[0], [1]):
                raise ValueError("Dual quaternion does not have norm 1! Got {0}".format(norm[0]))

    @property
    def qr(self):
        """:obj:`numpy.ndarray` of float: A 4-entry quaternion in wxyz format.
        """
        qr_wxyz = np.roll(self._qr, 1)
        return qr_wxyz

    @qr.setter
    def qr(self, qr_wxyz):
        qr_wxyz = np.array([n for n in qr_wxyz])
        qr_xyzw = np.roll(qr_wxyz, -1)
        self._qr = qr_xyzw

    @property
    def qd(self):
        """:obj:`numpy.ndarray` of float: A 4-entry quaternion in wxyz format.
        """
        qd_wxyz = np.roll(self._qd, 1)
        return qd_wxyz

    @qd.setter
    def qd(self, qd_wxyz):
        qd_wxyz = np.array([n for n in qd_wxyz])
        if qd_wxyz[0] != 0:
            raise ValueError('Invalid dual quaternion! First value of Qd must be 0. Got {0}'.format(qd))
        qd_xyzw = np.roll(qd_wxyz, -1)
        self._qd = qd_xyzw

    @property
    def conjugate(self):
        """:obj:`DualQuaternion`: The conjugate of this quaternion.
        """
        qr_c_xyzw = quaternion_conjugate(self._qr)
        qd_c_xyzw = quaternion_conjugate(self._qd)

        qr_c_wxyz = np.roll(qr_c_xyzw, 1)
        qd_c_wxyz = np.roll(qd_c_xyzw, 1)
        return DualQuaternion(qr_c_wxyz, qd_c_wxyz)

    @property
    def norm(self):
        """:obj:`tuple` of :obj:`numpy.ndarray`: The normalized vectors for qr and qd, respectively.
        """
        qr_c = quaternion_conjugate(self._qr)
        qd_c = quaternion_conjugate(self._qd)

        qr_norm = np.linalg.norm(quaternion_multiply(self._qr, qr_c))
        qd_norm = np.linalg.norm(quaternion_multiply(self._qr, qd_c) + quaternion_multiply(self._qd, qr_c))

        return (qr_norm, qd_norm)

    @property
    def normalized(self):
        """:obj:`DualQuaternion`: This quaternion with qr normalized.
        """
        qr = self.qr /1./ np.linalg.norm(self.qr)
        return DualQuaternion(qr, self.qd, True)

    def copy(self):
        """Return a copy of this quaternion.

        Returns
        -------
        :obj:`DualQuaternion`
            The copied DualQuaternion.
        """
        return DualQuaternion(self.qr.copy(), self.qd.copy())

    @staticmethod
    def interpolate(dq0, dq1, t):
        """Return the interpolation of two DualQuaternions.

        This uses the Dual Quaternion Linear Blending Method as described by Matthew Smith's
        'Applications of Dual Quaternions in Three Dimensional Transformation and Interpolation'
        https://www.cosc.canterbury.ac.nz/research/reports/HonsReps/2013/hons_1305.pdf

        Parameters
        ----------
        dq0 : :obj:`DualQuaternion`
            The first DualQuaternion.

        dq1 : :obj:`DualQuaternion`
            The second DualQuaternion.

        t : float
            The interpolation step in [0,1]. When t=0, this returns dq0, and
            when t=1, this returns dq1.

        Returns
        -------
        :obj:`DualQuaternion`
            The interpolated DualQuaternion.

        Raises
        ------
        ValueError
            If t isn't in [0,1].
        """
        if not 0 <= t <= 1:
            raise ValueError("Interpolation step must be between 0 and 1! Got {0}".format(t))

        dqt = dq0 * (1-t) + dq1 * t
        return dqt.normalized

    def __mul__(self, val):
        """Multiplies the dual quaternion by another dual quaternion or a
        scalar.

        Parameters
        ----------
        val : :obj:`DualQuaternion` or number
            The value by which to multiply this dual quaternion.

        Returns
        -------
        :obj:`DualQuaternion`
            A new DualQuaternion that results from the multiplication.

        Raises
        ------
        ValueError
            If val is not a DualQuaternion or Number.
        """
        if isinstance(val, DualQuaternion):
            new_qr_xyzw = quaternion_multiply(self._qr, val._qr)
            new_qd_xyzw = quaternion_multiply(self._qr, val._qd) + quaternion_multiply(self._qd, val._qr)

            new_qr_wxyz = np.roll(new_qr_xyzw, 1)
            new_qd_wxyz = np.roll(new_qd_xyzw, 1)

            return DualQuaternion(new_qr_wxyz, new_qd_wxyz)
        elif isinstance(val, Number):
            new_qr_wxyz = val * self.qr
            new_qd_wxyz = val * self.qd

            return DualQuaternion(new_qr_wxyz, new_qd_wxyz, False)

        raise ValueError('Cannot multiply dual quaternion with object of type {0}'.format(type(val)))

    def __add__(self, val):
        """Adds the dual quaternion to another dual quaternion.

        Parameters
        ----------
        val : :obj:`DualQuaternion`
            The DualQuaternion to add to this one.

        Returns
        -------
        :obj:`DualQuaternion`
            A new DualQuaternion that results from the addition..

        Raises
        ------
        ValueError
            If val is not a DualQuaternion.
        """
        if not isinstance(val, DualQuaternion):
            raise ValueError('Cannot add dual quaternion with object of type {0}'.format(type(val)))

        new_qr_wxyz = self.qr + val.qr
        new_qd_wxyz = self.qd + val.qd
        new_qr_wxyz = new_qr_wxyz / np.linalg.norm(new_qr_wxyz)

        return DualQuaternion(new_qr_wxyz, new_qd_wxyz, False)

    def __str__(self):
        return '{0}+{1}e'.format(self.qr, self.qd)

    def __repr__(self):
        return 'DualQuaternion({0},{1})'.format(repr(self.qr), repr(self.qd))

