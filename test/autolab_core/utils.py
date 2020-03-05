"""
Commonly used helper functions
Author: Jeff Mahler
"""
import logging
import os
from six.moves import input

import numpy as np

def gen_experiment_id(n=10):
    """Generate a random string with n characters.

    Parameters
    ----------
    n : int
        The length of the string to be generated.

    Returns
    -------
    :obj:`str`
        A string with only alphabetic characters. 
    """
    chrs = 'abcdefghijklmnopqrstuvwxyz'
    inds = np.random.randint(0,len(chrs), size=n)
    return ''.join([chrs[i] for i in inds])

def get_elapsed_time(time_in_seconds):
    """ Helper function to get elapsed time in human-readable format.

    Parameters
    ----------
    time_in_seconds : float
        runtime, in seconds

    Returns
    -------
    str
        formatted human-readable string describing the time
    """
    if time_in_seconds < 60:
        return '%.1f seconds' % (time_in_seconds)
    elif time_in_seconds < 3600:
        return '%.1f minutes' % (time_in_seconds / 60)
    else:
        return '%.1f hours' % (time_in_seconds / 3600)

def mkdir_safe(path):
    """ Creates a directory if it does not already exist.

    Parameters
    ----------
    path : str
        path to the directory to create

    Returns
    -------
    bool
        True if the directory was created, False otherwise
    """
    if not os.path.exists(path):
        os.mkdir(path)
    
def histogram(values, num_bins, bounds, normalized=True, plot=False, color='b'):
    """Generate a histogram plot.

    Parameters
    ----------
    values : :obj:`numpy.ndarray`
        An array of values to put in the histogram.

    num_bins : int
        The number equal-width bins in the histogram.

    bounds : :obj:`tuple` of float
        Two floats - a min and a max - that define the lower and upper
        ranges of the histogram, respectively.

    normalized : bool
        If True, the bins will show the percentage of elements they contain
        rather than raw counts.

    plot : bool
        If True, this function uses pyplot to plot the histogram.

    color : :obj:`str`
        The color identifier for the plotted bins.

    Returns
    -------
    :obj:`tuple of `:obj:`numpy.ndarray`
        The values of the histogram and the bin edges as ndarrays.
    """
    hist, bins = np.histogram(values, bins=num_bins, range=bounds)
    width = (bins[1] - bins[0])
    if normalized:
        if np.sum(hist) > 0:
            hist = hist.astype(np.float32) / np.sum(hist)
    if plot:
        import matplotlib.pyplot as plt
        plt.bar(bins[:-1], hist, width=width, color=color)    
    return hist, bins

def skew(xi):
    """Return the skew-symmetric matrix that can be used to calculate
    cross-products with vector xi.

    Multiplying this matrix by a vector `v` gives the same result
    as `xi x v`.

    Parameters
    ----------
    xi : :obj:`numpy.ndarray` of float
        A 3-entry vector.

    Returns
    -------
    :obj:`numpy.ndarray` of float
        The 3x3 skew-symmetric cross product matrix for the vector.
    """
    S = np.array([[0, -xi[2], xi[1]],
                  [xi[2], 0, -xi[0]],
                  [-xi[1], xi[0], 0]])
    return S

def deskew(S):
    """Converts a skew-symmetric cross-product matrix to its corresponding
    vector. Only works for 3x3 matrices.

    Parameters
    ----------
    S : :obj:`numpy.ndarray` of float
        A 3x3 skew-symmetric matrix.

    Returns
    -------
    :obj:`numpy.ndarray` of float
        A 3-entry vector that corresponds to the given cross product matrix.
    """
    x = np.zeros(3)
    x[0] = S[2,1]
    x[1] = S[0,2]
    x[2] = S[1,0]
    return x

def reverse_dictionary(d):
    """ Reverses the key value pairs for a given dictionary.

    Parameters
    ----------
    d : :obj:`dict`
        dictionary to reverse

    Returns
    -------
    :obj:`dict`
        dictionary with keys and values swapped
    """
    rev_d = {}
    [rev_d.update({v:k}) for k, v in d.items()]
    return rev_d

def pretty_str_time(dt):
    """Get a pretty string for the given datetime object.
    
    Parameters
    ----------
    dt : :obj:`datetime`
        A datetime object to format.
    
    Returns
    -------
    :obj:`str`
        The `datetime` formatted as {year}_{month}_{day}_{hour}_{minute}.
    """
    return "{0}_{1}_{2}_{3}:{4}".format(dt.year, dt.month, dt.day, dt.hour, dt.minute)

def filenames(directory, tag='', sorted=False, recursive=False):
    """ Reads in all filenames from a directory that contain a specified substring.

    Parameters
    ----------
    directory : :obj:`str`
        the directory to read from
    tag : :obj:`str`
        optional tag to match in the filenames
    sorted : bool
        whether or not to sort the filenames
    recursive : bool
        whether or not to search for the files recursively

    Returns
    -------
    :obj:`list` of :obj:`str`
        filenames to read from
    """
    if recursive:
        f = [os.path.join(directory, f) for directory, _, filename in os.walk(directory) for f in filename if f.find(tag) > -1] 
    else:
        f = [os.path.join(directory, f) for f in os.listdir(directory) if f.find(tag) > -1]
    if sorted:
        f.sort()
    return f

def sph2cart(r, az, elev):
    """ Convert spherical to cartesian coordinates.

    Attributes
    ----------
    r : float
        radius
    az : float
        aziumth (angle about z axis)
    elev : float
        elevation from xy plane

    Returns
    -------
    float
        x-coordinate
    float
        y-coordinate
    float
        z-coordinate
    """
    x = r * np.cos(az) * np.sin(elev)
    y = r * np.sin(az) * np.sin(elev)
    z = r * np.cos(elev)
    return x, y, z

def cart2sph(x, y, z):
    """ Convert cartesian to spherical coordinates.

    Attributes
    ----------
    x : float
        x-coordinate
    y : float
        y-coordinate
    z : float
        z-coordinate

    Returns
    -------
    float
        radius
    float
        aziumth
    float
        elevation
    """
    r = np.sqrt(x**2 + y**2 + z**2)
    if x > 0 and y > 0:
        az = np.arctan(y / x)
    elif x > 0 and y < 0:
        az = 2*np.pi - np.arctan(-y / x)
    elif x < 0 and y > 0:
        az = np.pi - np.arctan(-y / x)    
    elif x < 0 and y < 0:
        az = np.pi + np.arctan(y / x)    
    elif x == 0 and y > 0:
        az = np.pi / 2
    elif x == 0 and y < 0:
        az = 3 * np.pi / 2
    elif y == 0 and x > 0:
        az = 0
    elif y == 0 and x < 0:
        az = np.pi
    elev = np.arccos(z / r)
    return r, az, elev

def keyboard_input(message, yesno=False):
    """ Get keyboard input from a human, optionally reasking for valid
    yes or no input.

    Parameters
    ----------
    message : :obj:`str`
        the message to display to the user
    yesno : :obj:`bool`
        whether or not to enforce yes or no inputs
    
    Returns
    -------
    :obj:`str`
        string input by the human
    """
    # add space for readability
    message += ' '

    # add yes or no to message
    if yesno:
        message += '[y/n] '

    # ask human
    human_input = input(message)
    if yesno:
        while human_input.lower() != 'n' and human_input.lower() != 'y':
            logging.info('Did not understand input. Please answer \'y\' or \'n\'')
            human_input = input(message)
    return human_input

def sqrt_ceil(n):
    """ Computes the square root of an number rounded up to the nearest integer. Very useful for plotting.

    Parameters
    ----------
    n : int
        number to sqrt

    Returns
    -------
    int
        the sqrt rounded up to the nearest integer
    """
    return int(np.ceil(np.sqrt(n)))

def is_positive_definite(A):
    """ Checks if a given matrix is positive definite.

    See https://stackoverflow.com/a/16266736 for details.

    Parameters
    ----------
    A : :obj:`numpy.ndarray` of float or int
        The square matrix of interest

    Returns
    -------
    bool
        whether or not A is positive definite
    """
    is_pd = True
    
    try:
        np.linalg.cholesky(A)
    except np.linalg.LinAlgError:
        is_pd = False

    return is_pd

def is_positive_semi_definite(A):
    """ Checks if a given matrix is positive semi definite.

    Parameters
    ----------
    A : :obj:`numpy.ndarray` of float or int
        The square matrix of interest

    Returns
    -------
    bool
        whether or not A is positive semi-definite
    """
    return is_positive_definite(A + np.eye(len(A)) * 1e-20)