"""
Module to serialize and deserialize JSON with numpy arrays. Adapted from
http://stackoverflow.com/a/24375113/723090 so that arrays are human-readable.

Author: Brian Hou
"""

import json as _json
import numpy as np

class NumpyEncoder(_json.JSONEncoder):
    """A numpy array to json encoder.
    """

    def default(self, obj):
        """Converts an ndarray into a dictionary for efficient serialization.

        The dict has three keys:
        - dtype : The datatype of the array as a string.
        - shape : The shape of the array as a tuple.
        - __ndarray__ : The data of the array as a list.

        Parameters
        ----------
        obj : :obj:`numpy.ndarray`
            The ndarray to encode.

        Returns
        -------
        :obj:`dict`
            The dictionary serialization of obj.
        
        Raises
        ------
        TypeError
            If obj isn't an ndarray.
        """
        if isinstance(obj, np.ndarray):
            return dict(__ndarray__=obj.tolist(),
                        dtype=str(obj.dtype),
                        shape=obj.shape)
        # Let the base class default method raise the TypeError
        return _json.JSONEncoder(self, obj)

def json_numpy_obj_hook(dct):
    """Decodes a previously encoded numpy ndarray with proper shape and dtype.

    Parameters
    ----------
    dct : :obj:`dict`
        The encoded dictionary.

    Returns
    -------
    :obj:`numpy.ndarray`
        The ndarray that `dct` was encoding.
    """
    if isinstance(dct, dict) and '__ndarray__' in dct:
        data = np.asarray(dct['__ndarray__'], dtype=dct['dtype'])
        return data.reshape(dct['shape'])
    return dct

def dump(*args, **kwargs):
    """Dump a numpy.ndarray to file stream.

    This works exactly like the usual `json.dump()` function,
    but it uses our custom serializer.
    """
    kwargs.update(dict(cls=NumpyEncoder,
                       sort_keys=True,
                       indent=4,
                       separators=(',', ': ')))
    return _json.dump(*args, **kwargs)

def load(*args, **kwargs):
    """Load an numpy.ndarray from a file stream.

    This works exactly like the usual `json.load()` function,
    but it uses our custom deserializer.
    """
    kwargs.update(dict(object_hook=json_numpy_obj_hook))
    return _json.load(*args, **kwargs)
