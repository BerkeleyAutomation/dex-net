"""
Custom exception types.
Author: Jeff Mahler
"""
class TerminateException(Exception):
    """ Signal to terminate.
    """
    def __init__(self, *args, **kwargs):
         Exception.__init__(self, *args, **kwargs)


