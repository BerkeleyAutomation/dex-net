"""
Utility class for logging.

Author: Vishal Satish
"""
import logging
import sys

import colorlog

ROOT_LOG_LEVEL = logging.INFO
ROOT_LOG_STREAM = sys.stdout

def configure_root():
    """Configure the root logger."""
    root_logger = logging.getLogger()

    # clear any existing handles to streams because we don't want duplicate logs
    # NOTE: we assume that any stream handles we find are to ROOT_LOG_STREAM, which is usually the case(because it is stdout). This is fine because we will be re-creating that handle. Otherwise we might be deleting a handle that won't be re-created, which could result in dropped logs.
    for hdlr in root_logger.handlers:
        if isinstance(hdlr, logging.StreamHandler):
            root_logger.removeHandler(hdlr)

    # configure the root logger
    root_logger.setLevel(ROOT_LOG_LEVEL)
    hdlr = logging.StreamHandler(ROOT_LOG_STREAM) 
    formatter = colorlog.ColoredFormatter(
                        '%(purple)s%(name)-10s %(log_color)s%(levelname)-8s%(reset)s %(white)s%(message)s',
                        reset=True,
                        log_colors={
                            'DEBUG': 'cyan',
                            'INFO': 'green',
                            'WARNING': 'yellow',
                            'ERROR': 'red',
                            'CRITICAL': 'red,bg_white',
                        } 
                                         )   
    hdlr.setFormatter(formatter)
    root_logger.addHandler(hdlr)

def add_root_log_file(log_file):
    """
    Add a log file to the root logger.

    Parameters
    ----------
    log_file :obj:`str`
        The path to the log file.
    """
    root_logger = logging.getLogger()

    # add a file handle to the root logger
    hdlr = logging.FileHandler(log_file)
    formatter = logging.Formatter('%(asctime)s %(name)-10s %(levelname)-8s %(message)s', datefmt='%m-%d %H:%M:%S')
    hdlr.setFormatter(formatter)
    root_logger.addHandler(hdlr)
    root_logger.info('Root logger now logging to {}'.format(log_file))

class Logger(object):
    ROOT_CONFIGURED = False

    @staticmethod
    def reconfigure_root():
        """Reconfigure the root logger."""
        configure_root()

    @staticmethod
    def get_logger(name, log_level=logging.INFO, log_file=None, global_log_file=False, silence=False):
        """
        Build a logger. All logs will be propagated up to the root logger if not silenced. If log_file is provided, logs will be written out to that file. If global_log_file is true, log_file will be handed the root logger, otherwise it will only be used by this particular logger.
    
        Parameters
        ----------
        name :obj:`str`
            The name of the logger to be built.
        log_level : `int`
            The log level. See the python logging module documentation for possible enum values.
        log_file :obj:`str`
            The path to the log file to log to.
        global_log_file :obj:`bool`
            Whether or not to use the given log_file for this particular logger or for the root logger.
        silence :obj:`bool`
            Whether or not to silence this logger. If it is silenced, the only way to get output from this logger is through a non-global log file.

        Returns
        -------
        :obj:`logging.Logger`
            A custom logger.
        """
        no_op = False
        # some checks for silencing/no-op logging
        if silence and global_log_file:
            raise ValueError("You can't silence a logger and log to a global log file!")
        if silence and log_file is None:
            logging.warning('You are creating a no-op logger!')
            no_op = True

        # configure the root logger if it hasn't been already
        if not Logger.ROOT_CONFIGURED:
            configure_root()
        Logger.ROOT_CONFIGURED = True
        
        # build a logger
        logger = logging.getLogger(name)
        logger.setLevel(log_level)

        # silence the logger by preventing it from propagating upwards to the root
        logger.propagate = not silence

        # configure the log file stream 
        if log_file is not None:
            # if the log file is global, add it to the root logger
            if global_log_file:
                add_root_log_file(log_file)
            # otherwise add it to this particular logger
            else:
                hdlr = logging.FileHandler(log_file)
                formatter = logging.Formatter('%(asctime)s %(name)-10s %(levelname)-8s %(message)s', datefmt='%m-%d %H:%M:%S')
                hdlr.setFormatter(formatter)
                logger.addHandler(hdlr)

        # add a no-op handler to suppress warnings about there being no handlers
        if no_op:
            logger.addHandler(logging.NullHandler())
        return logger

    @staticmethod
    def add_log_file(logger, log_file, global_log_file=False):
        """
        Add a log file to this logger. If global_log_file is true, log_file will be handed the root logger, otherwise it will only be used by this particular logger.

        Parameters
        ----------
        logger :obj:`logging.Logger`
            The logger.
        log_file :obj:`str`
            The path to the log file to log to.
        global_log_file :obj:`bool`
            Whether or not to use the given log_file for this particular logger or for the root logger.
        """

        if global_log_file:
            add_root_log_file(log_file)
        else:
            hdlr = logging.FileHandler(log_file)
            formatter = logging.Formatter('%(asctime)s %(name)-10s %(levelname)-8s %(message)s', datefmt='%m-%d %H:%M:%S')
            hdlr.setFormatter(formatter)
            logger.addHandler(hdlr)
