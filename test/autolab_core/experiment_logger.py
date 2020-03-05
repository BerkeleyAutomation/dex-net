'''
Class to handle experiment logging.
Authors: Jeff, Jacky
'''
from abc import ABCMeta, abstractmethod
import os
import csv
import shutil
import subprocess
from datetime import datetime
from time import time
import logging

import numpy as np

from .csv_model import CSVModel
from .yaml_config import YamlConfig
from .utils import gen_experiment_id

class ExperimentLogger:
    """Abstract class for experiment logging.

    Experiments are logged to CSV files, which are encapsulated with the
    :obj:`CSVModel` class.
    """
    __metaclass__ = ABCMeta

    _MASTER_RECORD_FILENAME = 'experiment_record.csv'

    def __init__(self, experiment_root_path, experiment_tag='experiment', log_to_file=True, sub_experiment_dirs=True):
        """Initialize an ExperimentLogger.

        Parameters
        ----------
        experiment_root_path : :obj:`str`
            The root directory in which to save experiment files.
        experiment_tag : :obj:`str`
            The tag to use when prefixing new experiments
        log_to_file : bool, optional
            Default: True
            If True will log all logging statements to a log file
        sub_experiment_dirs : bool, optional
            Defautl: True
            If True will make sub directories corresponding to generated experiment name
        """
        self.experiment_root_path = experiment_root_path

        # open the master record
        self.master_record_filepath = os.path.join(self.experiment_root_path, ExperimentLogger._MASTER_RECORD_FILENAME)
        self.master_record = CSVModel.get_or_create(self.master_record_filepath, self.experiment_meta_headers)

        # add new experiment to the master record
        self.id = ExperimentLogger.gen_experiment_ref(experiment_tag)
        self._master_record_uid = self.master_record.insert(self.experiment_meta_data)

        # make experiment output dir
        if sub_experiment_dirs:
            self.experiment_path = os.path.join(self.experiment_root_path, self.id)
            if not os.path.exists(self.experiment_path):
                os.makedirs(self.experiment_path)
        else:
            self.experiment_path = self.experiment_root_path

        if log_to_file:
            # redirect logging statements to a file
            if not sub_experiment_dirs:
                self.log_path = os.path.join(self.experiment_root_path, 'logs')
            else:
                self.log_path = self.experiment_path
            if not os.path.exists(self.log_path):
                os.makedirs(self.log_path)
            experiment_log = os.path.join(self.log_path, '%s.log' %(self.id))
            formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
            hdlr = logging.FileHandler(experiment_log)
            hdlr.setFormatter(formatter)
            logging.getLogger().addHandler(hdlr)

        # internal dir struct
        self._dirs = {}

    @staticmethod
    def gen_experiment_ref(experiment_tag, n=10):
        """ Generate a random string for naming.

        Parameters
        ----------
        experiment_tag : :obj:`str`
            tag to prefix name with
        n : int
            number of random chars to use

        Returns
        -------
        :obj:`str`
            string experiment ref
        """
        experiment_id = gen_experiment_id(n=n)
        return '{0}_{1}'.format(experiment_tag, experiment_id)

    def update_master_record(self, data):
        """Update a row of the experimental master record CSV with the given data.

        Parameters
        ----------
        uid : int
            The UID of the row to update.

        data : :obj:`dict`
            A dictionary mapping keys (header strings) to values, which
            represents the new row.
        """
        self.master_record.update_by_uid(self._master_record_uid, data)

    @abstractmethod
    def experiment_meta_headers(self):
        """Returns list of two-tuples of header names and types of meta information for the experiments

        Returns
        -------
        :obj:`tuple`
            The metadata for this experiment.
        """
        pass

    @abstractmethod
    def experiment_meta_data(self):
        """Returns the dict of header names and value of meta information for the experiments

        Returns
        -------
        :obj:`dict`
            The metadata for this experiment.
        """
        pass

    @property
    def dirs(self):
        return self._dirs.copy()

    def construct_internal_dirs(self, dirs, realize=False):
        cur_dir = self._dirs
        for dir in dirs:
            if dir not in cur_dir:
                cur_dir[dir] = {}
            cur_dir = cur_dir[dir]
        if realize:
            self._realize_dirs(dirs)

    def construct_internal_dirs_group(self, group_dirs):
        for dirs in group_dirs:
            self.construct_internal_dirs(dirs)

    def has_internal_dirs(self, dirs):
        cur_dir = self.dirs
        for dir in dirs:
            if dir not in cur_dir:
                return False
            cur_dir = cur_dir[dir]
        return True

    def dirs_to_path(self, dirs):
        rel_path = '/'.join(dirs)
        abs_path = os.path.join(self.experiment_path, rel_path)
        return abs_path

    def _realize_dirs(self, dirs):
        if not self.has_internal_dirs(dirs):
            raise Exception("Directory has not been constructed internally! {0}".format(dirs))
        abs_path = self.dirs_to_path(dirs)
        if not os.path.exists(abs_path):
            os.makedirs(abs_path)
        return abs_path

    def remove_dirs(self, dirs):
        if not self.has_internal_dirs(dirs):
            raise Exception("Directory has not been construted internally! {0}".format(dirs))

        path = self.dirs_to_path(dirs)
        if os.path.exists(path):
            subprocess.call(['trash', '-r', path])

        # remove the deepest node
        cur_dir = self.dirs
        for dir in dirs[:-1]:
            cur_dir = cur_dir[dir]
        cur_dir.pop(dirs[-1])

        for i in range(len(dirs) - 1):
            cur_dir = self._dirs
            depth = len(dirs) - i - 2
            for j in range(depth):
                cur_dir = cur_dir[dirs[j]]

            dir_to_remove = dirs[depth]
            if not cur_dir[dir_to_remove]:
                cur_dir.pop(dir_to_remove)
            else:
                break

    def copy_to_dir(self, src_file_path, target_dirs):
        abs_path = self._realize_dirs(target_dirs)
        basename = os.path.basename(src_file_path)
        target_file_path = os.path.join(abs_path, basename)

        logging.debug("Copying {0} to {1}".format(src_file_path, target_file_path))
        shutil.copyfile(src_file_path, target_file_path)

    def copy_dirs(self, src_dirs_path, target_dirs):
        if not self.has_internal_dirs(target_dirs):
            raise Exception("Directory has not been constructed internally! {0}".format(target_dirs))

        target_dirs_path = self.dirs_to_path(target_dirs)
        if os.path.exists(target_dirs_path):
            if len(os.listdir(target_dirs_path)) > 0:
                raise Exception("Target path for copying directories is not empty! Got: {0}".format(target_dirs_path))
            else:
                os.rmdir(target_dirs_path)
        shutil.copytree(src_dirs_path, target_dirs_path)

    @staticmethod
    def pretty_str_time(dt):
        return "{0}_{1}_{2}_{3}:{4}".format(dt.year, dt.month, dt.day, dt.hour, dt.minute)
