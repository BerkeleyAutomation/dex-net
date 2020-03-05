# -*- coding: utf-8 -*-
"""
Copyright ©2017. The Regents of the University of California (Regents). All Rights Reserved.
Permission to use, copy, modify, and distribute this software and its documentation for educational,
research, and not-for-profit purposes, without fee and without a signed licensing agreement, is
hereby granted, provided that the above copyright notice, this paragraph and the following two
paragraphs appear in all copies, modifications, and distributions. Contact The Office of Technology
Licensing, UC Berkeley, 2150 Shattuck Avenue, Suite 510, Berkeley, CA 94720-1620, (510) 643-
7201, otl@berkeley.edu, http://ipira.berkeley.edu/industry-info for commercial licensing opportunities.

IN NO EVENT SHALL REGENTS BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL,
INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF
THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF REGENTS HAS BEEN
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

REGENTS SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED
HEREUNDER IS PROVIDED "AS IS". REGENTS HAS NO OBLIGATION TO PROVIDE
MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
"""
"""
Classes for efficiently storing datasets containing several attributes of different types.
Designed for saving TensorFlow training datasets.
Author: Jeff Mahler
"""
import json
import logging
import numpy as np
import os
import shutil
import sys

from .constants import *
from .utils import *
from . import YamlConfig

TENSOR_EXT = '.npy'
COMPRESSED_TENSOR_EXT = '.npz'

class Tensor(object):
    """ Abstraction for 4-D tensor objects with a fixed allocation size. 
    The data structure can only be modified by appending a datapoint
    or removing the last datapoint, but can be read from any index at any time.
    """
    def __init__(self, shape, dtype=np.float32, data=None):
        self.cur_index = 0
        self.iter_index = 0
        self.dtype = dtype
        self.data = np.zeros(shape).astype(dtype)
        if data is not None:
            self.add_batch(data)

    @property
    def arr(self):
        return self.data[:self.cur_index,...]

    @property
    def size(self):
        return self.cur_index
    
    @property
    def shape(self):
        return self.data.shape

    @property
    def num_datapoints(self):
        return self.data.shape[0]

    @property
    def height(self):
        if len(self.data.shape) > 1:
            return self.data.shape[1]
        return None

    @property
    def width(self):
        if len(self.data.shape) > 2:
            return self.data.shape[2]
        return None

    @property
    def channels(self):
        if len(self.data.shape) > 3:
            return self.data.shape[3]
        return None

    @property
    def is_full(self):
        return self.cur_index == self.num_datapoints

    @property
    def has_data(self):
        return self.cur_index > 0

    def __getitem__(self, i):
        return self.datapoint(i)

    def __setitem__(self, i, data):
        if ind >= self.size:
            raise ValueError('Index %d out of bounds! Tensor has size %d' %(i, self.size))
        return self.set_datapoint(i, data)

    def __iter__(self):
        self.iter_index = 0
        return self

    def __next__(self):
        if self.iter_index >= self.size:
            raise StopIteration
        self.iter_index += 1
        return self.datapoint(self.iter_index-1)
    
    def next(self):
        return self.__next__()

    def reset(self):
        """ Resets the current index. """
        self.cur_index = 0

    def add(self, datapoint):
        """ Adds the datapoint to the tensor if room is available. """
        if not self.is_full:
            self.set_datapoint(self.cur_index, datapoint)
            self.cur_index += 1

    def add_batch(self, datapoints):
        """ Adds a batch of datapoints to the tensor if room is available. """
        num_datapoints_to_add = datapoints.shape[0]
        end_index = self.cur_index + num_datapoints_to_add
        if end_index <= self.num_datapoints:
            self.data[self.cur_index:end_index,...] = datapoints
            self.cur_index = end_index

    def delete_last(self):
        """ Removes the last datapoint. """
        if self.cur_index == 0:
            raise ValueError('Cannot delete datapoint from empty tensor!')
        self.cur_index -= 1
            
    def datapoint(self, ind):
        """ Returns the datapoint at the given index. """
        if self.height is None:
            return self.data[ind]
        return self.data[ind, ...].copy()

    def set_datapoint(self, ind, datapoint):
        """ Sets the value of the datapoint at the given index. """
        if ind >= self.num_datapoints:
            raise ValueError('Index %d out of bounds! Tensor has %d datapoints' %(ind, self.num_datapoints))
        self.data[ind, ...] = np.array(datapoint).astype(self.dtype)
            
    def data_slice(self, slice_ind):
        """ Returns a slice of datapoints """
        if self.height is None:
            return self.data[slice_ind]
        return self.data[slice_ind, ...]

    def save(self, filename, compressed=True):
        """ Save a tensor to disk. """
        # check for data
        if not self.has_data:
            return False

        # read ext and save accordingly
        _, file_ext = os.path.splitext(filename)
        if compressed:
            if file_ext != COMPRESSED_TENSOR_EXT:
                raise ValueError('Can only save compressed tensor with %s extension' %(COMPRESSED_TENSOR_EXT))
            np.savez_compressed(filename,
                                self.data[:self.cur_index,...])
        else:
            if file_ext != TENSOR_EXT:
                raise ValueError('Can only save tensor with .npy extension')
            np.save(filename, self.data[:self.cur_index,...])
        return True

    @staticmethod
    def load(filename, compressed=True, prealloc=None):
        """ Loads a tensor from disk. """
        # switch load based on file ext
        _, file_ext = os.path.splitext(filename)
        if compressed:
            if file_ext != COMPRESSED_TENSOR_EXT:
                raise ValueError('Can only load compressed tensor with %s extension' %(COMPRESSED_TENSOR_EXT))
            data = np.load(filename)['arr_0']
        else:
            if file_ext != TENSOR_EXT:
                raise ValueError('Can only load tensor with .npy extension')
            data = np.load(filename)
            
        # fill prealloc tensor
        if prealloc is not None:
            prealloc.reset()
            prealloc.add_batch(data)
            return prealloc
            
        # init new tensor
        tensor = Tensor(data.shape, data.dtype, data=data)
        return tensor

class TensorDatapoint(dict):
    """ A single tensor datapoint.
    Basically acts like a dictionary.
    """
    def __init__(self, field_names, *args, **kwargs):
        dict.__init__(self, *args, **kwargs)
        for field_name in field_names:
            self[field_name] = None

    @property
    def field_names(self):
        return list(self.keys())
            
class TensorDataset(object):
    """ A class for efficient storage and access of datasets containing datapoints
    with multiple attributes of different types (e.g. images and robot gripper poses).
    The dataset can only be modified by appending a datapoint
    or removing the last datapoint, but can be read from any index at any time.

    Under the hood, this class saves individual attributes in chunks as compressed NumPy files.
    Thus, reads are most efficient when performed in order rather than randomly, to prevent
    expensive I/O to read a single datapoint.
    """
    def __init__(self, filename, config, access_mode=WRITE_ACCESS, force_overwrite=False):
        # read params
        self._filename = filename
        self._config = config
        self._metadata = {}
        self._datapoints_per_file = config['datapoints_per_file']
        self._access_mode = access_mode

        # open dataset folder
        # create dataset if necessary
        if not os.path.exists(self._filename) and access_mode != READ_ONLY_ACCESS:
            os.mkdir(self._filename)
        # throw error if dataset doesn't exist
        elif not os.path.exists(self._filename) and access_mode == READ_ONLY_ACCESS:
            raise ValueError('Dataset %s does not exist!' %(self._filename))
        # check dataset empty
        elif access_mode == WRITE_ACCESS and os.path.exists(self._filename) and len(os.listdir(self._filename)) > 0:
            if not force_overwrite:
                human_input = keyboard_input('Dataset %s exists. Overwrite?' %(self._filename), yesno=True)
                if human_input.lower() == 'n':
                    raise ValueError('User opted not to overwrite dataset')
                
            # delete the old dataset
            shutil.rmtree(self._filename)
            os.mkdir(self._filename)
            
        # save config to location
        if access_mode == WRITE_ACCESS:
            config_filename = os.path.join(self._filename, 'config.json')
            json.dump(self._config, open(config_filename, 'w'),
                      indent=JSON_INDENT,
                      sort_keys=True)

        # init data storage
        self._has_unsaved_data = False
        self._allocate_tensors()

        # init tensor cache
        self._tensor_cache_file_num = {}
        for field_name in self.field_names:
            self._tensor_cache_file_num[field_name] = None

        # init index maps
        self._index_to_file_num = {}
        self._file_num_to_indices = {}
            
        # init state variables
        if access_mode == WRITE_ACCESS:
            # init no files
            self._num_tensors = 0
            self._num_datapoints = 0
            if not os.path.exists(self.tensor_dir):
                os.mkdir(self.tensor_dir)
            if not os.path.exists(self.split_dir):
                os.mkdir(self.split_dir)
        else:
            # read the metadata
            self._metadata = {}
            if os.path.exists(self.metadata_filename):
                self._metadata = json.load(open(self.metadata_filename, 'r'))
            
            # read the number of tensor files
            tensor_dir = self.tensor_dir
            tensor_filenames = filenames(tensor_dir, tag=COMPRESSED_TENSOR_EXT, sorted=True)
            pruned_tensor_filenames = []
            for filename in tensor_filenames:
                try:
                    file_num = int(filename[-9:-4])
                    pruned_tensor_filenames.append(filename)
                except:
                    pass
            tensor_filenames = pruned_tensor_filenames    
            file_nums = np.array([int(filename[-9:-4]) for filename in tensor_filenames])
            if len(file_nums) > 0:
                self._num_tensors = np.max(file_nums)+1
            else:
                self._num_tensors = 0

            # compute the number of datapoints
            self._num_datapoints_last_file = 0
            self._num_datapoints = 0
            if file_nums.shape[0] > 0:
                last_tensor_ind = np.where(file_nums == self._num_tensors-1)[0][0]
                found_last_file = False
                while self._num_tensors >= 0 and not found_last_file:
                    try:
                        last_tensor_data = np.load(tensor_filenames[last_tensor_ind])['arr_0']
                        found_last_file = True
                    except:
                        found_last_file = False
                        self._num_tensors -= 1
                        last_tensor_ind = np.where(file_nums == self._num_tensors-1)[0][0]
                        
                self._num_datapoints_last_file = last_tensor_data.shape[0]
                self._num_datapoints = self._datapoints_per_file * (self._num_tensors-1) + self._num_datapoints_last_file
                
            # set file index
            cur_file_num = 0
            start_datapoint_index = 0

            # set mapping from file num to datapoint indices
            self._file_num_to_indices[cur_file_num] = np.arange(self._datapoints_per_file) + start_datapoint_index

            for ind in range(self._num_datapoints):
                # update to the next file
                if ind > 0 and ind % self._datapoints_per_file == 0:
                    cur_file_num += 1
                    start_datapoint_index += self._datapoints_per_file

                    # set mapping from file num to datapoint indices
                    if cur_file_num < self._num_tensors-1:
                        self._file_num_to_indices[cur_file_num] = np.arange(self._datapoints_per_file) + start_datapoint_index
                    else:
                        self._file_num_to_indices[cur_file_num] = np.arange(self._num_datapoints_last_file) + start_datapoint_index

                # set mapping from index to file num
                self._index_to_file_num[ind] = cur_file_num

    @property
    def filename(self):
        return self._filename

    @property
    def config(self):
        return self._config

    @property
    def metadata(self):
        return self._metadata

    @property
    def metadata_filename(self):
        return os.path.join(self._filename, 'metadata.json')
    
    @property
    def num_tensors(self):
        return self._num_tensors

    @property
    def num_datapoints(self):
        return self._num_datapoints

    @property
    def datapoints_per_file(self):
        return self._datapoints_per_file

    @property
    def datapoints_per_tensor(self):
        return self._datapoints_per_file

    @property
    def field_names(self):
        return list(self._tensors.keys())

    @property
    def datapoint_template(self):
        return TensorDatapoint(self.field_names)

    @property
    def datapoint_indices(self):
        """ Returns an array of all dataset indices. """
        return np.arange(self._num_datapoints)

    @property
    def tensors(self):
        """ Returns the tensors dictionary. """
        return self._tensors

    @property
    def tensor_indices(self):
        """ Returns an array of all tensor indices. """
        return np.arange(self._num_tensors)
    
    @property
    def tensor_dir(self):
        """ Return the tensor directory. """
        return os.path.join(self._filename, 'tensors')

    @property
    def split_dir(self):
        """ Return the tensor directory. """
        return os.path.join(self._filename, 'splits')

    @property
    def split_names(self):
        """ Returns the list of all splits
        """
        split_names = os.listdir(self.split_dir)
        return split_names
    
    def datapoint_indices_for_tensor(self, tensor_index):
        """ Returns the indices for all datapoints in the given tensor. """
        if tensor_index >= self._num_tensors:
            raise ValueError('Tensor index %d is greater than the number of tensors (%d)' %(tensor_index, self._num_tensors))
        return self._file_num_to_indices[tensor_index]

    def tensor_index(self, datapoint_index):
        """ Returns the index of the tensor containing the referenced datapoint. """
        if datapoint_index >= self._num_datapoints:
            raise ValueError('Datapoint index %d is greater than the number of datapoints (%d)' %(datapoint_index, self._num_datapoints))
        return self._index_to_file_num[datapoint_index]

    def generate_tensor_filename(self, field_name, file_num, compressed=True):
        """ Generate a filename for a tensor. """
        file_ext = TENSOR_EXT
        if compressed:
            file_ext = COMPRESSED_TENSOR_EXT
        filename = os.path.join(self.filename, 'tensors', '%s_%05d%s' %(field_name, file_num, file_ext))
        return filename

    def train_indices_filename(self, split_name):
        """ Returns the filename for the training indices. """
        return os.path.join(self.split_dir, split_name, 'train_indices.npz')

    def val_indices_filename(self, split_name):
        """ Returns the filename for the validation indices. """
        return os.path.join(self.split_dir, split_name, 'val_indices.npz')

    def split_metadata_filename(self, split_name):
        """ Returns the filename for split metadata. """
        return os.path.join(self.split_dir, split_name, 'metadata.json')

    def has_split(self, split_name):
        """ Checks whether or not the split with the given name exists.
        
        Parameters
        ----------
        split_name : str
            name of the split
        """
        if os.path.exists(os.path.join(self.split_dir, split_name)):
            return True
        return False
    
    def _allocate_tensors(self):
        """ Allocates the tensors in the dataset. """
        # init tensors dict
        self._tensors = {}

        # allocate tensor for each data field
        for field_name, field_spec in self._config['fields'].items():
            # parse attributes
            field_dtype = np.dtype(field_spec['dtype'])
            
            # parse shape
            field_shape = [self._datapoints_per_file]
            if 'height' in field_spec.keys():
                field_shape.append(field_spec['height'])
                if 'width' in field_spec.keys():
                    field_shape.append(field_spec['width'])
                    if 'channels' in field_spec.keys():
                        field_shape.append(field_spec['channels'])
                        
            # create tensor
            self._tensors[field_name] = Tensor(field_shape, field_dtype)

    def add(self, datapoint):
        """ Adds a datapoint to the file. """
        # check access level
        if self._access_mode == READ_ONLY_ACCESS:
            raise ValueError('Cannot add datapoints with read-only access')

        # read tensor datapoint ind
        tensor_ind = self._num_datapoints // self._datapoints_per_file

        # check datapoint fields
        for field_name in datapoint.keys():
            if field_name not in self.field_names:
                raise ValueError('Field %s not specified in dataset' %(field_name))
        
        # store data in tensor
        cur_num_tensors = self._num_tensors
        new_num_tensors = cur_num_tensors
        for field_name in self.field_names:
            if tensor_ind < cur_num_tensors:
                # load tensor if it was previously allocated
                self._tensors[field_name] = self.tensor(field_name, tensor_ind)
            else:
                # clear tensor if this is a new tensor
                self._tensors[field_name].reset()
                self._tensor_cache_file_num[field_name] = tensor_ind
                new_num_tensors = cur_num_tensors + 1
                self._has_unsaved_data = True
            self._tensors[field_name].add(datapoint[field_name])
            cur_size = self._tensors[field_name].size

        # update num tensors
        if new_num_tensors > cur_num_tensors:
            self._num_tensors = new_num_tensors

        # update file indices
        self._index_to_file_num[self._num_datapoints] = tensor_ind
        self._file_num_to_indices[tensor_ind] = tensor_ind * self._datapoints_per_file + np.arange(cur_size)

        # save if tensors are full
        field_name = self.field_names[0]
        if self._tensors[field_name].is_full:
            # save next tensors to file
            logging.info('Dataset %s: Writing tensor %d to disk' %(self.filename, tensor_ind))
            self.write()

        # increment num datapoints
        self._num_datapoints += 1

    def __getitem__(self, ind):
        """ Indexes the dataset for the datapoint at the given index. """
        return self.datapoint(ind)

    def datapoint(self, ind, field_names=None):
        """ Loads a tensor datapoint for a given global index.

        Parameters
        ----------
        ind : int
            global index in the tensor
        field_names : :obj:`list` of str
            field names to load

        Returns
        -------
        :obj:`TensorDatapoint`
            the desired tensor datapoint
        """
        # flush if necessary
        if self._has_unsaved_data:
            self.flush()

        # check valid input
        if ind >= self._num_datapoints:
            raise ValueError('Index %d larger than the number of datapoints in the dataset (%d)' %(ind, self._num_datapoints))

        # load the field names
        if field_names is None:
            field_names = self.field_names
        
        # return the datapoint
        datapoint = TensorDatapoint(field_names)
        file_num = self._index_to_file_num[ind]
        for field_name in field_names:
            tensor = self.tensor(field_name, file_num)
            tensor_index = ind % self._datapoints_per_file
            datapoint[field_name] = tensor.datapoint(tensor_index)
        return datapoint

    def tensor(self, field_name, tensor_ind):
        """ Returns the tensor for a given field and tensor index.

        Parameters
        ----------
        field_name : str
            the name of the field to load
        tensor_index : int
            the index of the tensor

        Returns
        -------
        :obj:`Tensor`
            the desired tensor
        """
        if tensor_ind == self._tensor_cache_file_num[field_name]:
            return self._tensors[field_name]
        filename = self.generate_tensor_filename(field_name, tensor_ind, compressed=True)
        Tensor.load(filename, compressed=True,
                    prealloc=self._tensors[field_name])
        self._tensor_cache_file_num[field_name] = tensor_ind
        return self._tensors[field_name]

    def __iter__(self):
        """ Generate iterator. Not thread safe. """
        self._count = 0
        return self

    def __next__(self):
        """ Read the next datapoint.
        
        Returns
        -------
        :obj:`TensorDatapoint`
            the next datapoint
        """
        # terminate
        if self._count >= self._num_datapoints:
            raise StopIteration

        # init empty datapoint
        datapoint = self.datapoint(self._count)
        self._count += 1
        return datapoint

    def next(self):
        return self.__next__()

    def delete_last(self, num_to_delete=1):
        """ Deletes the last N datapoints from the dataset.

        Parameters
        ----------
        num_to_delete : int
            the number of datapoints to remove from the end of the dataset
        """
        # check access level
        if self._access_mode == READ_ONLY_ACCESS:
            raise ValueError('Cannot delete datapoints with read-only access')

        # check num to delete
        if num_to_delete > self._num_datapoints:
            raise ValueError('Cannot remove more than the number of datapoints in the dataset')

        # compute indices
        last_datapoint_ind = self._num_datapoints - 1
        last_tensor_ind = last_datapoint_ind // self._datapoints_per_file
        new_last_datapoint_ind = self._num_datapoints - 1 - num_to_delete
        new_num_datapoints = new_last_datapoint_ind + 1

        new_last_datapoint_ind = max(new_last_datapoint_ind, 0)
        new_last_tensor_ind = new_last_datapoint_ind // self._datapoints_per_file

        # delete all but the last tensor
        delete_tensor_ind = range(new_last_tensor_ind+1, last_tensor_ind+1) 
        for tensor_ind in delete_tensor_ind:
            for field_name in self.field_names:
                filename = self.generate_tensor_filename(field_name, tensor_ind)
                os.remove(filename)

        # update last tensor
        dataset_empty = False
        target_tensor_size = new_num_datapoints % self._datapoints_per_file
        if target_tensor_size == 0:
            if new_num_datapoints > 0:
                target_tensor_size = self._datapoints_per_file
            else:
                dataset_empty = True

        for field_name in self.field_names:
            new_last_tensor = self.tensor(field_name, new_last_tensor_ind)
            while new_last_tensor.size > target_tensor_size:
                new_last_tensor.delete_last()
            filename = self.generate_tensor_filename(field_name, new_last_tensor_ind)
            new_last_tensor.save(filename, compressed=True)
            if not new_last_tensor.has_data:
                os.remove(filename)
                new_last_tensor.reset()
        
        # update num datapoints            
        if self._num_datapoints - 1 - num_to_delete >= 0:
            self._num_datapoints = new_num_datapoints
        else:
            self._num_datapoints = 0

        # handle deleted tensor
        self._num_tensors = new_last_tensor_ind + 1
        if dataset_empty:
            self._num_tensors = 0
            
    def add_metadata(self, key, value):
        """ Adds metadata (key-value pairs) to the dataset.

        Parameters
        ----------
        key : str
            name for metadata
        value : :obj:`object` must be JSON serializable
            content of metadata
        """
        self._metadata[key] = value

        # write the current metadata to file
        json.dump(self._metadata, open(self.metadata_filename, 'w'),
                  indent=JSON_INDENT,
                  sort_keys=True)
    
    def write(self):
        """ Writes all tensors to the next file number. """
        # write the next file for all fields
        for field_name in self.field_names:
            filename = self.generate_tensor_filename(field_name, self._num_tensors-1)
            self._tensors[field_name].save(filename, compressed=True)

        # write the current metadata to file
        json.dump(self._metadata, open(self.metadata_filename, 'w'),
                  indent=JSON_INDENT,
                  sort_keys=True)

        # update
        self._has_unsaved_data = False
        
    def flush(self):
        """ Flushes the data tensors and saves metadata to disk. """
        self.write()

    @staticmethod
    def open(dataset_dir, access_mode=READ_ONLY_ACCESS):
        """ Opens a tensor dataset. """
        # check access mode
        if access_mode == WRITE_ACCESS:
            raise ValueError('Cannot open a dataset with write-only access')

        # read config
        try:
            # json load
            config_filename = os.path.join(dataset_dir, 'config.json')
            config = json.load(open(config_filename, 'r'))
        except:
            # YAML load
            config_filename = os.path.join(dataset_dir, 'config.yaml')
            config = YamlConfig(config_filename)

        # open dataset
        dataset = TensorDataset(dataset_dir, config, access_mode=access_mode)
        return dataset

    def split(self, split_name):
        """ Return the training and validation indices for the requested split.

        Parameters
        ----------
        split_name : str
            name of the split

        Returns
        -------
        :obj:`numpy.ndarray`
            array of training indices in the global dataset
        :obj:`numpy.ndarray`
            array of validation indices in the global dataset
        dict
            metadata about the split
        """
        if not self.has_split(split_name):
            raise ValueError('Split %s does not exist!' %(split_name))
        metadata_filename = self.split_metadata_filename(split_name)
        train_filename = self.train_indices_filename(split_name)
        val_filename = self.val_indices_filename(split_name)        

        metadata = json.load(open(metadata_filename, 'r'))
        train_indices = np.load(train_filename)['arr_0']
        val_indices = np.load(val_filename)['arr_0']
        return train_indices, val_indices, metadata
        
    def make_split(self, split_name, val_indices=None, train_pct=0.8, field_name=None):
        """ Splits the dataset into train and test according
        to the given attribute.
        The split is saved with the dataset for future access.

        Parameters
        ----------
        split_name : str
            name of the split (for future accesses)
        val_indices : 
            indices of the validation datapoints (overrides other variables if not None) 
        train_pct : float
            percent of data to use for training
        field_name : str
            name of the field to use in splitting (None for raw indices)

        Returns
        -------
        :obj:`numpy.ndarray`
            array of training indices in the global dataset
        :obj:`numpy.ndarray`
            array of validation indices in the global dataset
        """
        # check train percentage
        if train_pct < 0 or train_pct > 1:
            raise ValueError('Train pct must be a float between 0 and 1')

        # check existence
        if self.has_split(split_name):
            raise ValueError('Cannot create split %s - it already exists! To overwrite, delete split with TensorDataset.delete_split(split_name)' %(split_name))

        # perform splitting
        if val_indices is not None:
            all_indices = np.arange(self.num_datapoints)
            train_indices = np.setdiff1d(all_indices, val_indices)
        elif field_name is None:
            # split on indices
            indices = np.arange(self.num_datapoints)
            num_train = int(train_pct * self.num_datapoints)
            np.random.shuffle(indices)
            train_indices = indices[:num_train]
            val_indices = indices[num_train:]
            field_name = 'index'
        elif field_name == 'split':
            # split on binary values
            train_indices = []
            val_indices = []
            for i in range(self.num_datapoints):
                datapoint = self.datapoint(i, field_names=['split'])
                split = datapoint['split']
                if split == TRAIN_ID:
                    train_indices.append(i)
                else:
                    val_indices.append(i)
            train_indices = np.array(train_indices)
            val_indices = np.array(val_indices)
        else:
            # split on field name

            # check valid field
            if field_name not in self.config['fields'].keys():
                raise ValueError('Field %d not in dataset!' %(field_name))
            if 'height' in self.config['fields'][field_name].keys():
                raise ValueError('Can only split on scalar fields!')

            # find unique values
            values = []
            for i in range(self.num_datapoints):
                datapoint = self.datapoint(i, field_names=[field_name])
                values.append(datapoint[field_name])
            unique_values = np.unique(values)
            num_unique = unique_values.shape[0]
            num_train = int(train_pct * num_unique)

            # split
            np.random.shuffle(unique_values)
            train_values = unique_values[:num_train]
            val_values = unique_values[num_train:]

            # aggregate indices
            train_indices = []
            val_indices = []
            for i in range(self.num_datatpoints):
                datapoint = self.datapoint(i, field_names=[field_name])
                if datapoint[field_name] in train_values:
                    train_indices.append(i)
                else:
                    val_indices.append(i)
            train_indices = np.array(train_indices)
            val_indices = np.array(val_indices)                    
                
        # sort indices
        train_indices.sort()
        val_indices.sort()

        # save to disk
        if not os.path.exists(self.split_dir):
            os.mkdir(self.split_dir)
        split_dir = os.path.join(self.split_dir, split_name)
        os.mkdir(split_dir)
        train_filename = self.train_indices_filename(split_name)
        val_filename = self.val_indices_filename(split_name)        
        np.savez_compressed(train_filename, train_indices)
        np.savez_compressed(val_filename, val_indices)
        metadata_filename = self.split_metadata_filename(split_name)
        metadata = {
            'field_name': field_name,
            'train_pct': train_pct
        }
        json.dump(metadata, open(metadata_filename, 'w'),
                  indent=JSON_INDENT,
                  sort_keys=True)
        return train_indices, val_indices

    def delete_split(self, split_name):
        """ Delete a split of the dataset.

        Parameters
        ----------
        split_name : str
            name of the split to delete
        """
        if self.has_split(split_name):
            shutil.rmtree(os.path.join(self.split_dir, split_name))
