"""

"""

import sys
import dexnet
from autolab_core import YamlConfig
import os
import time
import yaml

# Multiprocess
import multiprocessing as mp
from functools import partial

# Default database file path
DEFAULT_DB_PATH = 'ws_kit.db.hdf5'
#DEFAULT_DB_PATH = 'ws_random_coll.db.hdf5'
# Default database configuration file path
DEFAULT_CFG = 'cfg/apps/custom_database.yaml'
# Default object directory
DEFAULT_OBJ_PATH = '/home/borrego/dataset/'
# Default gripper name
DEFAULT_GRIPPER = 'baxter'
# Default dataset name
DEFAULT_DS = 'kit'
#DEFAULT_DS = 'random_objects'
# Default grasp output directory
DEFAULT_OUT = 'out'

class CreateDBUtil(object):

  def __init__(self):
    """ Creates object  
    """    
    self.api = dexnet.DexNet()
    self.cfg = YamlConfig(DEFAULT_CFG)
    self.api.open_database(DEFAULT_DB_PATH, create_db=True)
    self.api.open_dataset(DEFAULT_DS, self.cfg, create_ds=True)

    #self.addObjects(DEFAULT_DS, DEFAULT_OBJ_PATH, self.cfg)
    #obj_name = self.api.list_objects()[0]
    #self.api.sample_grasps(config=self.cfg, object_name=None, gripper_name=DEFAULT_GRIPPER)

    self.exportGrasps(DEFAULT_GRIPPER, DEFAULT_OUT)

    #self.api.compute_metrics(config=self.cfg, object_name=None, metric_name="force_closure", gripper_name=DEFAULT_GRIPPER)

    #self.api.display_object(obj_name, config=self.cfg)
    #self.api.display_stable_poses('000', config=self.cfg)
    #for obj_name in self.api.list_objects():
    #  self.api.display_grasps(obj_name, DEFAULT_GRIPPER, 'force_closure', config=self.cfg)

    self.api.close_database()

  def addObjects(self, dataset, object_path, config=None):
    """ Adds objects to the currently active dataset
    """

    dataset_path = os.path.join(DEFAULT_OBJ_PATH, dataset)

    # Google random object dataset
    if dataset == 'random_objects':

        for i in range(0, 1000):

          obj_name = "{0:0>3}_coll".format(i)
          obj_filename = '{name}.{ext}'.format(name=obj_name, ext='obj')
          obj_path = os.path.join(dataset_path, obj_filename)
          print(obj_path)

          try:
            self.api.add_object(obj_path, config)
          except Exception as e:
            print("Adding object failed: {}".format(str(e)))

    elif dataset == 'kit':

        files = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path) \
            if os.path.isfile(os.path.join(dataset_path, f))]

        for obj_path in files:
            try:
              self.api.add_object(obj_path, config)
            except Exception as e:
                print("Adding object failed: {}".format(str(e)))


  def exportGrasps(self, gripper_name, out_dir):
    """ Exports grasp comfigurations to file
    """

    for obj_name in self.api.list_objects():

      out_name = os.path.join(out_dir, '{}.grasp.yml'.format(obj_name))

      data = dict(
        object = dict(
          name = obj_name,
          grasp_candidates = dict()
        )
      )
      data['object']['grasp_candidates'][gripper_name] = dict()

      for i, grasp in enumerate(self.api.get_grasps(obj_name, gripper_name)):
        data['object']['grasp_candidates'][gripper_name][i] = dict()
        data['object']['grasp_candidates'][gripper_name][i]['tf'] = \
          grasp.gripper_pose().matrix.flatten().tolist()

      with open(out_name, 'w') as outfile:
        yaml.dump(data, outfile, default_flow_style=False)

def main(argv):
  """ Main executable function
  """
  db_util = CreateDBUtil()

if __name__ == '__main__':
  main(sys.argv)
