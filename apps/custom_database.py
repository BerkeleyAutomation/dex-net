"""

"""

import sys
import dexnet
from autolab_core import YamlConfig
import os

# Default database file path
DEFAULT_DB_PATH = 'kit_tmp.db.hdf5'
# Default database configuration file path
DEFAULT_CFG = 'cfg/apps/custom_database.yaml'
# Default object directory
DEFAULT_OBJ_PATH = '/home/borrego/dataset/'
# Default grasp output filename
DEFAULT_OUT = 'out.yml'
# Default dataset name
DEFAULT_DS = 'kit'

# Supported mesh file extensions
SUPPORTED_MESH_FORMATS = ['.obj', '.off', '.wrl', '.stl']

class CreateDBUtil(object):

  def __init__(self):
    """ Creates object  
    """
    self.api = dexnet.DexNet()
    self.cfg = YamlConfig(DEFAULT_CFG)
    self.api.open_database(DEFAULT_DB_PATH, create_db=True)
    self.api.open_dataset(DEFAULT_DS, self.cfg, create_ds=True)
    
    obj_name = "Seal_800_tex"

    self.addObjects(DEFAULT_DS, DEFAULT_OBJ_PATH, self.cfg)
    self.api.sample_grasps(config=self.cfg, object_name=obj_name, gripper_name="baxter")
    
    #self.exportGrasps(DEFAULT_OUT)

    # first = self.api.list_objects()[0]

    #self.api.display_object(first, config=self.cfg)

    self.api.compute_metrics(config=self.cfg, metric_name="force_closure", object_name=obj_name, gripper_name="baxter")
    #self.api.display_stable_poses('000', config=self.cfg)
    self.api.display_grasps(obj_name, 'baxter', 'force_closure', config=self.cfg)

    self.api.close_database()

  def addObjects(self, dataset, object_path, config=None):
    """ Adds objects to the currently active dataset
    """

    dataset_path = os.path.join(DEFAULT_OBJ_PATH, dataset)

    # Google random object dataset
    if dataset == 'random_objects':

        for i in range(0, 1):

          obj_name = "{0:0>3}".format(i)
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
              if obj_path == '/home/borrego/dataset/kit/Seal_800_tex.obj':
                self.api.add_object(obj_path, config)
            except Exception as e:
                print("Adding object failed: {}".format(str(e)))


  def exportGrasps(self, out_path):
    """ Exports grasp comfigurations to file
    """
    for obj_name in self.api.list_objects():
      for grasp in self.api.get_grasps(obj_name, 'baxter'):
        print(grasp.center)


def main(argv):
  """ Main executable function
  """
  db_util = CreateDBUtil()

if __name__ == '__main__':
  main(sys.argv)
