"""

"""

import sys
import dexnet
from autolab_core import YamlConfig
import os
import time
import yaml
import numpy as np
import dexnet.grasping.gripper as gr

# Stable pose
from meshpy import StablePose 

# Default database configuration file path
DEFAULT_CFG = 'cfg/apps/custom_database.yaml'
# Default object directory
DEFAULT_OBJ_PATH = '/home/borrego/dataset/'
# Default gripper name
DEFAULT_GRIPPER = 'baxter'
# Default dataset name
#DEFAULT_DS = 'kit'
DEFAULT_DS = 'random_objects'
# Default grasp output directory
DEFAULT_OUT = '../OUT/random'

class CreateDBUtil(object):

  def __init__(self):
    """ Creates object  
    """    
    self.api = dexnet.DexNet()
    self.cfg = YamlConfig(DEFAULT_CFG)
    self.api.open_database(self.cfg['db_path'], create_db=True)
    self.api.open_dataset(self.cfg['ds_name'], self.cfg, create_ds=True)

    #self.addObjects(self.cfg['ds_name'], self.cfg['obj_path'], self.cfg)
    #obj_name = self.api.list_objects()[0]

    #for object_name in self.api.list_objects():
    #  obj = self.api.dataset[object_name]
    #  obj.stable_poses_ = obj.mesh_.stable_poses(min_prob=self.cfg['stp_min_prob'])
    #  self.api.dataset.store_stable_poses(object_name, obj.stable_poses_, force_overwrite=True)

    #for i in range(500, 1000):
    #  object_name = '{0:0>3}_coll'.format(i)
    #  self.api.sample_grasps(config=self.cfg, object_name=object_name, gripper_name=self.cfg['gripper'])

    #self.importStablePoses(self.cfg['in_stable_poses_dir'], self.cfg)
    self.exportGrasps(self.cfg['gripper'], self.cfg['out_grasps_dir'], self.cfg)


    #for i in range(0, 1000):
    #  object_name = '{0:0>3}_coll'.format(i)
    #for object_name in self.api.list_objects():
    #  self.api.compute_metrics(config=self.cfg, object_name=object_name, metric_name="force_closure", gripper_name=DEFAULT_GRIPPER)

    #self.api.display_object(obj_name, config=self.cfg)
    #self.api.display_stable_poses('000', config=self.cfg)
    #for obj_name in self.api.list_objects():
    
    #for i in range(0, 1000, 5):
    #  object_name = '{0:0>3}_coll'.format(i)
    #  self.api.display_grasps(object_name, self.cfg['gripper'], 'force_closure', config=self.cfg)

    self.api.close_database()

  def addObjects(self, dataset, object_path, config):
    """ Adds objects to the currently active dataset
    """

    dataset_path = os.path.join(config['ds_name'], dataset)

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


  def exportGrasps(self, gripper_name, out_dir, config):
    """ Exports grasp comfigurations to file
    """

    gripper = gr.RobotGripper.load(gripper_name, gripper_dir=config['gripper_dir'])

    for obj_name in self.api.list_objects():

      out_name = os.path.join(out_dir, '{}.grasp.yml'.format(obj_name))

      data = dict(
        object = dict(
          name = str(obj_name),
          grasp_candidates = dict()
        )
      )
      data['object']['grasp_candidates'][gripper_name] = dict()

      for i, grasp in enumerate(self.api.get_grasps(obj_name, gripper_name)):
        
        stable_pose = self.api.dataset.stable_pose(obj_name, 'pose_0')
        corrected_grasp = grasp.perpendicular_table(stable_pose)
        grasp_pose = corrected_grasp.gripper_pose(gripper)
        
        data['object']['grasp_candidates'][gripper_name][i] = dict()
        data['object']['grasp_candidates'][gripper_name][i]['tf'] = \
          grasp_pose.matrix.flatten().tolist()[0:12]

      with open(out_name, 'w') as outfile:
        yaml.dump(data, outfile, default_flow_style=False)

  def importStablePoses(self, in_dir, config):
    ''' Import stable poses from external files to HDF5 database
    '''

    prob = 1.0

    for obj_name in self.api.list_objects():

      stable_poses = []

      in_name = os.path.join(in_dir, '{}.rest.yml'.format(obj_name))
      data = []

      with open(in_name, 'r') as f:
        data = yaml.load(f)      
        for i in range(len(data[obj_name]['tf'])):
          mat = data[obj_name]['tf'][i]
          rot = np.array([ mat[0:3], mat[4:7], mat[8:11] ])
          x0 = np.array([ mat[3], mat[7], mat[11] ])
          stable_pose = StablePose(prob, rot, x0, stp_id='pose_{}'.format(i))
          stable_poses.append(stable_pose)

      self.api.dataset.store_stable_poses(obj_name, stable_poses, force_overwrite=True)

  def importMetrics(self, in_dir, metric_name, config=None):
    print("TODO")


def main(argv):
  """ Main executable function
  """
  db_util = CreateDBUtil()

if __name__ == '__main__':
  main(sys.argv)
