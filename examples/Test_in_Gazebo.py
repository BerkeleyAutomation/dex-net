# -*- coding: utf-8 -*-

"""
YAML Configuration File Parameters
----------------------------------
database : str
    full path to a Dex-Net HDF5 database
dataset : str
    name of the dataset containing the object instance to grasp
gripper : str
    name of the gripper to use
metric : str
    name of the grasp robustness metric to use
object : str
    name of the object to use (in practice instance recognition is necessary to determine the object instance from images)
"""
#Imports for dexnet package
import os
from autolab_core import RigidTransform, YamlConfig 
from dexnet import DexNet
from dexnet.grasping import RobotGripper

#Imports for MoveIt Gazebo Integration 
import sys
import rospy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg

# Transform Listener
import tf

if __name__ == '__main__':
    rospy.init_node('move_group_python_interface', anonymous=True)
    listener = tf.TransformListener()
    
    filename = 'cfg/examples/execute_grasp_registration.yaml'
    config_filename = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                       '..',
				       filename)
    print "Configuration File Path: ", config_filename

    # Incase of no file name
    if config_filename is None:
        print "Configuration file path not specified"
  
    # Turn relative paths absolute
    if not os.path.isabs(config_filename):
        config_filename = os.path.join(os.getcwd(), config_filename)

    
    # Configuration file
    config = YamlConfig(config_filename)
    database_name = config['database']
    dataset_name = config['dataset']
    gripper_name = config['gripper']
    metric_name = config['metric']
    object_name = config['object']
    
    # Load gripper
    gripper = RobotGripper.load(gripper_name)

    # Open Dex-Net API
    dexnet_handle = DexNet()
    dexnet_handle.open_database(database_name)
    print database_name, dataset_name
    dexnet_handle.open_dataset(dataset_name)

    # Read the most robust grasp
    sorted_grasps = dexnet_handle.dataset.sorted_grasps(object_name, metric_name, 							
                                                             gripper=gripper_name)
    most_robust_grasp = sorted_grasps[0][0] 
    print most_robust_grasp.gripper_pose(gripper)
    # print most_robust_grasp.gripper_pose(gripper).translation
    # transform into the robot reference frame for control
    T_gripper_obj = most_robust_grasp.gripper_pose(gripper)
    while True:
	    try:
    		(trans,rot) = listener.lookupTransform('/panda_link7', '/panda_link0', rospy.Time(0))
    	    except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
	    	continue
    	    break
	
    print trans, rot
        # fake transformation from the camera to the robot
    T_camera_robot = RigidTransform(rotation=rot,
                                    translation=trans,
                                    from_frame='camera', to_frame='robot')

    # fake transformation from a known 3D object to the camera 
    T_obj_camera = RigidTransform(rotation=[0,0,0,1],
                                  translation=[0,0,0],
                                  from_frame='obj', to_frame='camera')
    T_gripper_robot = T_camera_robot * T_obj_camera * T_gripper_obj

    print T_gripper_robot

    moveit_commander.roscpp_initialize(sys.argv)
    robot = moveit_commander.RobotCommander()
    scene = moveit_commander.PlanningSceneInterface()

    group = moveit_commander.MoveGroupCommander("arm")
    
    display_trajectory_publisher = rospy.Publisher('/move_group/display_planned_path',
                                     moveit_msgs.msg.DisplayTrajectory, queue_size=20)
    planning_frame = group.get_planning_frame()
    print planning_frame   
    pose_goal = geometry_msgs.msg.Pose()
    pose_goal.orientation.x = T_gripper_robot.quaternion[0]
    pose_goal.orientation.y = T_gripper_robot.quaternion[1]
    pose_goal.orientation.z = T_gripper_robot.quaternion[2]
    pose_goal.orientation.w = T_gripper_robot.quaternion[3]

    pose_goal.position.x = T_gripper_robot.translation[0] 
    pose_goal.position.y =T_gripper_robot.translation[1]
    pose_goal.position.z = T_gripper_robot.translation[2]
    group.set_pose_target(pose_goal)
    plan = group.go(wait=True)
    group.stop()
    group.clear_pose_targets() # clear targets after planning
  #  group_hand = moveit_commander.MoveGroupCommander("hand")
'''
def GraspObject():
	comm = GraspActionGoal()
	comm.goal.width = 0.01
	comm.goal.speed = 0.05
	comm.goal.force = 5
	comm.goal.epsilon.inner = 0.1
	comm.goal.epsilon.outer = 0.1
        pubG = rospy.Publisher("franka_gripper/grasp/goal",  GraspActionGoal, 
                           queue_size=1,latch=True)
	rate = rospy.Rate(10)
	t_end = time.time() + 1  
	while time.time() < t_end:
		pubG.publish(comm)  
		rate.sleep()
		now = rospy.get_rostime()
	rospy.sleep(2)
	rospy.sleep(2)
def ReleaseObject():
	comm = MoveActionGoal()
	comm.goal.width = 0.035
	comm.goal.speed = 0.2
	pubR = rospy.Publisher("franka_gripper/move/goal",  MoveActionGoal , queue_size=1, latch=True)
	rate = rospy.Rate(10)
	t_end = time.time() + 1
	while time.time() < t_end:
		pubR.publish(comm)  
		rate.sleep()
		now = rospy.get_rostime()
	rospy.sleep(2)

def MoveGripper():
	comm = MoveActionGoal()

	comm.goal.width = 0.01
	comm.goal.speed = 0.2
	pubR = rospy.Publisher("franka_gripper/move/goal",  MoveActionGoal , 						queue_size=1, latch=True)
	rate = rospy.Rate(10)
	t_end = time.time() + 1
	while time.time() < t_end:
		pubR.publish(comm)  
		rate.sleep()
		now = rospy.get_rostime()

	rospy.sleep(2)
'''

