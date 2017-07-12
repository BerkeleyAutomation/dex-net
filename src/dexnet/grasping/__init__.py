from contacts import Contact3D, SurfaceWindow
from graspable_object import GraspableObject, GraspableObject3D
from grasp import Grasp, PointGrasp, ParallelJawPtGrasp3D
from gripper import RobotGripper
from grasp_quality_config import GraspQualityConfig, QuasiStaticGraspQualityConfig, RobustQuasiStaticGraspQualityConfig, GraspQualityConfigFactory
from quality import PointGraspMetrics3D
from random_variables import GraspableObjectPoseGaussianRV, ParallelJawGraspPoseGaussianRV, ParamsGaussianRV
from robust_grasp_quality import QuasiStaticGraspQualityRV, RobustPointGraspMetrics3D
from grasp_quality_function import GraspQualityResult, GraspQualityFunction, QuasiStaticQualityFunction, RobustQuasiStaticQualityFunction, GraspQualityFunctionFactory

try:
    from collision_checker import OpenRaveCollisionChecker, GraspCollisionChecker
except Exception:
    print 'Unable to import OpenRaveCollisionChecker and GraspCollisionChecker! Likely due to missing OpenRave dependency.'
    print 'Install OpenRave 0.9 from source if required. Instructions can be found at http://openrave.org/docs/latest_stable/coreapihtml/installation_linux.html'

from grasp_sampler import GraspSampler, UniformGraspSampler, GaussianGraspSampler, AntipodalGraspSampler

__all__ = ['Contact3D', 'GraspableObject', 'GraspableObject3D', 'ParallelJawPtGrasp3D',
           'Grasp', 'PointGrasp', 'RobotGripper', 'PointGraspMetrics3D',
           'GraspQualityConfig', 'QuasiStaticGraspQualityConfig', 'RobustQuasiStaticGraspQualityConfig', 'GraspQualityConfigFactory',
           'GraspSampler', 'UniformGraspSampler', 'GaussianGraspSampler', 'AntipodalGraspSampler',
           'GraspableObjectPoseGaussianRV', 'ParallelJawGraspPoseGaussianRV', 'ParamsGaussianRV',
           'QuasiStaticGraspQualityRV', 'RobustPointGraspMetrics3D',
           'GraspQualityResult', 'GraspQualityFunction', 'QuasiStaticQualityFunction', 'RobustQuasiStaticQualityFunction', 'GraspQualityFunctionFactory',
           'OpenRaveCollisionChecker', 'GraspCollisionChecker',
]

# module name spoofing for correct imports
import grasp
import sys
sys.modules['dexnet.grasp'] = grasp
