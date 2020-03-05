"""
Common 3D visualizations
Author: Matthew Matl and Jeff Mahler
"""
import uuid
import copy
import logging
import os

import imageio
import numpy as np
import trimesh
from shapely.geometry import Polygon

from autolab_core import RigidTransform, BagOfPoints, Point, PointCloud

from meshrender.scene import Scene
from meshrender.light import AmbientLight
from meshrender.scene_object import SceneObject
from meshrender import InstancedSceneObject, SceneViewer, MaterialProperties

class Visualizer3D:
    """
    Class containing static methods for visualization.
    The interface is styled after pyplot.
    Should be thought of as a namespace rather than a class.
    """
    _scene = Scene(background_color=np.array([1.0, 1.0, 1.0]))
    _scene.ambient_light = AmbientLight(color=[1.0, 1.0, 1.0], strength=1.0)
    _init_size = np.array([640,480])
    _save_directory = None


    @staticmethod
    def figure(bgcolor=(1,1,1), size=(1000,1000)):
        """Create a blank figure.

        Parameters
        ----------
        bgcolor : (3,) float
           Color of the background with values in [0,1].
        size : (2,) int
           Width and height of the figure in pixels.
        """
        Visualizer3D._scene = Scene(background_color=np.array(bgcolor))
        Visualizer3D._scene.ambient_light = AmbientLight(color=[1.0, 1.0, 1.0], strength=1.0)
        Visualizer3D._init_size = np.array(size)


    @staticmethod
    def show(animate=False, axis=np.array([0.,0.,1.]), clf=True, **kwargs):
        """Display the current figure and enable interaction.

        Parameters
        ----------
        animate : bool
            Whether or not to animate the scene.
        axis : (3,) float or None
            If present, the animation will rotate about the given axis in world coordinates.
            Otherwise, the animation will rotate in azimuth.
        clf : bool
            If true, the Visualizer is cleared after showing the figure.
        kwargs : dict
            Other keyword arguments for the SceneViewer instance.
        """
        x = SceneViewer(Visualizer3D._scene,
                        size=Visualizer3D._init_size,
                        animate=animate,
                        animate_axis=axis,
                        save_directory=Visualizer3D._save_directory,
                        **kwargs)
        if x.save_directory:
            Visualizer3D._save_directory = x.save_directory
        if clf:
            Visualizer3D.clf()


    @staticmethod
    def render(n_frames=1, axis=np.array([0.,0.,1.]), clf=True, **kwargs):
        """Render frames from the viewer.

        Parameters
        ----------
        n_frames : int
            Number of frames to render. If more than one, the scene will animate.
        axis : (3,) float or None
            If present, the animation will rotate about the given axis in world coordinates.
            Otherwise, the animation will rotate in azimuth.
        clf : bool
            If true, the Visualizer is cleared after rendering the figure.
        kwargs : dict
            Other keyword arguments for the SceneViewer instance.

        Returns
        -------
        list of perception.ColorImage
            A list of ColorImages rendered from the viewer.
        """
        v = SceneViewer(Visualizer3D._scene,
                        size=Visualizer3D._init_size,
                        animate=(n_frames > 1),
                        animate_axis=axis,
                        max_frames=n_frames,
                        **kwargs)

        if clf:
            Visualizer3D.clf()

        return v.saved_frames


    @staticmethod
    def save(filename, n_frames=1, axis=np.array([0.,0.,1.]), clf=True, **kwargs):
        """Save frames from the viewer out to a file.

        Parameters
        ----------
        filename : str
            The filename in which to save the output image. If more than one frame,
            should have extension .gif.
        n_frames : int
            Number of frames to render. If more than one, the scene will animate.
        axis : (3,) float or None
            If present, the animation will rotate about the given axis in world coordinates.
            Otherwise, the animation will rotate in azimuth.
        clf : bool
            If true, the Visualizer is cleared after rendering the figure.
        kwargs : dict
            Other keyword arguments for the SceneViewer instance.
        """
        if n_frames >1 and os.path.splitext(filename)[1] != '.gif':
            raise ValueError('Expected .gif file for multiple-frame save.')
        v = SceneViewer(Visualizer3D._scene,
                        size=Visualizer3D._init_size,
                        animate=(n_frames > 1),
                        animate_axis=axis,
                        max_frames=n_frames,
                        **kwargs)
        data = [m.data for m in v.saved_frames]
        if len(data) > 1:
            imageio.mimwrite(filename, data, fps=v._animate_rate, palettesize=128, subrectangles=True)
        else:
            imageio.imwrite(filename, data[0])

        if clf:
            Visualizer3D.clf()

    @staticmethod
    def save_loop(filename, framerate=30, time=3.0, axis=np.array([0.,0.,1.]), clf=True, **kwargs):
        """Off-screen save a GIF of one rotation about the scene.

        Parameters
        ----------
        filename : str
            The filename in which to save the output image (should have extension .gif)
        framerate : int
            The frame rate at which to animate motion.
        time : float
            The number of seconds for one rotation.
        axis : (3,) float or None
            If present, the animation will rotate about the given axis in world coordinates.
            Otherwise, the animation will rotate in azimuth.
        clf : bool
            If true, the Visualizer is cleared after rendering the figure.
        kwargs : dict
            Other keyword arguments for the SceneViewer instance.
        """
        n_frames = framerate * time
        az = 2.0 * np.pi / n_frames
        Visualizer3D.save(filename, n_frames=n_frames, axis=axis, clf=clf,
                          animate_rate=framerate, animate_az=az)
        if clf:
            Visualizer3D.clf()

    @staticmethod
    def clf():
        """Clear the current figure
        """
        Visualizer3D._scene = Scene(background_color=Visualizer3D._scene.background_color)
        Visualizer3D._scene.ambient_light = AmbientLight(color=[1.0, 1.0, 1.0], strength=1.0)


    @staticmethod
    def close(*args, **kwargs):
        """Close the current figure
        """
        pass

    @staticmethod
    def get_object_keys():
        """Return the visualizer's object keys.

        Returns
        -------
        list of str
            The keys for the visualizer's objects.
        """
        return Visualizer3D._scene.objects.keys()

    @staticmethod
    def get_object(name):
        """Return a SceneObject corresponding to the given name.

        Returns
        -------
        meshrender.SceneObject
            The corresponding SceneObject.
        """
        return Visualizer3D._scene.objects[name]

    @staticmethod
    def points(points, T_points_world=None, color=np.array([0,1,0]), scale=0.01, n_cuts=20, subsample=None, random=False, name=None):
        """Scatter a point cloud in pose T_points_world.

        Parameters
        ----------
        points : autolab_core.BagOfPoints or (n,3) float
            The point set to visualize.
        T_points_world : autolab_core.RigidTransform
            Pose of points, specified as a transformation from point frame to world frame.
        color : (3,) or (n,3) float
            Color of whole cloud or per-point colors
        scale : float
            Radius of each point.
        n_cuts : int
            Number of longitude/latitude lines on sphere points.
        subsample : int
            Parameter of subsampling to display fewer points.
        name : str
            A name for the object to be added.
        """
        if isinstance(points, BagOfPoints):
            if points.dim != 3:
                raise ValueError('BagOfPoints must have dimension 3xN!')
        else:
            if type(points) is not np.ndarray:
                raise ValueError('Points visualizer expects BagOfPoints or numpy array!')
            if len(points.shape) == 1:
                points = points[:,np.newaxis].T
            if len(points.shape) != 2 or points.shape[1] != 3:
                raise ValueError('Numpy array of points must have dimension (N,3)')
            frame = 'points'
            if T_points_world:
                frame = T_points_world.from_frame
            points = PointCloud(points.T, frame=frame)

        color = np.array(color)
        if subsample is not None:
            num_points = points.num_points
            points, inds = points.subsample(subsample, random=random)
            if color.shape[0] == num_points and color.shape[1] == 3:
                color = color[inds,:]

        # transform into world frame
        if points.frame != 'world':
            if T_points_world is None:
                T_points_world = RigidTransform(from_frame=points.frame, to_frame='world')
            points_world = T_points_world * points
        else:
            points_world = points

        point_data = points_world.data
        if len(point_data.shape) == 1:
            point_data = point_data[:,np.newaxis]
        point_data = point_data.T

        mpcolor = color
        if len(color.shape) > 1:
            mpcolor = color[0]
        mp = MaterialProperties(
            color = np.array(mpcolor),
            k_a = 0.5,
            k_d = 0.3,
            k_s = 0.0,
            alpha = 10.0,
            smooth=True
        )

        # For each point, create a sphere of the specified color and size.
        sphere = trimesh.creation.uv_sphere(scale, [n_cuts, n_cuts])
        raw_pose_data = np.tile(np.eye(4), (points.num_points, 1))
        raw_pose_data[3::4, :3] = point_data

        instcolor = None
        if color.ndim == 2 and color.shape[0] == points.num_points and color.shape[1] == 3:
            instcolor = color
        obj = InstancedSceneObject(sphere, raw_pose_data=raw_pose_data, colors=instcolor, material=mp)
        if name is None:
            name = str(uuid.uuid4())
        Visualizer3D._scene.add_object(name, obj)

    @staticmethod
    def mesh(mesh, T_mesh_world=RigidTransform(from_frame='obj', to_frame='world'),
             style='surface', smooth=False, color=(0.5,0.5,0.5), name=None):
        """Visualize a 3D triangular mesh.

        Parameters
        ----------
        mesh : trimesh.Trimesh
            The mesh to visualize.
        T_mesh_world : autolab_core.RigidTransform
            The pose of the mesh, specified as a transformation from mesh frame to world frame.
        style : str
            Triangular mesh style, either 'surface' or 'wireframe'.
        smooth : bool
            If true, the mesh is smoothed before rendering.
        color : 3-tuple
            Color tuple.
        name : str
            A name for the object to be added.
        """
        if not isinstance(mesh, trimesh.Trimesh):
            raise ValueError('Must provide a trimesh.Trimesh object')

        mp = MaterialProperties(
            color = np.array(color),
            k_a = 0.5,
            k_d = 0.3,
            k_s = 0.1,
            alpha = 10.0,
            smooth=smooth,
            wireframe=(style == 'wireframe')
        )

        obj = SceneObject(mesh, T_mesh_world, mp)
        if name is None:
            name = str(uuid.uuid4())
        Visualizer3D._scene.add_object(name, obj)


    @staticmethod
    def mesh_stable_pose(mesh, T_obj_table,
                         T_table_world=RigidTransform(from_frame='table', to_frame='world'),
                         style='wireframe', smooth=False, color=(0.5,0.5,0.5),
                         dim=0.15, plot_table=True, plot_com=False, name=None):
        """Visualize a mesh in a stable pose.

        Parameters
        ----------
        mesh : trimesh.Trimesh
            The mesh to visualize.
        T_obj_table : autolab_core.RigidTransform
            Pose of object relative to table.
        T_table_world : autolab_core.RigidTransform
            Pose of table relative to world.
        style : str
            Triangular mesh style, either 'surface' or 'wireframe'.
        smooth : bool
            If true, the mesh is smoothed before rendering.
        color : 3-tuple
            Color tuple.
        dim : float
            The side-length for the table.
        plot_table : bool
            If true, a table is visualized as well.
        plot_com : bool
            If true, a ball is visualized at the object's center of mass.
        name : str
            A name for the object to be added.

        Returns
        -------
        autolab_core.RigidTransform
            The pose of the mesh in world frame.
        """
        T_obj_table = T_obj_table.as_frames('obj', 'table')
        T_obj_world = T_table_world * T_obj_table

        Visualizer3D.mesh(mesh, T_obj_world, style=style, smooth=smooth, color=color, name=name)
        if plot_table:
            Visualizer3D.table(T_table_world, dim=dim)
        if plot_com:
            Visualizer3D.points(Point(np.array(mesh.center_mass), 'obj'), T_obj_world, scale=0.01)
        return T_obj_world

    @staticmethod
    def pose(T_frame_world, alpha=0.1, tube_radius=0.005, center_scale=0.01):
        """Plot a 3D pose as a set of axes (x red, y green, z blue).

        Parameters
        ----------
        T_frame_world : autolab_core.RigidTransform
            The pose relative to world coordinates.
        alpha : float
            Length of plotted x,y,z axes.
        tube_radius : float
            Radius of plotted x,y,z axes.
        center_scale : float
            Radius of the pose's origin ball.
        """
        R = T_frame_world.rotation
        t = T_frame_world.translation

        x_axis_tf = np.array([t, t + alpha * R[:,0]])
        y_axis_tf = np.array([t, t + alpha * R[:,1]])
        z_axis_tf = np.array([t, t + alpha * R[:,2]])

        Visualizer3D.points(t, color=(1,1,1), scale=center_scale)
        Visualizer3D.plot3d(x_axis_tf, color=(1,0,0), tube_radius=tube_radius)
        Visualizer3D.plot3d(y_axis_tf, color=(0,1,0), tube_radius=tube_radius)
        Visualizer3D.plot3d(z_axis_tf, color=(0,0,1), tube_radius=tube_radius)

    @staticmethod
    def table(T_table_world=RigidTransform(from_frame='table', to_frame='world'), dim=0.16, color=(0,0,0)):
        """Plot a table mesh in 3D.

        Parameters
        ----------
        T_table_world : autolab_core.RigidTransform
            Pose of table relative to world.
        dim : float
            The side-length for the table.
        color : 3-tuple
            Color tuple.
        """

        table_vertices = np.array([[ dim,  dim, 0],
                                   [ dim, -dim, 0],
                                   [-dim,  dim, 0],
                                   [-dim, -dim, 0]]).astype('float')
        table_tris = np.array([[0, 1, 2], [1, 2, 3]])
        table_mesh = trimesh.Trimesh(table_vertices, table_tris)
        table_mesh.apply_transform(T_table_world.matrix)
        Visualizer3D.mesh(table_mesh, style='surface', smooth=True, color=color)

    @staticmethod
    def plot3d(points, color=(0.5, 0.5, 0.5), tube_radius=0.005, n_components=30, name=None):
        """Plot a 3d curve through a set of points using tubes.

        Parameters
        ----------
        points : (n,3) float
            A series of 3D points that define a curve in space.
        color : (3,) float
            The color of the tube.
        tube_radius : float
            Radius of tube representing curve.
        n_components : int
            The number of edges in each polygon representing the tube.
        name : str
            A name for the object to be added.
        """
        points = np.asanyarray(points)
        mp = MaterialProperties(
            color = np.array(color),
            k_a = 0.5,
            k_d = 0.3,
            k_s = 0.0,
            alpha = 10.0,
            smooth=True
        )

        # Generate circular polygon
        vec = np.array([0,1]) * tube_radius
        angle = np.pi * 2.0 / n_components
        rotmat = np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)]
        ])
        perim = []
        for i in range(n_components):
            perim.append(vec)
            vec = np.dot(rotmat, vec)
        poly = Polygon(perim)

        # Sweep it out along the path
        mesh = trimesh.creation.sweep_polygon(poly, points)
        obj = SceneObject(mesh, material=mp)
        if name is None:
            name = str(uuid.uuid4())
        Visualizer3D._scene.add_object(name, obj)
