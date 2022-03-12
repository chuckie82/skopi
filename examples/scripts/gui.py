import sys
sys.path.append("../..")

import argparse
import collections
import os
import numpy as np
import numba
import matplotlib
import matplotlib.pyplot as plt
import h5py as h5
import time

from matplotlib.backends.qt_compat import QtWidgets, QtCore
## from matplotlib.backends.qt_compat import QtWidgets, QtCore, is_pyqt5
## if is_pyqt5():
##     from matplotlib.backends.backend_qt5agg import (
##             FigureCanvas, NavigationToolbar2QT as NavigationToolbar)
## else:
##     from matplotlib.backends.backend_qt4agg import (
##         FigureCanvas, NavigationToolbar2QT as NavigationToolbar)
from matplotlib.backends.backend_qt5agg import (
        FigureCanvas, NavigationToolbar2QT as NavigationToolbar)
from matplotlib.figure import Figure
from matplotlib.colors import LogNorm
from mpl_toolkits.mplot3d import Axes3D

import skopi as sk
import skopi.gpu as sg
from skopi.detector.pnccd import PnccdDetector


# Set default matplotlib parameters
matplotlib.rcParams['image.origin'] = 'lower'
matplotlib.rcParams['image.interpolation'] = None
matplotlib.rcParams['image.cmap'] = 'jet'


class ApplicationWindow(QtWidgets.QMainWindow):
    def __init__(self, pdb_file, colors=False, debug=False):
        super(ApplicationWindow, self).__init__()
        self.debug = debug

        # Create a particle object
        self.particle = sk.Particle()
        self.particle.read_pdb(pdb_file, ff='WK')

        # Load beam
        beam = sk.Beam('../input/beam/amo86615.beam') 

        # Load and initialize the detector
        self.det = PnccdDetector(
            geom='../input/lcls/amo86615/PNCCD::CalibV1/'
                 'Camp.0:pnCCD.1/geometry/0-end.data', 
            beam=beam)

        mesh_length = 151 if not debug else 31
        mesh, self.voxel_length = self.det.get_reciprocal_mesh(
            voxel_number_1d=mesh_length)

        self.volume = sg.calculate_diffraction_pattern_gpu(
            mesh, self.particle, return_type='intensity')

        self.pixel_momentum = self.det.pixel_position_reciprocal

        if colors:
            color_map = collections.defaultdict(lambda: "#000000", {
                "C": "#ff0000",
                "N": "#00ff00",
                "O": "#0000ff",
            })
            colors = [color_map[s] for s in self.particle.atomic_symbol]
        else:
            colors = None

        self._azim = None
        self._elev = None
        self._time = 0.
        self._uptodate = False

        self._main = QtWidgets.QWidget()
        self.setCentralWidget(self._main)
        layout = QtWidgets.QHBoxLayout(self._main)

        real3d_canvas = FigureCanvas(Figure(figsize=(4, 4)))
        layout.addWidget(real3d_canvas)
        self.addToolBar(NavigationToolbar(real3d_canvas, self))

        self._real3d_ax = real3d_canvas.figure.subplots(subplot_kw={"projection":'3d'})
        self._real3d_ax.scatter(
            -self.particle.atom_pos[:, 2],
            self.particle.atom_pos[:, 1],
            self.particle.atom_pos[:, 0],
            s=1,
            c=colors,
        )
        self._real3d_ax.set_title("3D Protein")
        self._real3d_ax.set_xlabel('-Z')
        self._real3d_ax.set_ylabel('Y')
        self._real3d_ax.set_zlabel('X')

        if self.debug:
            real2d_canvas = FigureCanvas(Figure(figsize=(4, 4)))
            layout.addWidget(real2d_canvas)
            self.addToolBar(NavigationToolbar(real2d_canvas, self))

            self._real2d_ax = real2d_canvas.figure.subplots()

        recip_canvas = FigureCanvas(Figure(figsize=(4, 4)))
        layout.addWidget(recip_canvas)
        self.addToolBar(NavigationToolbar(recip_canvas, self))

        self._recip_ax = recip_canvas.figure.subplots()

        self._timer = recip_canvas.new_timer(
            100, [(self._update_canvas, (), {})])
        self._timer.start()

    def _update_canvas(self):
        azim = np.radians(self._real3d_ax.azim)
        elev = np.radians(self._real3d_ax.elev)

        if azim != self._azim or elev != self._elev:
            # Record and mark for update
            self._azim = azim
            self._elev = elev
            self._time = time.time()
            self._uptodate = False
            return

        if self._uptodate:
            return

        if time.time() - self._time < 1.:
            # Wait a bit more
            return

        self._uptodate = True

        axis_azim = np.array([1., 0., 0.])
        axis_elev = np.array([0., 1., 0.])
        rot_azim = sk.geometry.angle_axis_to_rot3d(axis_azim, -azim)
        rot_elev = sk.geometry.angle_axis_to_rot3d(axis_elev, elev)
        rot = np.matmul(rot_elev, rot_azim)

        if self.debug:
            print("Azim: {:.2f} - elev: {:.2f}".format(azim, elev))

            rpos = np.matmul(rot, self.particle.atom_pos.T)

            self._real2d_ax.clear()
            self._real2d_ax.plot(
                rpos[1],
                rpos[0],
                ".")
            self._real2d_ax.set_title("Real-space XY projection")
            self._real2d_ax.set_aspect('equal')
            self._real2d_ax.set_xlabel('Y')
            self._real2d_ax.set_ylabel('X')
            self._real2d_ax.figure.canvas.draw()

        quat = sk.geometry.rotmat_to_quaternion(rot)
        slice_ = sk.geometry.take_slice(
            self.volume, self.voxel_length,
            self.pixel_momentum, quat, inverse=True)
        img = self.det.assemble_image_stack(slice_)
        self._recip_ax.clear()
        self._recip_ax.imshow(img, norm=LogNorm())
        self._recip_ax.set_title("Diffraction pattern")
        self._recip_ax.set_xlabel('K')
        self._recip_ax.set_ylabel('H')
        self._recip_ax.figure.canvas.draw()
        self._recip_ax.figure.canvas.draw()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Minimal GUI for skopi.')
    parser.add_argument(
        'pdb_file', type=str, nargs='?',
        default='../input/pdb/3iyf.pdb',
        help='path to PDB file')
    parser.add_argument(
        '-c', dest='colors', action='store_const',
        const=True, default=False,
        help='use colors for atoms')
    parser.add_argument(
        '-d', dest='debug', action='store_const',
        const=True, default=False,
        help='use debug mode')

    args = parser.parse_args()

    if not os.path.exists(args.pdb_file):
        sys.exit("Failure: PDB file {} does not exist.".format(args.pdb_file))

    app = QtWidgets.QApplication(sys.argv)

    print("Application starting, please be patient.")

    window = ApplicationWindow(
        args.pdb_file, colors=args.colors, debug=args.debug)
    window.show()

    app.exec_()
