# -*- coding: utf-8 -*-
"""
3-patch series-fed microstrip array (openEMS Python API, matplotlib instead of pylab)
"""

import os, tempfile, glob
import numpy as np
import matplotlib.pyplot as plt

from CSXCAD import ContinuousStructure
from openEMS import openEMS
from openEMS.physical_constants import *

# --------------------- Parameters ---------------------
unit = 1e-3  # 1 mm
f0 = 5.8e9
fmax = 7.0e9
er = 3.66
h = 1.52  # mm
GW = 35.0  # mm
GL = 120.0  # mm
Wpatch = 17.2  # mm
Lpatch = 12.3  # mm
p = 28.6  # mm
Wms = 3.15  # mm
y_port = -GL / 2 + 3.0  # mm
port_len = 8.0  # mm

CL = 4.0  # mm
CW = 0.5  # mm
USE_GAP_INSET = True
GAP = 0.15  # mm

air_above = 25.0
air_side = 20.0
air_below = 10.0

# --------------------- Simulation setup ---------------------
Sim_Path = os.path.join(tempfile.gettempdir(), 'series3_patch')
post_proc_only = False
preview_geometry = True       # True -> write XML and open AppCSXCAD before run
open_paraview    = True        # True -> try to open field file(s) in ParaView after run

FDTD = openEMS()
FDTD.SetGaussExcite(f0, f0*0.25)
FDTD.SetBoundaryCond(['PML_8', 'PML_8', 'PML_8', 'PML_8', 'PML_8', 'PML_8'])

CSX = ContinuousStructure()
FDTD.SetCSX(CSX)
mesh = CSX.GetGrid()
mesh.SetDeltaUnit(unit)

lambda_sub_mm = C0 / (fmax * np.sqrt(er)) / unit
resolution = lambda_sub_mm / 15.0

third_mesh = np.array([2 * resolution / 3.0, -resolution / 3.0]) / 4.0

mesh.AddLine('x', 0)
mesh.AddLine('x', Wms / 2.0 + third_mesh)
mesh.AddLine('x', -Wms / 2.0 - third_mesh)
mesh.AddLine('x', Wpatch / 2.0 + 1.0)
mesh.AddLine('x', -Wpatch / 2.0 - 1.0)
mesh.AddLine('x', [-GW / 2.0 - air_side, GW / 2.0 + air_side])
mesh.SmoothMeshLines('x', resolution / 3.0)

yc = [-p, 0.0, +p]
y_lines = [
    y_port, y_port + port_len,
            yc[0] - Lpatch / 2.0 - 1.0, yc[0] + Lpatch / 2.0 + 1.0,
            yc[1] - Lpatch / 2.0 - 1.0, yc[1] + Lpatch / 2.0 + 1.0,
            yc[2] - Lpatch / 2.0 - 1.0, yc[2] + Lpatch / 2.0 + 1.0,
            -GL / 2.0 - air_side, +GL / 2.0 + air_side
]
for yv in y_lines: mesh.AddLine('y', yv)
mesh.SmoothMeshLines('y', resolution / 2.0)

mesh.AddLine('z', np.linspace(0, h, 5))
mesh.AddLine('z', h + 0.5)
mesh.AddLine('z', h + air_above)
mesh.AddLine('z', 0 - air_below)
mesh.SmoothMeshLines('z', resolution)

subst = CSX.AddMaterial('Subst', epsilon=er, kappa=0.002)
subst.AddBox([-GW / 2.0, -GL / 2.0, 0.0], [GW / 2.0, GL / 2.0, h])

pec = CSX.AddMetal('PEC')
pec.AddBox([-GW / 2.0, -GL / 2.0, 0.0], [GW / 2.0, GL / 2.0, 0.0], priority=5)

for k in range(0, 3):
    x1, x2 = -Wpatch / 2.0, +Wpatch / 2.0
    y1, y2 = yc[k] - Lpatch / 2.0, yc[k] + Lpatch / 2.0
    pec.AddBox([x1, y1, h], [x2, y2, h], priority=10)

air = CSX.AddMaterial('air', epsilon=1.0)
air.AddBox([-Wms/2.0,    yc[2] - Lpatch / 2.0,      h],
           [-Wms/2.0-CW, yc[2] - Lpatch / 2.0 + CL, h], priority=15)
air.AddBox([Wms/2.0,    yc[2] - Lpatch / 2.0,      h],
           [Wms/2.0+CW, yc[2] - Lpatch / 2.0 + CL, h], priority=15)

y_line_end = yc[2] + Lpatch / 2.0
pec.AddBox([-Wms/2.0, y_port,      h],
           [+Wms/2.0, y_line_end,  h], priority=12)

# --------------------- Port (MSL) ---------------------
port_start = [-Wms / 2.0, y_port, 0.0]
port_stop = [+Wms / 2.0, y_port + port_len, h]
#
# port[0] = FDTD.AddMSLPort(
#     1, pec, portstart, portstop,
#     'y', 'z',
#     excite=+1,
#     FeedShift=10 * resolution,
#     MeasPlaneShift=port_len / 3.0,
#     priority=30
# )

feed_R = 50
port = [None]
port[0] = FDTD.AddLumpedPort(1, feed_R, port_start, port_stop, 'z', 1.0, priority=5, edges2grid='xy')

# --------------------- Field dumps (for ParaView) ---------------------
dump_type_E = 0   # E-field
file_type   = 0   # 0 for VTK (requires VTK-enabled build of CSXCAD/openEMS), 1 - HDF5 file

et = CSX.AddDump('Et', dump_type=dump_type_E, file_type=file_type)
# Save a big box including substrate + air above
et.AddBox(start=[-GW/2.0 - 5.0, -GL/2.0 - 5.0, -5.0],
          stop =[ GW/2.0 + 5.0,  GL/2.0 + 5.0,  h + air_above + 5.0])
# (Optional) also H-field or current density:
# ht = CSX.AddDump('Ht', dump_type=1, file_type=file_type); ht.AddBox(start=..., stop=...)
# jt = CSX.AddDump('Jt', dump_type=3, file_type=file_type); jt.AddBox(start=..., stop=...)

# --------------------- (Optional) Preview geometry ---------------------
if preview_geometry:
    CSX_file = os.path.join(Sim_Path, 'series3.xml')
    os.makedirs(Sim_Path, exist_ok=True)
    CSX.Write2XML(CSX_file)
    from CSXCAD import AppCSXCAD_BIN

    os.system(AppCSXCAD_BIN + f' "{CSX_file}"')

if not post_proc_only:
    FDTD.SetEndCriteria(1e-4)  # ~ -40 dB Energy
    FDTD.Run(Sim_Path, cleanup=True, numThreads=24)

# --------------------- Post-processing ---------------------
f = np.linspace(2.0e9, fmax, 1601)
port[0].CalcPort(Sim_Path, f, ref_impedance=50)
s11 = port[0].uf_ref / port[0].uf_inc

plt.figure(figsize=(7, 4))
plt.plot(f / 1e9, 20 * np.log10(abs(s11)), 'b-', lw=2, label='|S11|')
plt.grid(True, ls='--', alpha=0.4)
plt.legend()
plt.xlabel('Frequency (GHz)')
plt.ylabel('|S11| (dB)')
plt.ylim([-40, 2])
plt.title('3-patch series-fed |S11|')
plt.show()

# --------------------- Open in ParaView (optional) ---------------------
if open_paraview:
    # Try to find a field file; prefer VTK if available
    candidates = []
    # VTK XML rectilinear grids often saved as .vtr; also try *.vtk, *.vtu just in case
    candidates += glob.glob(os.path.join(Sim_Path, 'Et*.[vV][tT][rR]'))
    candidates += glob.glob(os.path.join(Sim_Path, 'Et*.[vV][tT][kK]'))
    candidates += glob.glob(os.path.join(Sim_Path, 'Et*.[vV][tT][uU]'))
    # HDF5 fallback (ParaView can open via appropriate readers/plugins)
    candidates += glob.glob(os.path.join(Sim_Path, 'Et*.[hH]5'))

    if candidates:
        candidates.sort()
        first = candidates[0]
        print(f'[info] Opening in ParaView: {first}')
        os.system(f'paraview.exe "{first}"')
    else:
        print('[warn] No field dump files found to open in ParaView.')
