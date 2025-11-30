### Import Libraries
import datetime
import os, tempfile

import numpy as np
from pylab import *

from CSXCAD import ContinuousStructure
from openEMS import openEMS
from openEMS.physical_constants import *
from datetime import datetime

from utils import *

from matplotlib.backends.backend_pdf import PdfPages

from stl_model import build_csx
from apply_meshlines_from_csv import apply_meshlines_from_csv
import extract_edges_from_stl

# 1 = enable / 0 = disable different plots
draw_CAD = 0  # Show 3D model before simulation
draw_CAD_exit = 0  # Abort execution after displaying 3D model

# 1=enable / 0=disable simulation (can be used to draw plots without running simulation)
enable_simulation = 0  # temporary dirs must contain data for plots when enable_simulation=0
save_to_pdf = 1  # prints all the result to the pdf, doesn't show any plots interactively

draw_complex_impedance = 1  # Show impedance Re/Im plots - impedance plot
draw_s11 = 1  # Show S11 plot - impedance plot
draw_smith_chart = 1  # Show Smith Chart           - impedance plot

draw_Ez_absolute = 1  # Show Ez electromagnetic field slice
draw_Ez_snap = 1

draw_Js_absolute = 1  # Show the surface current density of the patch
draw_Jx = 1
draw_Jy = 1

draw_directivity_polar_db = 1  # Show directivity polar db plot - radiation pattern
draw_directivity_polar = 1  # Show directivity polar plot - radiation pattern
draw_directivity_db = 1  # Show directivity dB plot    - radiation pattern
draw_3d_pattern = 1  # Show antenna 3D pattern     - radiation pattern

# setup feeding
feed_width = 3.85
feed_R = 50

# frequency of interest
f0 = 5.8e9  # center frequency
fc = 0.5e9  # 20 dB corner frequency

# substrate setup
substrate_epsR = 3  # ZYF300CA-P
substrate_tan = 0.0018   # ZYF300CA-P
substrate_kappa = substrate_tan * 2 * pi * f0 * EPS0 * substrate_epsR
substrate_cells = 4

# wavelengths
c0 = 299792458  # m/s
wavelength_freespace = 1000 * c0 / f0
wavelength_substrate = wavelength_freespace / sqrt(substrate_epsR)

# patches
patch_number = 5
patch_width = 20.1
patch_length = 10.7

# antenna dimensions
antenna_length_max = wavelength_substrate*patch_number  # This length used for substrate length

# substrate dimensions
substrate_width = wavelength_freespace
substrate_length = antenna_length_max + wavelength_freespace/2
substrate_thickness = 1.524 + 0.001  # todo: this is hack!

# Air gap(number of free space 1/4-wavelengths)
air_gap = 2

# size of the simulation box
sim_box_start = [-20, -20, -10]
sim_box_stop = [58, 200, 25]

# General parameter setup
Sim_Path = os.path.join(tempfile.gettempdir(), f"patch_line_analysis")

# FDTD setup
FDTD = openEMS(NrTS=60000, EndCriteria=1e-5)
FDTD.SetGaussExcite(f0, fc)
FDTD.SetBoundaryCond(['PML_8', 'PML_8', 'PML_8', 'PML_8', 'PML_8', 'PML_8'])

CSX = build_csx(substrate_epsR, substrate_kappa)
FDTD.SetCSX(CSX)

# apply geometry specific mesh lines
freecad_dir = "./freecad_macro_output"
extract_edges_from_stl.main([f"{freecad_dir}/top_gen_model.stl"])
mesh = apply_meshlines_from_csv(CSX, prefix="meshlines/meshlines", scale=1)

extract_edges_from_stl.main([f"{freecad_dir}/substrate_gen_model.stl"])
mesh = apply_meshlines_from_csv(CSX, prefix="meshlines/meshlines", scale=1)

# add a couple of central mesh lines to feed line and patches
mesh.AddLine('x', [6.3, 25])

mesh.SetDeltaUnit(1e-3)
mesh_res = C0 / (f0 + fc) / 1e-3 / 20

# Generate properties, primitives and mesh-grid
# initialize the mesh with the "air-box" dimensions
mesh.AddLine('x', [sim_box_start[0], sim_box_stop[0]])
mesh.AddLine('y', [sim_box_start[1], sim_box_stop[1]])
mesh.AddLine('z', [sim_box_start[2], sim_box_stop[2]])

coord_precision = 3  # down to 0.001mm

#  apply the excitation & resist as a current source
feed_pos_y = 0
feed_pos_x = 8.725
start = [-feed_width / 2 + feed_pos_x, feed_pos_y, -0.001]
stop = [feed_width / 2 + feed_pos_x, feed_pos_y + 1.0, substrate_thickness]
port = FDTD.AddLumpedPort(1, feed_R, start, stop, 'z', 1.0,
                          priority=15,
                          #edges2grid='xy'
                          )

# add extra cells to discretize the substrate thickness
mesh.AddLine('z', linspace(-0.001, substrate_thickness, substrate_cells + 1))

# Add mesh for feeding port
res = mesh_res / 8  # Finer resolution for feeding line
mesh.AddLine('y', [feed_pos_y, feed_pos_y - res, feed_pos_y + res])

# increase density
mesh.SmoothMeshLines('all', mesh_res, 1.4)

# Add the nf2ff recording box
nf2ff = FDTD.CreateNF2FFBox(start=[2, 0, -2], stop=[36, 178, 15])

# Add the dumping of E field
et = CSX.AddDump('Et', dump_type=0, file_type=1)
# Save a big box including substrate + air above
et.AddBox(start=sim_box_start,
          stop=sim_box_stop)
# (Optional) also H-field or current density:
# ht = CSX.AddDump('Ht', dump_type=1, file_type=file_type); ht.AddBox(start=..., stop=...)
jt = CSX.AddDump('Jt', dump_type=3, file_type=1)
jt.AddBox(start=sim_box_start,
          stop=sim_box_stop)

# Run the simulation
if draw_CAD:  # debugging only
    CSX_file = os.path.join(Sim_Path, 'patch_line.xml')
    if not os.path.exists(Sim_Path):
        os.mkdir(Sim_Path)
    CSX.Write2XML(CSX_file)
    from CSXCAD import AppCSXCAD_BIN

    os.system(AppCSXCAD_BIN + ' "{}"'.format(CSX_file))

if draw_CAD_exit:
    exit()

if enable_simulation:
    FDTD.Run(Sim_Path, verbose=3, cleanup=True, numThreads=4)

# ###########################################################
# Post-processing and plotting
# ###########################################################

# directory of the script file itself
script_dir = os.path.dirname(os.path.abspath(__file__))
reports_dir = os.path.join(script_dir, "reports")
os.makedirs(reports_dir, exist_ok=True)
timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
filename = f"{timestamp} results.pdf"
pdf_path = os.path.join(reports_dir, filename)
pdf = PdfPages(pdf_path) if save_to_pdf else None

def finalize_plot(pdf=pdf, save_to_pdf=save_to_pdf):
    """Save or show the current Matplotlib figure depending on mode."""
    if save_to_pdf and pdf is not None:
        pdf.savefig()
        plt.close()

freq = np.linspace(f0 - fc, f0 + fc, 401)
port.CalcPort(Sim_Path, freq)

# ##################################
# Plot Complex Impedance
# ##################################
if draw_complex_impedance:
    Zin = port.uf_tot / port.if_tot
    figure()
    plot(freq / 1e9, np.abs(Zin), 'k-', label='Zin real')
    plot(freq / 1e9, np.imag(Zin), 'r--', label='Zin imag')
    grid()
    legend()
    ylabel('Ohm')
    xlabel('Frequency (GHz)')
    finalize_plot()

# ##################################
# Plot S11 (dB)
# ##################################
s11 = port.uf_ref / port.uf_inc
s11_dB = 20.0 * np.log10(np.abs(s11))
if draw_s11:
    plot_s11(freq, s11_dB)
    finalize_plot()

# Required frequency 5.8GHz
freqInd = freq.shape[0] // 2
s11_1 = s11[freqInd]

# Find minimum on S11 graph
freqInd2 = int(np.argmin(np.abs(s11)))
s11_2 = s11[freqInd2]

# Find patch equivalent resistance R and normalized conductance G
# at required/actual resonant frequencies
patchR = (1 + s11_1) / (1 - s11_1) * feed_R
patchG_norm = (1 - s11_1) / (1 + s11_1)

patchR2 = (1 + s11_2) / (1 - s11_2) * feed_R
patchG2_norm = (1 - s11_2) / (1 + s11_2)

# ##################################
# Plot Smith Chart
# ##################################
if draw_smith_chart:
    ax, meta = plot_smith_skrf(
        s11, freq_hz=freq,
        idx_main=freqInd, idx_res=freqInd2,
        patch_width_mm=patch_width, patch_length_mm=patch_length,
        patchR=patchR, patchG=patchG_norm, patchR2=patchR2, patchG2=patchG2_norm,
    )
    finalize_plot()

# ###############
# Plot Ez field
# ###############
if draw_Ez_absolute or draw_Ez_snap:
    fd = read_hdf5_dump(f"{Sim_Path}/Et.h5")
    E_fd = td_to_fd_dft(fd.F_td, fd.time, fd.dt, freq[freqInd])  # -> (Nx, Ny, Nz, 3)

if draw_Ez_absolute:
    # 2D plot with your custom projection
    pc, info2d = plot_ez_2d(
        fd, E_fd, z_value=0.0008,
        func=lambda Ez: np.abs(Ez),
        func_str="|Ez|",
        cmap="jet",
        outside_color=None,
        clim=(4.8, 5.3)
    )
    finalize_plot()

    # 1D line cut along Y
    line, info1d = plot_ez_line_y(
        fd, E_fd, z_value=0.0008,
        func=lambda Ez: np.abs(Ez),
        func_str="|Ez|",
        #clim=(2, 12),
        x_value=0.008725
    )
    finalize_plot()

if draw_Ez_snap:
    # 2D plot with your custom projection
    pc, info2d = plot_ez_2d(
        fd, real(E_fd), z_value=0.0008,
        func=lambda Ez: np.real(Ez),
        func_str=f"Real E_fd",
        cmap="jet",
        outside_color=None,
        clim=(4.8, 5.3)
    )
    finalize_plot()

    # 1D line cut along Y
    line, info1d = plot_ez_line_y(
        fd, real(E_fd), z_value=0.0008,
        func=lambda Ez: np.real(Ez),
        func_str=f"Real E_fd",
        #clim=(-12, 4),
        x_value=0.008725
    )
    finalize_plot()

if draw_Js_absolute or draw_Jx or draw_Jy:
    fd = read_hdf5_dump(f"{Sim_Path}/Jt.h5")
    J_fd = td_to_fd_dft(fd.F_td, fd.time, fd.dt, freq[freqInd])  # -> (Nx, Ny, Nz, 3)

if draw_Js_absolute:
    # 2D plot with your custom projection
    pc, info2d = plot_js_2d(
        fd, J_fd, z_value=substrate_thickness/1000,
        func=lambda Jx, Jy, Jz: np.sqrt((Jx * np.conj(Jx) + Jy * np.conj(Jy)).real),
        func_str="|Js|",
        cmap="jet",
        #clim=(0, 50)
    )
    finalize_plot()

if draw_Jx:
    # 2D plot with your custom projection
    pc, info2d = plot_js_2d(
        fd, J_fd, z_value=substrate_thickness/1000,
        func=lambda Jx, Jy, Jz: np.abs(Jx),
        func_str="|Jx|",
        cmap="jet",
        #clim=(0, 50)
    )
    finalize_plot()

if draw_Jy:
    # 2D plot with your custom projection
    pc, info2d = plot_js_2d(
        fd, J_fd, z_value=substrate_thickness/1000,
        func=lambda Jx, Jy, Jz: np.abs(Jy),
        func_str="|Jy|",
        cmap="jet",
        #clim=(0, 50)
    )
    finalize_plot()


# ################
# Plot Directivity
# ################
if draw_directivity_polar_db or draw_directivity_polar or draw_directivity_db:
    theta = np.arange(-180.0, 180.0, 2.0)
    phi = [0., 90.]
    nf2ff_res = nf2ff.CalcNF2FF(Sim_Path, freq[freqInd], theta, phi, center=[20, 85, 1e-3])

if draw_directivity_polar_db:
    plot_directivity_db_polar(theta, nf2ff_res, freq[freqInd])
    finalize_plot()

if draw_directivity_polar:
    plot_directivity_linear_polar(theta, nf2ff_res, freq[freqInd])
    finalize_plot()

if draw_directivity_db:
    plot_directivity_db(theta, nf2ff_res, freq[freqInd])
    finalize_plot()

# For 3D plot, need full phi range
if draw_3d_pattern:
    theta_3d = np.arange(0.0, 181.0, 5.0)  # 0 to 180 degrees
    phi_3d = np.arange(0.0, 360.0, 5.0)    # 0 to 360 degrees
    nf2ff_res_3d = nf2ff.CalcNF2FF(Sim_Path, freq[freqInd], theta_3d, phi_3d, center=[20, 85, 1e-3])
    plot_directivity_3d(theta_3d, phi_3d, nf2ff_res_3d, freq[freqInd])
    finalize_plot()

if save_to_pdf:
    pdf.close()
else:
    plt.show()
