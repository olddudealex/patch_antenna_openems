### Import Libraries
import os, tempfile

import numpy as np
from pylab import *

from CSXCAD import ContinuousStructure
from openEMS import openEMS
from openEMS.physical_constants import *

from utils import *

from matplotlib.backends.backend_pdf import PdfPages

# 1 = enable / 0 = disable different plots
draw_CAD = 0  # Show 3D model before simulation
draw_CAD_exit = 0  # Abort execution after displaying 3D model

# 1=enable / 0=disable simulation (can be used to draw plots without running simulation)
enable_simulation = 1  # temporary dirs must contain data for plots when enable_simulation=0
save_to_pdf = 1  # prints all the result to the pdf, doesn't show any plots interactively

draw_complex_impedance = 1  # Show impedance Re/Im plots - impedance plot
draw_s11 = 1  # Show S11 plot - impedance plot
draw_smith_chart = 1  # Show Smith Chart           - impedance plot

draw_Ez_absolute = 1  # Show Ez electromagnetic field slice
draw_Ez_snap = 1

draw_Js_absolute = 1  # Show the surface current density of the patch
draw_Jx = 1
draw_Jy = 1

draw_directivety_polar = 1  # Show directivity polar plot - radiation pattern
draw_directivity_db = 1  # Show directivity dB plot    - radiation pattern
draw_3d_pattern = 1  # Show antenna 3D pattern     - radiation pattern

draw_vtk_field = 1  # Export E-field to VTK for ParaView visualization
vtk_phase_step_deg = 10  # Phase step in degrees for VTK animation (default: 10)

# patch width (resonant length) in x-direction
feed_lengths = []
resistances = []
conductances = []
frequencies = []

# setup feeding
feed_width = 3.9  # mm
feed_R = 150

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

# antenna dimensions
antenna_length_max = wavelength_substrate * (patch_number+0.5)  # This length used for substrate length

# substrate dimensions
substrate_width = wavelength_freespace
substrate_length = antenna_length_max + wavelength_freespace/2
substrate_thickness = 1.524

# Air gap(number of free space 1/4-wavelengths)
air_gap = 2

# size of the simulation box
sim_box = [substrate_width+wavelength_freespace/4*air_gap,
           substrate_length+wavelength_freespace/4*air_gap,
           wavelength_freespace*2/4*air_gap]

# geometrical parameters
patch_length = 14.1  # mm
patch_width = 19  # mm

# feed length between the patches
feed_length = 14.1  # mm

# matching stub
feed_length_0 = 10.2  # mm; = 14.1 - 3.9
matching_length_1 = 0  # mm
matching_length_2 = 0  # mm
matching_length_3 = 0  # mm, it's just input feed line, not for matching
matching_width = 3.9  # mm

########################################################################################

# the array of rectangulars that could be used as mask for visualization
rects_mm = []

# General parameter setup
Sim_Path = os.path.join(tempfile.gettempdir(), f"patch_line_analysis_{patch_length}_{patch_width}_"
                                               f"{patch_number}_{feed_length}")

# FDTD setup
FDTD = openEMS(NrTS=60000, EndCriteria=1e-5)
FDTD.SetGaussExcite(f0, fc)
FDTD.SetBoundaryCond(['PML_8', 'PML_8', 'PML_8', 'PML_8', 'PML_8', 'PML_8'])

CSX = ContinuousStructure()
FDTD.SetCSX(CSX)
mesh = CSX.GetGrid()
mesh.SetDeltaUnit(1e-3)
mesh_res = C0 / (f0 + fc) / 1e-3 / 20

# Generate properties, primitives and mesh-grid
# initialize the mesh with the "air-box" dimensions
mesh.AddLine('x', [-sim_box[0] / 2, sim_box[0] / 2])
mesh.AddLine('y', [-sim_box[1] / 2, sim_box[1] / 2])
mesh.AddLine('z', [-sim_box[2] / 3, sim_box[2] * 2 / 3])

# create substrate
substrate = CSX.AddMaterial('substrate', epsilon=substrate_epsR, kappa=substrate_kappa)
start = [-substrate_width / 2, -substrate_length / 2, 0]
stop = [substrate_width / 2, substrate_length / 2, substrate_thickness]
substrate.AddBox(priority=0, start=start, stop=stop)

# create patch and microstrips
patch = CSX.AddMetal('patch')  # create a perfect electric conductor (PEC)

feed = CSX.AddMetal('feed')
y_shift_first = (patch_length + feed_length) * (patch_number - 1) / 2
coord_precision = 3  # down to 0.001mm

start = [-feed_width / 2, -y_shift_first + patch_length / 2, substrate_thickness]
stop = [feed_width / 2, y_shift_first - patch_length / 2, substrate_thickness]
feed.AddBox(priority=11, start=start, stop=stop)
#FDTD.AddEdges2Grid(dirs='x', properties=feed, metal_edge_res=mesh_res / 4)
#FDTD.AddEdges2Grid(dirs='x', properties=feed, metal_edge_res=None)
mesh.AddLine('x', 0)

for patch_index in range(patch_number):
    y_shift = y_shift_first - (patch_length + feed_length) * patch_index

    # --- patch coordinates ---
    start = np.round([-patch_width / 2,
                      -patch_length / 2 - y_shift,
                      substrate_thickness], coord_precision)
    stop = np.round([patch_width / 2,
                     patch_length / 2 - y_shift,
                     substrate_thickness], coord_precision)

    patch.AddBox(priority=10, start=start.tolist(), stop=stop.tolist())
    FDTD.AddEdges2Grid(dirs='xy', properties=patch, metal_edge_res=None)  # use mesh exact alignment to patch edges

    rects_mm.append({
        "x1": float(start[0]), "x2": float(stop[0]),
        "y1": float(start[1]), "y2": float(stop[1])
    })

network = CSX.AddMetal('network')

# add matching scheme in the beginning
reference_surface_y = -y_shift_first - patch_length / 2 - feed_length_0

start = [-feed_width / 2, reference_surface_y + feed_length_0, substrate_thickness]
stop = [feed_width / 2, reference_surface_y, substrate_thickness]
network.AddBox(priority=11, start=start, stop=stop)

start = [-matching_width / 2, reference_surface_y, substrate_thickness]
stop = [matching_width / 2, reference_surface_y - matching_length_1 - matching_width - matching_length_3,
        substrate_thickness]
network.AddBox(priority=11, start=start, stop=stop)

start = [matching_width / 2, reference_surface_y - matching_length_1, substrate_thickness]
stop = [matching_width / 2 + matching_length_2, reference_surface_y - matching_length_1 - matching_width,
        substrate_thickness]
network.AddBox(priority=11, start=start, stop=stop)
FDTD.AddEdges2Grid(dirs='x', properties=network, metal_edge_res=mesh_res / 4)
#FDTD.AddEdges2Grid(dirs='y', properties=network, metal_edge_res=mesh_res / 2)
FDTD.AddEdges2Grid(dirs='xy', properties=network, metal_edge_res=None)

# create ground (same size as substrate)
gnd = CSX.AddMetal('gnd')  # create a perfect electric conductor (PEC)
start = [-substrate_width/2, -substrate_length/2, 0]
stop = [substrate_width/2, substrate_length/2, 0]
gnd.AddBox(start, stop, priority=10)
FDTD.AddEdges2Grid(dirs='xy', properties=gnd)

# apply the excitation & resist as a current source
feed_pos = reference_surface_y - matching_length_1 - matching_width - matching_length_3
start = [-matching_width/2, feed_pos, 0]
stop = [matching_width/2, feed_pos, substrate_thickness]
port = FDTD.AddLumpedPort(1, feed_R, start, stop, 'z', 1.0,
                          priority=5,
                          #edges2grid='xy'
                          )

# add extra cells to discretize the substrate thickness
mesh.AddLine('z', linspace(0, substrate_thickness, substrate_cells + 1))

# Add mesh for feeding port
res = mesh_res / 4  # Finer resolution for feeding line
mesh.AddLine('y', [ feed_pos, feed_pos - res, feed_pos + res])

# increase density
mesh.SmoothMeshLines('all', mesh_res, 1.4)

# Add the nf2ff recording box
nf2ff = FDTD.CreateNF2FFBox()

# Add the dumping of E field
et = CSX.AddDump('Et', dump_type=0, file_type=1)
# Save a big box including substrate + air above
et.AddBox(start=[-sim_box[0] / 2, -sim_box[1] / 2, -sim_box[2] / 3],
          stop=[sim_box[0] / 2, sim_box[1] / 2, sim_box[2] * 2 / 3])
# (Optional) also H-field or current density:
# ht = CSX.AddDump('Ht', dump_type=1, file_type=file_type); ht.AddBox(start=..., stop=...)
jt = CSX.AddDump('Jt', dump_type=3, file_type=1)
jt.AddBox(start=[-sim_box[0] / 2, -sim_box[1] / 2, -sim_box[2] / 3],
          stop=[sim_box[0] / 2, sim_box[1] / 2, sim_box[2] * 2 / 3])

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
filename = (f"feed_length={feed_length} length={patch_length}mm width={patch_width}mm "
            f"patch_number={patch_number} results.pdf")
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

touchstone_dir = os.path.join(script_dir, "touchstone")
os.makedirs(touchstone_dir, exist_ok=True)
s1p_filename = f"patch_line_{patch_length}mm_{patch_width}mm_{feed_length}mm_{feed_width}mm.s1p"
s1p_filename = s1p_filename.replace('.', '_').replace('_.s1p', '.s1p')
s1p_path = os.path.join(touchstone_dir, s1p_filename)
write_s1p_file(s1p_path, freq, feed_R, s11)

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
    patch_y_edges = [rect["y1"] for rect in rects_mm] + [rect["y2"] for rect in rects_mm]

if draw_Ez_absolute:
    # 2D plot with your custom projection
    pc, info2d = plot_ez_2d(
        fd, E_fd, z_value=0.0008,
        func=lambda Ez: np.abs(Ez),
        func_str="|Ez|",
        cmap="jet",
        #clim=(2, 12),
        #rects_mm=rects_mm,
        outside_color="lightgray"
    )
    finalize_plot()

    # 1D line cut along Y
    line, info1d = plot_ez_line_y(
        fd, E_fd, z_value=0.0008,
        func=lambda Ez: np.abs(Ez),
        func_str="|Ez|",
        #clim=(2, 12),
        x_value=None, y_lines_mm=np.array(patch_y_edges)
    )
    finalize_plot()

if draw_Ez_snap:
    ix, iy1, iz = coord_to_index(x=0, y=feed_pos/1000, z=0.0008, fd=fd)
    _, iy2, _ = coord_to_index(x=0, y=(-feed_pos-feed_length)/1000, z=0.0008, fd=fd)

    reference_line = E_fd[ix, iy1:iy2, iz, 2]
    dphi = find_max_ampl_phase(reference_line)

    E_snap = np.real(E_fd * np.exp(-1j * dphi))

    # 2D plot with your custom projection
    pc, info2d = plot_ez_2d(
        fd, E_snap, z_value=0.0008,
        func=lambda Ez: Ez,
        func_str=f"Ez snapshot (dphi={dphi*180/np.pi:.2f}deg)",
        cmap="jet",
        #rects_mm=rects_mm,
        outside_color="lightgray",
        #clim=(-12, 4)
    )
    finalize_plot()

    # 1D line cut along Y
    line, info1d = plot_ez_line_y(
        fd, E_snap, z_value=0.0008,
        func=lambda Ez: Ez,
        func_str=f"Ez snapshot (dphi={dphi*180/np.pi:.2f}deg)",
        #clim=(-12, 4),
        x_value=None,
        y_lines_mm=np.array(patch_y_edges)
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
if draw_directivety_polar or draw_directivity_db:
    theta = np.arange(-180.0, 180.0, 2.0)
    phi = [0., 90.]
    nf2ff_res = nf2ff.CalcNF2FF(Sim_Path, freq[freqInd], theta, phi, center=[0, 0, 1e-3])

if draw_directivety_polar:
    plot_directivity_db_polar(theta, nf2ff_res, freq[freqInd])
    finalize_plot()

if draw_directivity_db:
    plot_directivity_db(theta, nf2ff_res, freq[freqInd])
    finalize_plot()

# For 3D plot, need full phi range
if draw_3d_pattern:
    theta_3d = np.arange(0.0, 181.0, 5.0)  # 0 to 180 degrees
    phi_3d = np.arange(0.0, 360.0, 5.0)    # 0 to 360 degrees
    nf2ff_res_3d = nf2ff.CalcNF2FF(Sim_Path, freq[freqInd], theta_3d, phi_3d, center=[0, 0, 1e-3])
    plot_directivity_3d(theta_3d, phi_3d, nf2ff_res_3d, freq[freqInd])
    finalize_plot()

    # Save interactive HTML version
    html_filename = (f"feed_length={feed_length} length={patch_length}mm width={patch_width}mm "
                    f"patch_number={patch_number} 3D_pattern.html")
    html_path = os.path.join(reports_dir, html_filename)
    plot_directivity_3d_interactive(theta_3d, phi_3d, nf2ff_res_3d, freq[freqInd], html_path)

# Export E-field to VTK for ParaView visualization
if draw_vtk_field:
    vtk_dir = os.path.join(reports_dir, f"vtk_feedL={feed_length}_L={patch_length}mm_W={patch_width}mm_N={patch_number}")
    export_efield_vtk_at_frequency(
        hdf5_path=os.path.join(Sim_Path, 'Et.h5'),
        target_freq=f0,
        output_dir=vtk_dir,
        phase_step_deg=vtk_phase_step_deg
    )

# If you still want to keep those tracking arrays:
feed_lengths.append(feed_length)
resistances.append(patchR2)
conductances.append(patchG2_norm)
frequencies.append(meta["freq_res_hz"]/1e9 if meta["freq_res_hz"] else None)

if save_to_pdf:
    pdf.close()
else:
    plt.show()
