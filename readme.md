openEMS Antenna Analysis Scripts

Overview
- This repository contains Python scripts to build and analyze microstrip patch antenna structures using openEMS/CSXCAD.
- The primary entry points are the scripts whose filenames contain the word "analysis":
  - patch_line_analysis.py — parametric line array of patches built procedurally.
  - patch_line_analysis_with_feed.py — analysis using STL-based geometry with a defined lumped-port feed.
  - patch_analysis.py — single-patch analysis with optional insets and quarter-wave matching.

Supporting modules:
- stl_model.py — builds CSXCAD materials from STL files for the with_feed workflow.
- extract_edges_from_stl.py — extracts axis-aligned edges from STL into CSV meshlines for grid snapping.
- apply_meshlines_from_csv.py — reads meshlines_*.csv and applies them to the CSXCAD grid.
- utils.py — post-processing helpers: HDF5 reading, TD→FD conversion, Smith chart, field plots, and more.

Prerequisites
- Python 3.8+ recommended.
- openEMS with Python bindings and CSXCAD Python module available in your Python environment.
- Additional Python packages:
  - numpy, matplotlib, h5py, scikit-rf, trimesh

Install Python dependencies
- If you have pip:
  - pip install numpy matplotlib h5py scikit-rf trimesh

Notes on openEMS/CSXCAD
- Install openEMS and CSXCAD per the official documentation for your platform.
- Ensure the Python modules (openEMS, CSXCAD) are importable (PYTHONPATH updated or installed site-wide).
- On Windows, you may need to add the directories containing the DLLs to your PATH.

How to run the analysis scripts (Windows cmd.exe)
- Run the procedural patch-line analysis:
  - python patch_line_analysis.py
- Run the STL-driven patch-line analysis with feed:
  - python patch_line_analysis_with_feed.py
- Run the single patch analysis:
  - python patch_analysis.py

What each analysis script does
1) patch_line_analysis.py
- Builds a linear array of rectangular patches on a dielectric substrate procedurally (no STL).
- Adds a feed network and runs FDTD simulation.
- Exports PDF plots:
  - S11, Smith chart
  - Field slices (Ez absolute, snapshots)
  - Surface currents (Js, Jx, Jy)
  - Directivity (polar and Cartesian), if enabled
- Output PDFs are saved next to the script (in this repository directory).

2) patch_line_analysis_with_feed.py
- Uses STL geometry (in line_array_openEMS_simulation/) to construct materials/objects via stl_model.py.
- Automatically extracts axis-aligned edges from the STL to generate grid meshlines:
  - Uses extract_edges_from_stl.py to write meshlines_x.csv, meshlines_y.csv, meshlines_z.csv.
  - Uses apply_meshlines_from_csv.py to add those lines to the CSXCAD grid.
- Sets up a lumped port in the specified position and runs the FDTD simulation.
- Produces the same suite of plots and saves them as a timestamped PDF in this directory.

3) patch_analysis.py
- Single rectangular patch with optional insets and quarter-wave transformer.
- Useful for parameter sweeps over width/length and plotting resonance and resistance trends.
- Produces summary result PDFs in this directory.

Common configuration switches
- At the top of each analysis script:
  - enable_simulation: 1 to run FDTD; 0 to only post-process existing results in temp directories.
  - save_to_pdf: 1 to save all figures to a single PDF; 0 to show plots interactively.
  - draw_CAD: 1 to write CSX XML and open AppCSXCAD for preview; 0 to skip.
  - draw_* flags: control which plots to generate (S11, Smith, fields, currents, directivity).
- Simulation outputs are stored in a temp folder (e.g., %TEMP%\patch_line_analysis_...).
- Generated PDFs are saved next to the script.

STL-based workflow details (patch_line_analysis_with_feed.py)
- Expects STL files in ./line_array_openEMS_simulation/:
  - top_gen_model.stl
  - bottom_gen_model.stl
  - substrate_gen_model.stl
- The script calls extract_edges_from_stl.main(["<stl_path>"]) for each relevant STL to produce:
  - meshlines_x.csv, meshlines_y.csv, meshlines_z.csv (in the current working directory)
- Mesh lines are applied (scaled if needed) using apply_meshlines_from_csv.
- Units:
  - The code treats the meshlines directly as the same units as your model settings (default millimeter delta unit is set via mesh.SetDeltaUnit(1e-3)).

Outputs
- Each analysis script writes a PDF summary beside the script file. Examples:
  - patch_line_analysis.py: "feed_length=... length=...mm width=...mm patch_number=... results.pdf"
  - patch_line_analysis_with_feed.py: "<UTC timestamp> results.pdf"
  - patch_analysis.py: parameterized filenames reflecting the sweep settings
- Intermediate HDF5 field dumps (Et.h5, Jt.h5) are written in the temp simulation directory and used by utils.py for post-processing.

Tuning the mesh and resolution
- Resolution is set by mesh_res = C0 / (f0 + fc) / 1e-3 / 20 and further refined around feeds.
- With STL geometry, meshlines derived from axis-aligned edges help snap the grid to important boundaries.
- Adjust mesh.SmoothMeshLines('all', mesh_res, <ratio>) to relax/refine transitions.

Troubleshooting
- ImportError: openEMS or CSXCAD
  - Ensure openEMS and CSXCAD are installed and their Python modules are importable.
  - Verify PATH/PYTHONPATH (Windows) includes directories with required DLLs and Python packages.
- ImportError: scikit-rf or trimesh
  - Install via pip: pip install scikit-rf trimesh
- Missing Et.h5 / Jt.h5 when enable_simulation=0
  - Re-run with enable_simulation=1 to regenerate results, or ensure the temp directory has the required HDF5 files.
- No STL meshlines CSV files appear
  - Ensure extract_edges_from_stl runs and current working directory is writable.
  - Check the STL paths and that they are valid meshes.

Repository map
- patch_line_analysis.py — main analysis (procedural patch line)
- patch_line_analysis_with_feed.py — main analysis (STL-driven)
- patch_analysis.py — main analysis (single patch)
- stl_model.py — materials/geometry from STL
- extract_edges_from_stl.py — CLI to extract axis-aligned edge coordinates into CSV
- apply_meshlines_from_csv.py — add CSV meshlines into CSXCAD grid
- utils.py — post-processing and plotting helpers

Quick start (Windows)
- Ensure dependencies and openEMS/CSXCAD are available in Python.
- Run one of the analysis scripts:
  - python patch_line_analysis.py
  - python patch_line_analysis_with_feed.py
  - python patch_analysis.py
- Open the generated PDF in this directory to review the results.
