# openEMS Microstrip Patch Antenna Array Simulation Framework

## Overview

This is an openEMS-based simulation and analysis framework for designing microstrip patch antenna arrays. It automates electromagnetic FDTD (Finite-Difference Time-Domain) simulations for linear arrays of rectangular patch antennas on dielectric substrates, with a focus on ~5.8 GHz applications for millimeter-wave radar systems.

### Key Capabilities
- Build patch antenna arrays procedurally or from STL 3D models
- Run FDTD electromagnetic simulations
- Extract S-parameters (reflection/transmission coefficients)
- Generate comprehensive analysis plots (impedance, radiation patterns, fields)
- Support parametric sweeps for antenna optimization
- Convert time-domain field data to frequency-domain representations
- Export results as multi-page PDF reports

---

## Project Structure

### Primary Analysis Scripts (Entry Points)

**[patch_line_analysis.py](patch_line_analysis.py)** - Procedural patch line array simulation
- Builds geometry directly in CSXCAD without STL
- Simulates a linear array of microstrip patches
- Default configuration: 5 elements, 14.1×19mm patches, 5.8 GHz
- Generates multi-page PDF with S11, Smith chart, field slices, surface currents, and directivity patterns

**[patch_line_analysis_with_feed.py](patch_line_analysis_with_feed.py)** - STL-based array with explicit feed port
- Loads CAD models (STL) and converts them to CSXCAD materials
- Automatic mesh line extraction from STL edges for accurate discretization
- Lumped port feed specification
- Geometry-driven mesh snapping for improved accuracy

**[patch_analysis.py](patch_analysis.py)** - Single-patch parameter sweeps
- Optimization via parametric length/width/inset sweeps
- Per-sweep PDF with resonance characteristics and impedance trends
- Supports quarter-wave matching transformers

**[patch_2port_analysis.py](patch_2port_analysis.py)** - Two-port network analysis
- Analyzes coupled patch pairs
- Generates S2P files for two-port network analysis

### Supporting Modules

**[utils.py](utils.py)** (1052 lines) - Core post-processing library
- **HDF5 Data I/O**: Parse openEMS time-domain field dumps
- **Frequency-Domain Conversion**: DFT and FFT-based spectrum extraction
- **Field Visualization**: 2D colormaps, line cuts, surface current density
- **Network Analysis**: Smith charts, S-parameter plots with bandwidth annotations
- **Parameter Conversions**: S↔ABCD↔Z0 transformations
- **File Export**: Touchstone file generation (.s1p, .s2p)
- **Radiation Patterns**: Directivity plots (linear/dB, polar/Cartesian)

**[stl_model.py](stl_model.py)** - CAD material builder
- Creates CSXCAD structure from STL files
- Loads materials from `freecad_macro_output/`:
  - `top_gen_model.stl` → PEC material (patches)
  - `bottom_gen_model.stl` → Ground plane
  - `substrate_gen_model.stl` → Dielectric
- Assigns material priorities for overlap handling

**[extract_edges_from_stl.py](extract_edges_from_stl.py)** - Mesh extraction tool
- Extracts axis-aligned edges from STL models
- Generates CSV files with mesh coordinates (x, y, z)
- Configurable angle and position tolerances
- Usage: `extract_edges_from_stl.main(["path/to/model.stl"])`

**[apply_meshlines_from_csv.py](apply_meshlines_from_csv.py)** - Grid configuration
- Applies extracted meshlines to CSXCAD grid
- Snaps FDTD grid points to STL geometry edges
- Supports coordinate scaling for unit conversion

---

## Technology Stack

| Component | Purpose |
|-----------|---------|
| **openEMS** | FDTD electromagnetic solver (core physics engine) |
| **CSXCAD** | 3D geometry CAD engine |
| **NumPy** | Numerical array operations and Fourier transforms |
| **Matplotlib** | 2D plotting and multi-page PDF export |
| **h5py** | HDF5 file I/O for field data |
| **scikit-rf** | Network parameter analysis and Smith charts |
| **trimesh** | 3D mesh processing for STL import |

---

## Prerequisites

### System Requirements
- **Python 3.8+** (3.10+ recommended)
- **OpenEMS & CSXCAD** installed with Python bindings
- Windows/Linux/macOS with proper PATH/PYTHONPATH setup

### Installation

```bash
# Install Python dependencies
pip install numpy matplotlib h5py scikit-rf trimesh

# Install openEMS and CSXCAD per official documentation
# Ensure Python modules are importable (update PYTHONPATH if needed)
# On Windows, add DLL directories to PATH
```

---

## Usage

### Quick Start

```bash
# Run procedural patch-line analysis
python patch_line_analysis.py

# Run STL-driven patch-line analysis with feed
python patch_line_analysis_with_feed.py

# Run single patch analysis
python patch_analysis.py
```

### Configuration Parameters

All scripts expose configurable parameters at the top:

```python
# Frequency
f0 = 5.8e9  # Hz - Center frequency
fc = 0.5e9  # Hz - Bandwidth (20 dB corner frequency)

# Substrate (ZYF300CA-P)
substrate_epsR = 3.0
substrate_tan = 0.0018
substrate_thickness = 1.524  # mm

# Patch Geometry
patch_length = 14.1  # mm - Resonant length
patch_width = 19     # mm
patch_number = 5     # Array element count
feed_length = 14.1   # mm
feed_width = 3.9     # mm
feed_R = 50          # Ohm - Feed impedance (50-150Ω typical)

# Output Controls
enable_simulation = 1  # 1: Run FDTD, 0: Post-process only
save_to_pdf = 1       # 1: Batch PDF, 0: Interactive display
draw_CAD = 0          # 1: Preview 3D geometry, 0: Skip
draw_S11 = 1          # Enable/disable specific plots
draw_smith_chart = 1
draw_fields = 1
```

---

## Workflow Pipeline

```
1. Load Configuration
   ↓
2. Build Geometry
   - Procedural: Direct CSXCAD objects
   - STL-based: Load STL → Extract meshlines → Apply grid snapping
   ↓
3. Configure FDTD Solver
   - Set boundary conditions (PML_8)
   - Set excitation (Gaussian pulse)
   - Configure mesh resolution
   ↓
4. Run FDTD Simulation (if enable_simulation=1)
   - Execute openEMS solver
   - Output: HDF5 field dumps (Et.h5, Jt.h5)
   ↓
5. Post-Processing (utils.py)
   - Read HDF5 dumps
   - Convert time-domain → frequency-domain
   - Extract S-parameters
   - Generate visualizations
   ↓
6. Export Results
   - Multi-page PDF report
   - S-parameter files (.s1p, .s2p)
   - Logs and metadata
```

---

## Output Files

### Generated Artifacts

**PDF Reports** - Multi-page analysis with:
- S11 reflection coefficient vs frequency
- Smith chart impedance plots
- E-field snapshots (2D slices)
- Surface current density (magnitude + components)
- Directivity patterns (if NF2FF enabled)

**S-Parameter Files** - Touchstone format (.s1p/.s2p):
- Compatible with RF analysis tools
- Frequency range: typically 4.5–7 GHz
- Example naming: `patch_line_14_1mm_19mm_14_1mm_1mm.s1p`

**Mesh Configuration** - CSV files in `meshlines/`:
- `meshlines_x.csv`, `meshlines_y.csv`, `meshlines_z.csv`
- Coordinate values for FDTD grid optimization

**Temporary Files** - Simulation directory:
- Location: `%TEMP%/patch_line_analysis_TIMESTAMP/`
- Contains: CSX XML, HDF5 field dumps, solver logs

---

## STL-Based Workflow Details

The `patch_line_analysis_with_feed.py` script uses STL geometry:

1. **Expected STL Files** in `./freecad_macro_output/`:
   - `top_gen_model.stl` - PEC patches
   - `bottom_gen_model.stl` - Ground plane
   - `substrate_gen_model.stl` - Dielectric layer
   - Generated by FreeCAD macro exports

2. **Mesh Line Extraction**:
   - Script calls `extract_edges_from_stl.main()` for each STL
   - Generates `meshlines_{x,y,z}.csv` in `meshlines/` directory
   - Axis-aligned edges extracted with configurable tolerances

3. **Grid Application**:
   - `apply_meshlines_from_csv()` reads from `meshlines/` directory
   - Supports coordinate scaling (e.g., mm → m with `scale=1e-3`)
   - Mesh snaps to geometry boundaries for accuracy

4. **Material Assignment**:
   - PEC (priority 10): Conductive patches
   - SUBSTRATE (priority 0): Dielectric layer
   - Ground plane: PEC material

---

## Antenna Design Specifications

### Default Configuration
```
Frequency: 5.8 GHz (C-band radar)
Substrate: ZYF300CA-P
  - Permittivity (εr): 3.0
  - Loss tangent (tan δ): 0.0018
  - Thickness: 1.524 mm
Patch: 14.1 × 19 mm rectangular
Array: 5 elements linear
Feed: 3.9–14.1 mm microstrip line
Impedance: 50–150 Ω (configurable)
Matching: Optional quarter-wave transformers
```

### Simulation Parameters
```
FDTD timesteps: 60,000
End criteria: 1e-5 (-50 dB energy decay)
Boundary conditions: PML_8 (6 faces)
Mesh resolution: λ/20 at f0+fc (~3.4 mm)
Excitation: Gaussian pulse (f0 ± fc)
```

---

## Data Flow Architecture

```
Analysis Script
    ↓
[Build Geometry] ──→ Procedural CSXCAD ──┐
    ↓                                     │
[STL Import] ──→ extract_edges_from_stl ──┤
    ↓            ↓                        │
stl_model.py   meshlines/*.csv           │
(reads from    (writes to meshlines/)    │
freecad_macro_ ↓                         │
output/)   apply_meshlines_from_csv ─────┘
    ↓
[FDTD Solver] ──→ openEMS ──→ HDF5 dumps (temp dir)
    ↓
[Post-Process] ──→ utils.py ──→ read_hdf5_dump()
    ↓                         ↓
    │                    td_to_fd_fft()
    ↓                         ↓
[Visualize] ────────────→ plot_s11(), plot_smith_skrf()
    ↓                    plot_ez_2d(), plot_js_2d()
    │                    plot_directivity_*()
    ↓                         ↓
[Export] ────────────────→ reports/*.pdf + touchstone/*.s1p/.s2p
```

---

## Troubleshooting

### Import Errors

**openEMS or CSXCAD not found**
- Ensure openEMS and CSXCAD are installed with Python bindings
- Verify PATH/PYTHONPATH includes required DLLs and modules (Windows)
- Test: `python -c "import openEMS, CSXCAD"`

**scikit-rf or trimesh not found**
- Install via pip: `pip install scikit-rf trimesh`

### Simulation Issues

**Missing Et.h5/Jt.h5 when enable_simulation=0**
- Re-run with `enable_simulation=1` to regenerate results
- Check temp directory (`%TEMP%/patch_line_analysis_*`) for HDF5 files

**No STL meshlines CSV files appear**
- Ensure `extract_edges_from_stl` runs successfully
- Check that `meshlines/` directory is created (auto-created by script)
- Verify STL file paths in `freecad_macro_output/` are valid
- Check that STL files contain proper meshes

**Simulation fails or crashes**
- Check mesh resolution (too fine → memory issues)
- Verify boundary conditions don't intersect geometry
- Review openEMS error logs in temp directory

---

## Design Patterns

### Configuration-Driven
- All tunable parameters at script top (not in functions)
- Boolean flags for workflow control (`draw_*`, `enable_simulation`)
- Easy parametric sweeps via script modification

### Modular Post-Processing
- Pure functions in `utils.py` (td_to_fd_*, plot_*, conversions)
- Decoupled from FDTD solver → can reuse cached results
- Flexible field slicing with geometry masking

### CAD Integration
- FreeCAD → STL export workflow
- Mesh line extraction for grid optimization
- Material priority system for overlap handling

---

## Project Status

This framework is under active development for antenna design research. The repository contains:
- Multiple parameter sweep campaigns (30+ result PDFs)
- Historical analysis archives in subdirectories
- Evolution from procedural to CAD-based workflows

Recent development focus:
- STL import support for complex geometries
- Mesh optimization and feed modeling
- Documentation and reproducibility improvements

---

## Repository Map

```
├── patch_line_analysis.py              # Main: procedural patch line
├── patch_line_analysis_with_feed.py    # Main: STL-driven analysis
├── patch_analysis.py                   # Main: single patch sweeps
├── patch_2port_analysis.py             # Supplementary: 2-port analysis
├── utils.py                            # Post-processing library
├── stl_model.py                        # STL → CSXCAD conversion
├── extract_edges_from_stl.py           # Mesh line extraction CLI
├── apply_meshlines_from_csv.py         # Grid configuration utility
│
├── freecad_macro_output/               # FreeCAD generated STL models
│   ├── top_gen_model.stl               # PEC patches
│   ├── bottom_gen_model.stl            # Ground plane
│   ├── substrate_gen_model.stl         # Dielectric substrate
│   ├── air_gen_model.stl               # Air volume (optional)
│   └── line_array_openEMS.py           # Auto-generated FreeCAD script
│
├── meshlines/                          # FDTD mesh coordinates
│   ├── meshlines_x.csv
│   ├── meshlines_y.csv
│   └── meshlines_z.csv
│
├── touchstone/                         # S-parameter exports
│   ├── *.s1p                           # Single-port measurements
│   └── *.s2p                           # Two-port measurements
│
└── reports/                            # Analysis PDF reports
    ├── best/                           # Best performing designs
    ├── Hor_Polarity_with_Individual_Feed_Lines/
    ├── Ver_Polarity_with_Individual_Feed_Lines/
    └── Ver_Polarity_with_One_Through_Feed_Line/
```

---

## Summary

This framework integrates openEMS FDTD simulation with comprehensive post-processing for microstrip patch antenna array design. It supports both procedural and CAD-based workflows, with automated S-parameter extraction, field visualization, and professional PDF reporting for parametric design studies and rapid prototyping.
