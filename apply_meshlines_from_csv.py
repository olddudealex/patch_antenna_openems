#!/usr/bin/env python3
# apply_meshlines_from_csv.py
# Read meshlines CSVs and add them to a CSXCAD grid: mesh.AddLine('x'|'y'|'z', coords)

import os
import numpy as np

def _load_csv_maybe_empty(path: str) -> np.ndarray:
    """Load 1D float array from CSV. Returns empty array if file doesn't exist."""
    if not os.path.isfile(path):
        return np.array([], dtype=float)
    # ndmin=1 ensures a 1D array even if the file has a single value
    return np.loadtxt(path, dtype=float, ndmin=1)

def _prepare_coords(vals, scale: float) -> list[float]:
    """Scale, dedupe, sort, return as plain Python list."""
    vals = np.asarray(vals, dtype=float).ravel()
    if vals.size == 0:
        return []
    vals = vals * scale
    # sort + unique with a small rounding to avoid floating fuzz duplicates
    vals = np.unique(np.round(vals, 12))
    return vals.tolist()

def apply_meshlines_from_csv(csx, prefix: str = "meshlines", scale: float = 1.0):
    """
    Create a grid in CSXCAD and add lines from CSVs:
      <prefix>_x.csv, <prefix>_y.csv, <prefix>_z.csv

    Args:
        csx   : CSXCAD.ContinuousStructure (or CSX, depending on your setup)
        prefix: file prefix used when saving the CSVs
        scale : multiply coordinates by this factor before adding to mesh
                e.g., if CSV is in mm and openEMS expects meters -> scale=1e-3

    Returns:
        mesh  : the created grid object (so you can keep configuring it)
    """
    # Lazy import so this file can be reused without CSXCAD installed
    import CSXCAD

    x_path = f"{prefix}_x.csv"
    y_path = f"{prefix}_y.csv"
    z_path = f"{prefix}_z.csv"

    x_vals = _prepare_coords(_load_csv_maybe_empty(x_path), scale)
    y_vals = _prepare_coords(_load_csv_maybe_empty(y_path), scale)
    z_vals = _prepare_coords(_load_csv_maybe_empty(z_path), scale)

    # Create (or get) the grid and add lines
    mesh = csx.GetGrid()

    if x_vals:
        mesh.AddLine('x', x_vals)
    if y_vals:
        mesh.AddLine('y', y_vals)
    if z_vals:
        pass
        # mesh.AddLine('z', z_vals)

    return mesh
