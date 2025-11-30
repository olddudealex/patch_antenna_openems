#!/usr/bin/env python3
"""
stl_axis_edges_to_meshlines_fixed.py
- Load STL (ASCII/binary)
- Find edges parallel to X/Y/Z (within angle tolerance)
- Output unique x, y, z coordinates for mesh snapping
"""

import argparse
import numpy as np
import trimesh


def unit(v):
    n = np.linalg.norm(v, axis=1, keepdims=True)
    n[n == 0.0] = 1.0
    return v / n


def dedupe(vals, tol=1e-6):
    """Return sorted unique values with tolerance."""
    vals = np.sort(np.asarray(vals).ravel())
    if vals.size == 0:
        return vals
    out = [vals[0]]
    for x in vals[1:]:
        if abs(x - out[-1]) > tol:
            out.append(x)
    return np.array(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Extract axis-aligned edge coordinates from STL")
    ap.add_argument("stl", help="Path to STL file (ASCII or binary)")
    ap.add_argument("--ang_tol_deg", type=float, default=1.0,
                    help="Angle tolerance (deg) for parallelism to axis (default 1°)")
    ap.add_argument("--pos_tol", type=float, default=1e-6,
                    help="Position dedupe tolerance (default 1e-6, same units as STL)")
    ap.add_argument("--out_prefix", default="meshlines/meshlines",
                    help="Prefix for CSV outputs (default: meshlines/meshlines)")
    ap.add_argument("--emit_python_snippet", action="store_true",
                    help="Print a Python snippet for openEMS grid")
    args = ap.parse_args(argv)

    # Load STL mesh
    mesh = trimesh.load(args.stl, force='mesh')
    if mesh.is_empty:
        raise SystemExit("Loaded STL is empty or invalid.")

    # Get unique edges and vertices
    eu = mesh.edges_unique  # (M, 2)
    vv = mesh.vertices      # (N, 3)

    # Compute edge vectors manually
    ev = vv[eu[:, 1]] - vv[eu[:, 0]]  # (M, 3)
    dir_u = unit(ev)

    # Determine axis-parallel edges
    ang = np.deg2rad(args.ang_tol_deg)
    cos_thresh = np.cos(ang)
    mask_x = np.abs(dir_u @ np.array([1, 0, 0])) >= cos_thresh
    mask_y = np.abs(dir_u @ np.array([0, 1, 0])) >= cos_thresh
    mask_z = np.abs(dir_u @ np.array([0, 0, 1])) >= cos_thresh

    # Collect all vertex coordinates belonging to axis-aligned edges
    vx, vy, vz = [], [], []

    def collect(mask):
        if np.any(mask):
            e = eu[mask]
            v = vv[e].reshape(-1, 3)
            return v[:, 0], v[:, 1], v[:, 2]
        return [], [], []

    for m in [mask_x, mask_y, mask_z]:
        x, y, z = collect(m)
        vx.extend(x)
        vy.extend(y)
        vz.extend(z)

    # Deduplicate
    x_lines = dedupe(vx, tol=args.pos_tol)
    y_lines = dedupe(vy, tol=args.pos_tol)
    z_lines = dedupe(vz, tol=args.pos_tol)

    # Write CSVs
    import os
    prefix_dir = os.path.dirname(args.out_prefix)
    if prefix_dir:
        os.makedirs(prefix_dir, exist_ok=True)
    np.savetxt(f"{args.out_prefix}_x.csv", x_lines, fmt="%.9g")
    np.savetxt(f"{args.out_prefix}_y.csv", y_lines, fmt="%.9g")
    np.savetxt(f"{args.out_prefix}_z.csv", z_lines, fmt="%.9g")

    print(f"[OK] Found {len(x_lines)} unique x, {len(y_lines)} y, {len(z_lines)} z lines")
    print(f"[OK] Saved: {args.out_prefix}_x.csv, _y.csv, _z.csv")

    if args.emit_python_snippet:
        print("\n# ---- Paste into your openEMS Python model ----")
        print("# If STL units are mm, multiply all below by 1e-3 before use.")
        def arr(a, scale=1.0):
            return "[" + ", ".join(f"{scale*v:.9g}" for v in a) + "]"
        print(f"x_lines = {arr(x_lines)}")
        print(f"y_lines = {arr(y_lines)}")
        print(f"z_lines = {arr(z_lines)}")
        print("# Example:")
        print("# FDTD.SetRectilinearGrid(x_lines, y_lines, z_lines)")

if __name__ == "__main__":
    main()
