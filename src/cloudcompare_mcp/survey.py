"""Native survey tools for caves and other enclosed 3D spaces (no CloudCompare needed).

- cut_ceiling: "scappellamento" — remove the vault/ceiling relative to the local floor,
  handling stacked galleries (multi-level caves) and sloping passages.
- extract_sections: vertical cross-sections along a line or perpendicular to a centerline.
- LAS helpers: read/write preserving every attribute, merge with a scan_id field,
  apply 4x4 rigid transforms.

All functions work on LAS/LAZ and keep the original georeferencing (scales/offsets).
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

# ── LAS helpers ──────────────────────────────────────────────────────────────


def is_las(path: str) -> bool:
    return Path(path).suffix.lower() in (".las", ".laz")


def read_las(path: str) -> Any:
    import laspy

    return laspy.read(path)


def write_las(las: Any, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    las.write(path)  # .laz is compressed automatically (lazrs backend)


def subset_las(las: Any, mask: Any) -> Any:
    import copy

    import laspy

    out = laspy.LasData(copy.deepcopy(las.header))  # own header: extra dims added later stay local
    out.points = las.points[mask].copy()
    return out


def add_float_field(las: Any, name: str, values: Any, description: str = "") -> None:
    import laspy

    if name not in las.point_format.dimension_names:
        las.add_extra_dim(laspy.ExtraBytesParams(name=name, type=np.float32, description=description[:31]))
    las[name] = np.asarray(values, dtype=np.float32)


def common_shift(paths: list[str]) -> tuple[float, float, float] | None:
    """CloudCompare global shift shared by all files (from the first LAS/LAZ header).

    CloudCompare stores coordinates as float32 after subtracting a shift. If two files
    get different shifts, multi-cloud operations (ICP, C2C, merge) mix frames. A single
    explicit shift avoids that. Rounded to 100 m so it is readable in logs.
    """
    import laspy

    for p in paths:
        if is_las(p) and Path(p).is_file():
            with laspy.open(p) as r:
                mins, maxs = r.header.mins, r.header.maxs
            cx, cy, cz = ((mins[i] + maxs[i]) / 2 for i in range(3))
            sx = -round(cx / 100.0) * 100.0 if abs(cx) > 1e4 else 0.0
            sy = -round(cy / 100.0) * 100.0 if abs(cy) > 1e4 else 0.0
            sz = -round(cz / 100.0) * 100.0 if abs(cz) > 1e4 else 0.0
            return (sx, sy, sz)
    return None


def apply_rigid_transform(las: Any, matrix: Any, shift: tuple[float, float, float] | None) -> None:
    """Apply a 4x4 transform computed by CloudCompare in shifted (local) coordinates."""
    m = np.asarray(matrix, dtype=np.float64)
    r, t = m[:3, :3], m[:3, 3]
    s = np.asarray(shift or (0.0, 0.0, 0.0), dtype=np.float64)
    xyz = np.column_stack([np.asarray(las.x), np.asarray(las.y), np.asarray(las.z)]) + s
    xyz = xyz @ r.T + t - s
    las.x, las.y, las.z = xyz[:, 0], xyz[:, 1], xyz[:, 2]


def read_matrix_file(path: str) -> Any:
    rows = []
    for line in Path(path).read_text().splitlines():
        vals = [float(v) for v in line.replace(",", " ").split()]
        if len(vals) == 4:
            rows.append(vals)
    if len(rows) < 4:
        raise ValueError(f"Could not read a 4x4 matrix from {path}")
    return np.array(rows[:4])


def merge_las_with_ids(paths: list[str], output: str, field: str = "scan_id") -> dict:
    """Merge LAS/LAZ files into one, keeping all standard attributes and adding scan_id.

    The point format / version of the first file is used; other files are converted
    (missing attributes filled with 0).
    """
    import laspy

    clouds = [laspy.read(p) for p in paths]
    ref = clouds[0]
    pf_id = ref.header.point_format.id
    header = laspy.LasHeader(point_format=pf_id, version=ref.header.version)
    header.scales = ref.header.scales
    all_xyz = np.vstack([np.column_stack([np.asarray(c.x), np.asarray(c.y), np.asarray(c.z)]) for c in clouds])
    header.offsets = np.floor(all_xyz.min(axis=0))
    header.add_extra_dim(laspy.ExtraBytesParams(name=field, type=np.uint16, description="source scan index"))
    try:
        crs = ref.header.parse_crs()
        if crs is not None:
            header.add_crs(crs)
    except Exception:  # noqa: BLE001
        pass

    out = laspy.LasData(header)
    n_total = len(all_xyz)
    out.points = laspy.ScaleAwarePointRecord.zeros(n_total, header=header)
    std_dims = [d for d in laspy.PointFormat(pf_id).dimension_names if d not in ("X", "Y", "Z")]
    start = 0
    counts = []
    for i, c in enumerate(clouds):
        if c.header.point_format.id != pf_id:
            c = laspy.convert(c, point_format_id=pf_id)
        n = len(c.points)
        sl = slice(start, start + n)
        for d in std_dims:
            if d in c.point_format.dimension_names:
                out[d][sl] = c[d]
        out[field][sl] = i
        counts.append(n)
        start += n
    out.x, out.y, out.z = all_xyz[:, 0], all_xyz[:, 1], all_xyz[:, 2]
    write_las(out, output)
    return {"points_per_scan": counts, "total_points": int(n_total)}


# ── Ceiling cut (scappellamento) ─────────────────────────────────────────────


def ceiling_cut_mask(
    x: Any, y: Any, z: Any,
    cell: float = 0.5,
    min_void: float = 0.8,
    h_min: float | None = None,
    h_max: float = 1.5,
    neighbourhood: int = 7,
) -> tuple[Any, Any]:
    """Return (keep_mask, height_above_floor) for a floor-relative ceiling cut.

    The plan is divided into square cells. In each cell the points are sorted by Z and
    split into vertical segments wherever there is an empty gap > min_void. Seen from
    inside a cave, a column crosses floor, air, ceiling, rock, floor of the level above,
    air, ceiling… so segments alternate floor / ceiling, starting with a floor. A segment
    that contains walls (floor and ceiling joined) counts as a floor segment and is cut
    at its base + h_max. If the lowest segment of a cell is much higher than the floor
    of the neighbouring cells (floor hidden by occlusion), it is treated as a ceiling.

    height_above_floor is NaN for points on ceiling segments.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    ix = np.floor((x - x.min()) / cell).astype(np.int64)
    iy = np.floor((y - y.min()) / cell).astype(np.int64)
    nx, ny = int(ix.max()) + 1, int(iy.max()) + 1
    cid = ix * ny + iy

    order = np.lexsort((z, cid))
    cs, zs = cid[order], z[order]
    new_cell = np.r_[True, cs[1:] != cs[:-1]]
    gap = np.r_[0.0, np.diff(zs)]
    seg_start = new_cell | ((gap > min_void) & ~new_cell)
    seg_id = np.cumsum(seg_start) - 1
    first_seg_of_cell = np.maximum.accumulate(np.where(new_cell, seg_id, 0))
    k = seg_id - first_seg_of_cell
    seg_min = zs[seg_start][seg_id]

    # lowest point of every occupied cell, and the 25th percentile of the lowest points
    # in the surrounding neighbourhood x neighbourhood cells (vectorised over occupied cells)
    occ = cs[new_cell]
    occ_low = zs[new_cell]
    grid = np.full((nx, ny), np.nan)
    ox, oy = occ // ny, occ % ny
    grid[ox, oy] = occ_low
    r = neighbourhood // 2
    stack = np.full(((2 * r + 1) ** 2, len(occ)), np.nan)
    j = 0
    for dx in range(-r, r + 1):
        for dy in range(-r, r + 1):
            qx, qy = ox + dx, oy + dy
            ok = (qx >= 0) & (qx < nx) & (qy >= 0) & (qy < ny)
            stack[j, ok] = grid[qx[ok], qy[ok]]
            j += 1
    with np.errstate(invalid="ignore"), __import__("warnings").catch_warnings():
        __import__("warnings").simplefilter("ignore", RuntimeWarning)
        nb_floor = np.nanpercentile(stack, 25, axis=0)
        ceiling_first_occ = ((occ_low - nb_floor) > min_void).astype(np.int64)
    ceiling_first = np.zeros(nx * ny, dtype=np.int64)
    ceiling_first[occ] = ceiling_first_occ
    is_floor = ((k + ceiling_first[cs]) % 2) == 0

    hab_sorted = zs - seg_min
    keep_sorted = is_floor & (hab_sorted <= h_max)
    if h_min is not None:
        keep_sorted &= hab_sorted >= h_min

    keep = np.empty_like(keep_sorted)
    keep[order] = keep_sorted
    hab = np.empty_like(hab_sorted)
    hab[order] = np.where(is_floor, hab_sorted, np.nan)
    return keep, hab


# ── Sections ─────────────────────────────────────────────────────────────────


def read_centerline(path: str) -> list[list[float]]:
    """Read a polyline from GeoJSON (LineString / MultiLineString / Feature[Collection]) or CSV x,y."""
    p = Path(path)
    if p.suffix.lower() in (".geojson", ".json"):
        data = json.loads(p.read_text())

        def _coords(obj: dict) -> list:
            t = obj.get("type")
            if t == "FeatureCollection":
                for f in obj["features"]:
                    c = _coords(f)
                    if c:
                        return c
                return []
            if t == "Feature":
                return _coords(obj["geometry"])
            if t == "LineString":
                return obj["coordinates"]
            if t == "MultiLineString":
                return [pt for part in obj["coordinates"] for pt in part]
            return []

        coords = _coords(data)
    else:
        coords = []
        with open(p, newline="") as f:
            for row in csv.reader(f):
                try:
                    coords.append([float(row[0]), float(row[1])])
                except (ValueError, IndexError):
                    continue
    coords = [[float(c[0]), float(c[1])] for c in coords]
    if len(coords) < 2:
        raise ValueError(f"No polyline with at least 2 vertices found in {path}")
    return coords


def stations_along(polyline: list[list[float]], spacing: float, start: float = 0.0) -> list[dict]:
    """Points every `spacing` m along a polyline, with the local direction (unit vector)."""
    pts = np.asarray(polyline, dtype=np.float64)
    seg = np.diff(pts, axis=0)
    seg_len = np.hypot(seg[:, 0], seg[:, 1])
    keep = seg_len > 0
    pts = np.vstack([pts[0], pts[1:][keep]])
    seg, seg_len = seg[keep], seg_len[keep]
    cum = np.r_[0.0, np.cumsum(seg_len)]
    total = float(cum[-1])
    out = []
    s = start
    while s <= total + 1e-9:
        i = min(int(np.searchsorted(cum, s, side="right") - 1), len(seg) - 1)
        t = (s - cum[i]) / seg_len[i]
        p = pts[i] + t * seg[i]
        # smooth direction: average of neighbouring segment directions near vertices
        d = seg[i] / seg_len[i]
        out.append({"chainage": round(s, 3), "x": float(p[0]), "y": float(p[1]), "dir": d})
        s += spacing
    return out


def section_masks(
    x: Any, y: Any, origin: Any, direction: Any, half_width: float, thickness: float,
    along_range: tuple[float, float] | None = None,
) -> tuple[Any, Any]:
    """Points within thickness/2 of the vertical plane through `origin` containing `direction`.

    Returns (mask, s) where s is the signed distance along `direction` from origin.
    """
    d = np.asarray(direction, dtype=np.float64)
    d = d / np.hypot(d[0], d[1])
    n = np.array([-d[1], d[0]])
    dx = x - origin[0]
    dy = y - origin[1]
    s = dx * d[0] + dy * d[1]
    off = dx * n[0] + dy * n[1]
    lo, hi = along_range if along_range else (-half_width, half_width)
    mask = (np.abs(off) <= thickness / 2) & (s >= lo) & (s <= hi)
    return mask, s


def plot_sections(sections: list[dict], title: str) -> str:
    """Grid of section profiles (s vs z, true scale) as base64 PNG."""
    import base64
    import io

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(sections)
    cols = min(4, n)
    rows = math.ceil(n / cols)
    size = (12, 4.5) if n == 1 else (4.2 * cols, 3.6 * rows)
    fig, axes = plt.subplots(rows, cols, figsize=size, squeeze=False, facecolor="white")
    zmin = min(float(np.min(sec["z"])) for sec in sections if len(sec["z"])) if any(len(s["z"]) for s in sections) else 0
    zmax = max(float(np.max(sec["z"])) for sec in sections if len(sec["z"])) if any(len(s["z"]) for s in sections) else 1
    for ax, sec in zip(axes.ravel(), sections):
        if len(sec["z"]):
            ax.scatter(sec["s"], sec["z"], s=0.8, c="#1f3b5c", linewidths=0, rasterized=True)
        lo, hi = sec.get("xlim") or ((float(np.min(sec["s"])) - 0.5, float(np.max(sec["s"])) + 0.5) if len(sec["s"]) else (-1, 1))
        ax.set_xlim(lo, hi)
        ax.set_ylim(zmin - 0.3, zmax + 0.3)
        ax.set_aspect("equal", adjustable="box")
        label = sec["name"] + (f" @ {sec['chainage']:g} m" if sec.get("chainage") is not None else "")
        ax.set_title(f"{label}  ({sec['n']:,} pts)", fontsize=9)
        ax.tick_params(labelsize=7)
        ax.grid(True, lw=0.3, alpha=0.5)
        if sec.get("chainage") is not None:
            ax.axvline(0, color="#c0392b", lw=0.6, alpha=0.6)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def plot_cut(x: Any, y: Any, z: Any, keep: Any, title: str, check_sections: list[dict]) -> str:
    """Plan of kept points + check sections (kept green / removed red) as base64 PNG."""
    import base64
    import io

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(0)
    n = len(x)
    idx = rng.choice(n, min(n, 400_000), replace=False)
    fig = plt.figure(figsize=(15, 10), facecolor="white")
    gs = fig.add_gridspec(2, max(1, len(check_sections)), height_ratios=[2.2, 1])
    ax = fig.add_subplot(gs[0, :])
    k = keep[idx]
    o = np.argsort(z[idx][k])
    sc = ax.scatter(x[idx][k][o], y[idx][k][o], c=z[idx][k][o], s=0.3, cmap="viridis", linewidths=0, rasterized=True)
    for i, cs in enumerate(check_sections):
        ax.plot(cs["trace_x"], cs["trace_y"], color="#c0392b", lw=1)
        ax.annotate(cs["name"], (cs["trace_x"][1], cs["trace_y"][1]), color="#c0392b", fontsize=8)
    ax.set_aspect("equal", adjustable="datalim")
    ax.ticklabel_format(useOffset=False, style="plain")
    ax.tick_params(labelsize=7)
    ax.set_title(title, fontsize=11)
    fig.colorbar(sc, ax=ax, fraction=0.025, pad=0.01).set_label("Z (m)", fontsize=8)
    for i, cs in enumerate(check_sections):
        a = fig.add_subplot(gs[1, i])
        m = cs["mask"]
        a.scatter(cs["s"][m & keep], z[m & keep], s=0.8, c="#27ae60", linewidths=0, label="kept")
        a.scatter(cs["s"][m & ~keep], z[m & ~keep], s=0.8, c="#e74c3c", linewidths=0, label="removed")
        a.set_aspect("equal", adjustable="datalim")
        a.set_title(f"check {cs['name']}", fontsize=9)
        a.tick_params(labelsize=7)
        if i == 0:
            a.legend(fontsize=7, markerscale=6, loc="upper right")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=90)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()
