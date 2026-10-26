"""CloudCompare MCP Server — wraps CloudCompare CLI for AI-assisted point cloud processing.

Fork with fixes for underground / 3D surveys (caves, tunnels, hypogea):
- CloudCompare CLI calls follow the documented argument order and semantics
  (C2C compared/reference order, ICP data/model order, -OCTREE_NORMALS,
  -AUTO_SAVE OFF so no stray files are written next to the inputs,
  quoted -SAVE_CLOUDS FILE names so paths with spaces work, global shift).
- Density is reported as true 3D point spacing (nearest-neighbour distance),
  not points per XY bounding-box area, which is meaningless for enclosed spaces.
- LAS/LAZ files are read in chunks with random sampling, so very large scans
  do not have to fit in memory for metadata / visualization.
- "intensity" colouring uses the real intensity values.
"""

import base64
import io
import json
import os
import platform
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import ImageContent, TextContent, Tool

# ── CloudCompare binary discovery ────────────────────────────────────────────

_CC_CANDIDATES: dict[str, list[str]] = {
    "Windows": [
        r"C:\Program Files\CloudCompare\CloudCompare.exe",
        r"C:\Program Files (x86)\CloudCompare\CloudCompare.exe",
    ],
    "Darwin": [
        "/Applications/CloudCompare.app/Contents/MacOS/CloudCompare",
        "/Applications/cloudcompare.app/Contents/MacOS/CloudCompare",
        "/usr/local/bin/cloudcompare",
        "/opt/homebrew/bin/cloudcompare",
    ],
    "Linux": [
        "/usr/bin/cloudcompare",
        "/usr/bin/CloudCompare",
        "/usr/local/bin/cloudcompare",
        "/snap/bin/cloudcompare",
        "/opt/cloudcompare/bin/cloudcompare",
    ],
}

# Generous default: TLS cave scans can take a long time to process.
_CC_TIMEOUT_S = int(os.environ.get("CLOUDCOMPARE_TIMEOUT", "7200"))


def find_cloudcompare() -> str | None:
    if env := os.environ.get("CLOUDCOMPARE_PATH"):
        return env if Path(env).is_file() else None
    if found := shutil.which("cloudcompare") or shutil.which("CloudCompare"):
        return found
    for candidate in _CC_CANDIDATES.get(platform.system(), []):
        if Path(candidate).is_file():
            return candidate
    return None


def cc_run(
    args: list[str],
    cwd: str | None = None,
    managed: bool = True,
    timeout: int | None = None,
) -> tuple[int, str, str]:
    """Run CloudCompare in command-line mode.

    -SILENT must be the first argument. A log file is always requested because
    in silent mode (notably on Windows) CloudCompare writes nothing to stdout.
    With managed=True, automatic saving after every step is disabled, so only
    the explicit -SAVE_CLOUDS / -SAVE_MESHES at the end write files.
    """
    binary = find_cloudcompare()
    if not binary:
        raise FileNotFoundError(
            "CloudCompare executable not found. Install CloudCompare (>= 2.12) and set "
            "the CLOUDCOMPARE_PATH environment variable if needed."
        )
    with tempfile.TemporaryDirectory(prefix="ccmcp_") as tmp:
        log_path = str(Path(tmp) / "cc_log.txt")
        prefix = ["-SILENT", "-LOG_FILE", log_path]
        if managed:
            prefix += ["-AUTO_SAVE", "OFF"]
        try:
            result = subprocess.run(
                [binary, *prefix, *args],
                capture_output=True,
                text=True,
                cwd=cwd,
                timeout=timeout or _CC_TIMEOUT_S,
            )
            rc, stdout, stderr = result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            rc, stdout, stderr = -1, "", f"CloudCompare timed out after {timeout or _CC_TIMEOUT_S} s"
        log = ""
        if Path(log_path).is_file():
            log = Path(log_path).read_text(encoding="utf-8", errors="replace")
    # On some platforms CloudCompare writes the same messages to stdout and to
    # the log file; prefer the log file and fall back to stdout.
    if log.strip():
        stdout = log
    return rc, stdout, stderr


def _open(path: str) -> list[str]:
    """-O with automatic global shift (avoids precision loss with UTM / national grid coords)."""
    return ["-O", "-GLOBAL_SHIFT", "AUTO", path]


def _quote(path: str) -> str:
    """Quote a file name for -SAVE_CLOUDS FILE (CloudCompare splits that list on spaces)."""
    if '"' in path:
        raise ValueError(f"File names containing double quotes are not supported: {path}")
    return f'"{path}"'


def _save_clouds(*paths: str) -> list[str]:
    """One file name per cloud currently loaded, in loading order."""
    return ["-SAVE_CLOUDS", "FILE", " ".join(_quote(p) for p in paths)]


def _export_fmt(path: str) -> list[str]:
    return ["-C_EXPORT_FMT", _ext_flag(path)]


# ── Native point cloud readers (no CloudCompare needed) ──────────────────────
#
# Every loader returns (xyz Nx3 float64, colors Nx3 float [0,1] or None,
# intensity N float or None, meta). When max_points is given and the file is
# larger, a uniform random sample is returned and meta["point_count"] still
# reports the full count.

_DEFAULT_SAMPLE = 2_000_000


def _random_subset(n: int, k: int, seed: int = 42) -> Any:
    import numpy as np

    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, k, replace=False))


def _las_colors(points: Any) -> Any:
    import numpy as np

    r = np.asarray(points.red, dtype=np.float32)
    g = np.asarray(points.green, dtype=np.float32)
    b = np.asarray(points.blue, dtype=np.float32)
    m = max(float(r.max(initial=0)), float(g.max(initial=0)), float(b.max(initial=0)))
    scale = 65535.0 if m > 255 else 255.0 if m > 1 else 1.0
    return np.column_stack([r, g, b]) / scale


def _load_las(filepath: str, max_points: int | None = None) -> tuple[Any, Any, Any, dict]:
    import laspy
    import numpy as np

    with laspy.open(filepath) as reader:
        header = reader.header
        n_total = int(header.point_count)
        dims = list(header.point_format.dimension_names)
        has_rgb = all(d in dims for d in ("red", "green", "blue"))
        has_intensity = "intensity" in dims

        keep_prob = 1.0 if not max_points or n_total <= max_points else max_points / n_total
        rng = np.random.default_rng(42)
        xyz_parts, rgb_parts, int_parts = [], [], []
        for chunk in reader.chunk_iterator(1_000_000):
            if keep_prob < 1.0:
                mask = rng.random(len(chunk)) < keep_prob
                chunk = chunk[mask]
                if len(chunk) == 0:
                    continue
            xyz_parts.append(np.column_stack([
                np.asarray(chunk.x, dtype=np.float64),
                np.asarray(chunk.y, dtype=np.float64),
                np.asarray(chunk.z, dtype=np.float64),
            ]))
            if has_rgb:
                rgb_parts.append(_las_colors(chunk))
            if has_intensity:
                int_parts.append(np.asarray(chunk.intensity, dtype=np.float32))

    xyz = np.vstack(xyz_parts) if xyz_parts else np.empty((0, 3))
    colors = np.vstack(rgb_parts) if rgb_parts else None
    intensity = np.concatenate(int_parts) if int_parts else None

    mins, maxs = header.mins, header.maxs
    meta: dict = {
        "format": "LAZ" if filepath.lower().endswith(".laz") else "LAS",
        "las_version": f"{header.version.major}.{header.version.minor}",
        "point_format": int(header.point_format.id),
        "point_count": n_total,
        "has_rgb": has_rgb,
        "has_intensity": has_intensity,
        "has_normals": "normal_x" in dims or "NormalX" in dims,
        "scalar_fields": [d for d in dims if d not in ("X", "Y", "Z", "red", "green", "blue")],
        "file_size_mb": round(Path(filepath).stat().st_size / 1_048_576, 2),
        "bbox": {
            "x": [round(float(mins[0]), 3), round(float(maxs[0]), 3)],
            "y": [round(float(mins[1]), 3), round(float(maxs[1]), 3)],
            "z": [round(float(mins[2]), 3), round(float(maxs[2]), 3)],
        },
        "_bbox_from_header": True,
    }
    try:
        crs = header.parse_crs()
        if crs is not None:
            meta["crs"] = crs.to_string()
    except Exception:  # noqa: BLE001 — CRS parsing is optional (needs pyproj)
        pass
    return xyz, colors, intensity, meta


def _subsample_loaded(xyz: Any, colors: Any, intensity: Any, max_points: int | None) -> tuple[Any, Any, Any]:
    if not max_points or len(xyz) <= max_points:
        return xyz, colors, intensity
    idx = _random_subset(len(xyz), max_points)
    return (
        xyz[idx],
        colors[idx] if colors is not None else None,
        intensity[idx] if intensity is not None else None,
    )


def _load_ply(filepath: str, max_points: int | None = None) -> tuple[Any, Any, Any, dict]:
    import numpy as np
    from plyfile import PlyData

    ply = PlyData.read(filepath)
    vertex = ply["vertex"]
    props = {p.name for p in vertex.properties}
    xyz = np.column_stack([
        np.asarray(vertex["x"], dtype=np.float64),
        np.asarray(vertex["y"], dtype=np.float64),
        np.asarray(vertex["z"], dtype=np.float64),
    ])

    has_rgb = {"red", "green", "blue"}.issubset(props)
    colors = None
    if has_rgb:
        rgb = np.column_stack([
            np.asarray(vertex["red"], dtype=np.float32),
            np.asarray(vertex["green"], dtype=np.float32),
            np.asarray(vertex["blue"], dtype=np.float32),
        ])
        colors = rgb / (255.0 if rgb.max(initial=0) > 1 else 1.0)

    int_name = next((n for n in ("intensity", "scalar_intensity", "scalar_Intensity") if n in props), None)
    intensity = np.asarray(vertex[int_name], dtype=np.float32) if int_name else None

    meta: dict = {
        "format": "PLY",
        "point_count": len(xyz),
        "has_rgb": has_rgb,
        "has_intensity": intensity is not None,
        "has_normals": "nx" in props or "normal_x" in props,
        "scalar_fields": sorted(props - {"x", "y", "z", "red", "green", "blue", "nx", "ny", "nz"}),
        "file_size_mb": round(Path(filepath).stat().st_size / 1_048_576, 2),
    }
    xyz, colors, intensity = _subsample_loaded(xyz, colors, intensity, max_points)
    return xyz, colors, intensity, meta


def _load_ascii(filepath: str, max_points: int | None = None) -> tuple[Any, Any, Any, dict]:
    """Whitespace / comma delimited XYZ (first 3 columns = x,y,z; 4 cols = +intensity; 6+ = +RGB)."""
    import numpy as np

    with open(filepath, encoding="utf-8", errors="ignore") as f:
        sample = "".join(f.readline() for _ in range(50))
    delimiter = "," if sample.count(",") > sample.count(" ") else None
    data = np.loadtxt(filepath, comments=["#", "//", "!"], delimiter=delimiter, ndmin=2)
    xyz = data[:, :3].astype(np.float64)
    colors = None
    intensity = None
    if data.shape[1] >= 6:
        rgb = data[:, 3:6].astype(np.float32)
        if rgb.max(initial=0) > 1:
            rgb /= 255.0
        colors = np.clip(rgb, 0, 1)
    if data.shape[1] == 4 or data.shape[1] >= 7:
        intensity = data[:, 3 if data.shape[1] == 4 else 6].astype(np.float32)

    meta: dict = {
        "format": "ASCII",
        "point_count": len(xyz),
        "has_rgb": colors is not None,
        "has_intensity": intensity is not None,
        "has_normals": False,
        "scalar_fields": [],
        "file_size_mb": round(Path(filepath).stat().st_size / 1_048_576, 2),
    }
    xyz, colors, intensity = _subsample_loaded(xyz, colors, intensity, max_points)
    return xyz, colors, intensity, meta


def _load_pcd(filepath: str, max_points: int | None = None) -> tuple[Any, Any, Any, dict]:
    """ASCII PCD (XYZ, XYZRGB, XYZI)."""
    import numpy as np

    with open(filepath, encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    fields: list[str] = []
    data_start = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("FIELDS"):
            fields = stripped.split()[1:]
        elif stripped.startswith("DATA"):
            if "ascii" not in stripped:
                raise ValueError(
                    "Binary PCD is not supported for native reading. "
                    "Use convert_format (CloudCompare) to convert it to PLY or LAS first."
                )
            data_start = i + 1
            break

    data = np.loadtxt(lines[data_start:], ndmin=2)
    fi = {name: idx for idx, name in enumerate(fields)}
    xyz = np.column_stack([data[:, fi["x"]], data[:, fi["y"]], data[:, fi["z"]]]).astype(np.float64)

    colors = None
    if "rgb" in fi or "rgba" in fi:
        col = data[:, fi.get("rgb", fi.get("rgba"))]
        packed = col.astype(np.float32).view(np.uint32)
        colors = np.column_stack([
            ((packed >> 16) & 0xFF).astype(np.float32),
            ((packed >> 8) & 0xFF).astype(np.float32),
            (packed & 0xFF).astype(np.float32),
        ]) / 255.0
    intensity = data[:, fi["intensity"]].astype(np.float32) if "intensity" in fi else None

    meta: dict = {
        "format": "PCD",
        "point_count": len(xyz),
        "has_rgb": colors is not None,
        "has_intensity": intensity is not None,
        "has_normals": "normal_x" in fi,
        "scalar_fields": [f for f in fields if f not in ("x", "y", "z", "rgb", "rgba")],
        "file_size_mb": round(Path(filepath).stat().st_size / 1_048_576, 2),
    }
    xyz, colors, intensity = _subsample_loaded(xyz, colors, intensity, max_points)
    return xyz, colors, intensity, meta


def _load_cloud(filepath: str, max_points: int | None = _DEFAULT_SAMPLE) -> tuple[Any, Any, Any, dict]:
    ext = Path(filepath).suffix.lower()
    loaders = {
        ".las": _load_las, ".laz": _load_las,
        ".ply": _load_ply,
        ".pcd": _load_pcd,
        ".xyz": _load_ascii, ".asc": _load_ascii,
        ".txt": _load_ascii, ".csv": _load_ascii, ".pts": _load_ascii,
    }
    loader = loaders.get(ext)
    if loader is None:
        raise ValueError(
            f"Unsupported format '{ext}' for native reading. "
            "Supported: LAS, LAZ, PLY, PCD (ASCII), XYZ, ASC, TXT, CSV. "
            "For other formats (E57, BIN…), use convert_format (CloudCompare) first."
        )
    xyz, colors, intensity, meta = loader(filepath, max_points)
    if len(xyz) == 0:
        raise ValueError("The file contains no points.")
    meta["sampled_points"] = len(xyz)
    return xyz, colors, intensity, meta


# ── Geometry statistics (3D-aware) ───────────────────────────────────────────

def _spacing_stats(xyz: Any, sampling_fraction: float, n_query: int = 50_000) -> dict:
    """Nearest-neighbour spacing, the meaningful density measure for 3D surveys.

    If the cloud was randomly subsampled with fraction f before this step, the
    measured spacing is inflated by ~1/sqrt(f) for points lying on surfaces
    (walls, floor, ceiling); we correct for that and flag the value as estimated.
    """
    import numpy as np
    from scipy.spatial import cKDTree

    n = len(xyz)
    if n < 2:
        return {}
    centered = xyz - xyz.mean(axis=0)  # better float precision with large coordinates
    tree = cKDTree(centered)
    q_idx = _random_subset(n, min(n_query, n), seed=7)
    dists, _ = tree.query(centered[q_idx], k=2)
    nn = dists[:, 1]
    nn = nn[nn > 0]  # ignore exact duplicates
    if nn.size == 0:
        return {"duplicate_points_only": True}

    f = sampling_fraction if 0 < sampling_fraction < 1 else 1.0
    corr = float(np.sqrt(f))
    median = float(np.median(nn)) * corr
    mean = float(np.mean(nn)) * corr
    p95 = float(np.percentile(nn, 95)) * corr

    # Surface density (same definition as CloudCompare's "surface density"):
    # neighbours inside a small sphere / area of the disc it cuts on the wall.
    r = 5.0 * float(np.median(nn))  # in sample units: ~25 neighbours on a surface
    counts = tree.query_ball_point(centered[q_idx], r, return_length=True)
    surf_density = float(np.mean(counts - 1)) / (np.pi * r * r) / f
    out = {
        "median_point_spacing_m": round(median, 4),
        "mean_point_spacing_m": round(mean, 4),
        "p95_point_spacing_m": round(p95, 4),
        "surface_density_pts_per_m2": round(surf_density, 1),
        "surface_density_radius_m": round(r * corr, 4),
    }
    if corr != 1.0:
        out["spacing_estimated_from_subsample"] = True
    return out


def _cloud_meta_stats(xyz: Any, meta: dict) -> dict:
    """Add bounding box, extent and 3D spacing statistics to meta."""
    import numpy as np

    if "bbox" not in meta:
        lo, hi = xyz.min(axis=0), xyz.max(axis=0)
        meta["bbox"] = {k: [round(float(lo[i]), 3), round(float(hi[i]), 3)] for i, k in enumerate("xyz")}
    bb = meta["bbox"]
    ext = {k: round(bb[k][1] - bb[k][0], 3) for k in "xyz"}
    meta["extent_m"] = ext
    meta["bbox_diagonal_m"] = round(float(np.sqrt(sum(v * v for v in ext.values()))), 3)
    frac = meta["sampled_points"] / meta["point_count"] if meta["point_count"] else 1.0
    meta["spacing"] = _spacing_stats(xyz, frac)
    meta.pop("_bbox_from_header", None)
    return meta


# ── Visualization ─────────────────────────────────────────────────────────────

def _fig_to_base64(fig: Any) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight", facecolor=fig.get_facecolor())
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def _pick_colors(xyz: Any, colors: Any, intensity: Any, color_by: str) -> tuple[Any, str | None, str]:
    """Return (c_array, cmap_name_or_None, colorbar_label)."""
    import numpy as np

    if color_by == "rgb" and colors is not None:
        return colors, None, ""
    if color_by == "intensity" and intensity is not None:
        lo, hi = np.percentile(intensity, [2, 98])
        if hi <= lo:
            hi = lo + 1
        return np.clip(intensity, lo, hi), "gray", "Intensity (2–98 %)"
    return xyz[:, 2], "viridis", "Z (m)"


def _scatter_kwargs(n: int) -> dict:
    s = max(0.05, min(3.0, 80_000 / n))
    alpha = max(0.08, min(1.0, 60_000 / n))
    return {"s": s, "alpha": alpha, "linewidths": 0, "rasterized": True}


_BG = "#0d1117"
_FG = "#e6edf3"
_GRAY = "#8b949e"


def _ax_style(ax: Any, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_facecolor(_BG)
    ax.set_title(title, color=_FG, fontsize=10, pad=4)
    ax.set_xlabel(xlabel, color=_GRAY, fontsize=8)
    ax.set_ylabel(ylabel, color=_GRAY, fontsize=8)
    ax.tick_params(colors=_GRAY, labelsize=7)
    ax.ticklabel_format(useOffset=False, style="plain")
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")


def _fmt(v: Any, spec: str = ".3f") -> str:
    return format(v, spec) if isinstance(v, (int, float)) else str(v)


def _build_figure(xyz: Any, colors: Any, intensity: Any, meta: dict, color_by: str, filename: str) -> Any:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.gridspec as gridspec
    import matplotlib.pyplot as plt

    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    c, cmap, cbar_label = _pick_colors(xyz, colors, intensity, color_by)
    kw = _scatter_kwargs(len(xyz))

    fig = plt.figure(figsize=(16, 11), facecolor=_BG)
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.25)

    def _add_cbar(mappable: Any, ax: Any) -> None:
        if cmap and cbar_label:
            cb = fig.colorbar(mappable, ax=ax, pad=0.02, fraction=0.04)
            cb.ax.yaxis.set_tick_params(color=_GRAY, labelcolor=_GRAY, labelsize=7)
            cb.set_label(cbar_label, color=_GRAY, fontsize=7)
            cb.outline.set_edgecolor("#30363d")

    # Top view: draw high points last so the plan reads like a ceiling-down view.
    order = z.argsort()
    ax0 = fig.add_subplot(gs[0, 0])
    sc0 = ax0.scatter(x[order], y[order], c=c[order] if cmap else c[order], cmap=cmap, **kw)
    _ax_style(ax0, "Plan view (XY)", "X (m)", "Y (m)")
    ax0.set_aspect("equal", adjustable="datalim")
    _add_cbar(sc0, ax0)

    ax1 = fig.add_subplot(gs[0, 1])
    sc1 = ax1.scatter(x, z, c=c, cmap=cmap, **kw)
    _ax_style(ax1, "Elevation (XZ)", "X (m)", "Z (m)")
    ax1.set_aspect("equal", adjustable="datalim")
    _add_cbar(sc1, ax1)

    ax2 = fig.add_subplot(gs[1, 0])
    sc2 = ax2.scatter(y, z, c=c, cmap=cmap, **kw)
    _ax_style(ax2, "Elevation (YZ)", "Y (m)", "Z (m)")
    ax2.set_aspect("equal", adjustable="datalim")
    _add_cbar(sc2, ax2)

    ax3 = fig.add_subplot(gs[1, 1])
    ax3.set_facecolor(_BG)
    ax3.axis("off")
    bb, ex, sp = meta.get("bbox", {}), meta.get("extent_m", {}), meta.get("spacing", {})
    sf = meta.get("scalar_fields", [])
    lines = [
        f"  Points      {meta['point_count']:>14,}",
        f"  File size   {meta.get('file_size_mb', '?'):>13} MB",
        "",
        "  Bounding box (m)",
        *[
            f"    {k.upper()}  [{_fmt(bb.get(k, ['?', '?'])[0], '.2f')}, {_fmt(bb.get(k, ['?', '?'])[1], '.2f')}]"
            f"  Δ {_fmt(ex.get(k, '?'), '.2f')}"
            for k in "xyz"
        ],
        "",
        "  Point spacing (3D nearest neighbour)",
        f"    median    {_fmt(sp.get('median_point_spacing_m', 'N/A'), '.4f')} m",
        f"    p95       {_fmt(sp.get('p95_point_spacing_m', 'N/A'), '.4f')} m",
        f"    density   {_fmt(sp.get('surface_density_pts_per_m2', 'N/A'), ',.0f')} pts/m² of surface",
        "    (estimated from subsample)" if sp.get("spacing_estimated_from_subsample") else "",
        "",
        "  Attributes",
        f"    RGB         {'✓' if meta.get('has_rgb') else '✗'}",
        f"    Intensity   {'✓' if meta.get('has_intensity') else '✗'}",
        f"    Normals     {'✓' if meta.get('has_normals') else '✗'}",
        f"    Fields      {len(sf):>3}" + (f"  ({', '.join(sf[:3])}{'…' if len(sf) > 3 else ''})" if sf else ""),
    ]
    ax3.text(0.02, 0.98, "\n".join(lines), transform=ax3.transAxes, color=_FG, fontsize=9,
             va="top", fontfamily="monospace", linespacing=1.5)
    ax3.set_title("Metadata", color=_FG, fontsize=10, pad=4)

    n_shown, n_total = len(xyz), meta["point_count"]
    subtitle = f"{n_shown:,} pts shown" + (f" (of {n_total:,})" if n_shown < n_total else "")
    fig.suptitle(f"{filename}   ·   {subtitle}", color=_FG, fontsize=12, y=0.995)
    return fig


# ── MCP Server setup ──────────────────────────────────────────────────────────

server = Server("cloudcompare-mcp")

_IO_PROPS = {
    "input_path": {"type": "string", "description": "Absolute path to the input cloud."},
    "output_path": {
        "type": "string",
        "description": "Absolute path of the file to write. Format from extension (.las .laz .ply .pcd .xyz .asc .txt .e57 .bin). Overwritten if it exists.",
    },
}

TOOLS: list[Tool] = [
    Tool(
        name="get_cloudcompare_info",
        description="Check whether CloudCompare is installed and return its path. Call this first.",
        inputSchema={"type": "object", "properties": {}},
    ),
    Tool(
        name="read_cloud_metadata",
        description=(
            "Read a point cloud natively (no CloudCompare needed) and return point count, "
            "bounding box, extent, 3D point spacing (median / p95 nearest-neighbour distance — "
            "the meaningful density measure for caves and other enclosed 3D spaces), "
            "RGB / intensity / normals presence and field names. LAS/LAZ are read in chunks, "
            "so huge files are fine. Supports LAS, LAZ, PLY, PCD (ASCII), XYZ, ASC, TXT, CSV."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "Absolute path to the point cloud file."},
                "max_points": {
                    "type": "integer",
                    "description": "Points sampled for statistics (default 2,000,000).",
                    "default": _DEFAULT_SAMPLE,
                },
            },
            "required": ["file_path"],
        },
    ),
    Tool(
        name="visualize_cloud",
        description=(
            "Render a point cloud as plan + two elevations + metadata panel and return it as a PNG "
            "the model can see. No CloudCompare needed. Supports LAS, LAZ, PLY, PCD, XYZ, ASC, TXT, CSV."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "Absolute path to the point cloud file."},
                "color_by": {
                    "type": "string",
                    "enum": ["height", "rgb", "intensity"],
                    "description": "'height' = Z gradient (default); 'rgb' = stored colours; 'intensity' = real intensity values. Falls back to height if the attribute is missing.",
                    "default": "height",
                },
                "max_points": {
                    "type": "integer",
                    "description": "Maximum points rendered (random sample). Default 400000.",
                    "default": 400000,
                },
            },
            "required": ["file_path"],
        },
    ),
    Tool(
        name="load_cloud_info",
        description=(
            "Open a cloud or mesh with CloudCompare and return its log (point count, global shift, "
            "scalar fields…). Works for any format CloudCompare reads (E57, BIN, OBJ, PTX…). Writes nothing."
        ),
        inputSchema={
            "type": "object",
            "properties": {"file_path": {"type": "string", "description": "Absolute path."}},
            "required": ["file_path"],
        },
    ),
    Tool(
        name="subsample",
        description="Reduce density with random, spatial (minimum spacing) or octree subsampling. Requires CloudCompare.",
        inputSchema={
            "type": "object",
            "properties": {
                **_IO_PROPS,
                "method": {"type": "string", "enum": ["RANDOM", "SPATIAL", "OCTREE"], "default": "SPATIAL"},
                "parameter": {
                    "type": "number",
                    "description": "RANDOM: points to keep. SPATIAL: minimum spacing in metres. OCTREE: level 1–21.",
                },
            },
            "required": ["input_path", "output_path", "parameter"],
        },
    ),
    Tool(
        name="compute_cloud_to_cloud_distances",
        description=(
            "C2C nearest-neighbour distances from the compared cloud to the reference cloud "
            "(e.g. new survey vs old survey). Saves the compared cloud with a distance scalar field. "
            "Requires CloudCompare."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "reference_path": {"type": "string", "description": "Reference cloud (distances are measured TO this)."},
                "compared_path": {"type": "string", "description": "Compared cloud (receives the distance field)."},
                "output_path": _IO_PROPS["output_path"],
                "max_distance": {"type": "number", "description": "Optional max search distance (m), speeds up computation."},
                "split_xyz": {"type": "boolean", "description": "Also store X/Y/Z components of the distance.", "default": False},
            },
            "required": ["reference_path", "compared_path", "output_path"],
        },
    ),
    Tool(
        name="compute_cloud_to_mesh_distances",
        description="Signed C2M distances from a cloud to a reference mesh. Saves the cloud with a distance field. Requires CloudCompare.",
        inputSchema={
            "type": "object",
            "properties": {
                "mesh_path": {"type": "string"},
                "cloud_path": {"type": "string"},
                "output_path": _IO_PROPS["output_path"],
            },
            "required": ["mesh_path", "cloud_path", "output_path"],
        },
    ),
    Tool(
        name="icp_registration",
        description=(
            "Align the data cloud onto the fixed model (reference) cloud with ICP. Saves the moved data "
            "cloud; the transformation matrix and final RMS are in the returned log. Requires CloudCompare."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "model_path": {"type": "string", "description": "Reference cloud — does not move."},
                "data_path": {"type": "string", "description": "Cloud to be moved onto the model."},
                "output_path": _IO_PROPS["output_path"],
                "overlap": {"type": "number", "default": 100, "description": "Expected overlap % (10–100)."},
                "iterations": {"type": "integer", "description": "Max iterations (default: until convergence)."},
                "min_error_diff": {"type": "number", "description": "Convergence threshold on RMS change (default 1e-6)."},
                "random_sampling_limit": {"type": "integer", "default": 50000},
                "adjust_scale": {"type": "boolean", "default": False},
            },
            "required": ["model_path", "data_path", "output_path"],
        },
    ),
    Tool(
        name="compute_normals",
        description=(
            "Estimate point normals with CloudCompare's octree method. For caves, orient them with "
            "'PLUS_BARYCENTER'/'MINUS_BARYCENTER' only for simple chambers; for galleries leave "
            "orientation empty or use 'mst_neighbors'. Requires CloudCompare."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                **_IO_PROPS,
                "radius": {
                    "description": "Neighbourhood radius in metres, or 'AUTO' (default).",
                    "type": ["number", "string"],
                    "default": "AUTO",
                },
                "model": {"type": "string", "enum": ["LS", "QUADRIC", "TRI"], "default": "QUADRIC"},
                "orientation": {
                    "type": "string",
                    "description": "Optional: PLUS_X, MINUS_X, PLUS_Y, MINUS_Y, PLUS_Z, MINUS_Z, PLUS_ORIGIN, MINUS_ORIGIN, PLUS_BARYCENTER, MINUS_BARYCENTER, PLUS_SENSOR_ORIGIN, MINUS_SENSOR_ORIGIN.",
                },
                "mst_neighbors": {"type": "integer", "description": "Optional: re-orient with a Minimum Spanning Tree using this many neighbours (e.g. 6–12)."},
            },
            "required": ["input_path", "output_path"],
        },
    ),
    Tool(
        name="filter_by_scalar_field",
        description="Keep only points whose scalar field value is within [min_val, max_val]. Requires CloudCompare.",
        inputSchema={
            "type": "object",
            "properties": {
                **_IO_PROPS,
                "min_val": {"type": "number"},
                "max_val": {"type": "number"},
                "scalar_field": {"type": "string", "description": "Field name or index to filter on (e.g. 'Intensity', 'C2C absolute distances'). Default: active field."},
            },
            "required": ["input_path", "output_path", "min_val", "max_val"],
        },
    ),
    Tool(
        name="statistical_outlier_removal",
        description="Remove isolated noise points (SOR filter). Requires CloudCompare.",
        inputSchema={
            "type": "object",
            "properties": {
                **_IO_PROPS,
                "knn": {"type": "integer", "default": 6},
                "n_sigma": {"type": "number", "default": 1.0},
            },
            "required": ["input_path", "output_path"],
        },
    ),
    Tool(
        name="merge_clouds",
        description="Merge two or more point clouds into one file. Requires CloudCompare.",
        inputSchema={
            "type": "object",
            "properties": {
                "input_paths": {"type": "array", "items": {"type": "string"}, "minItems": 2},
                "output_path": _IO_PROPS["output_path"],
            },
            "required": ["input_paths", "output_path"],
        },
    ),
    Tool(
        name="convert_format",
        description="Convert a point cloud to another format (from the output extension: .las .laz .ply .pcd .xyz .asc .txt .e57 .bin). Requires CloudCompare.",
        inputSchema={
            "type": "object",
            "properties": {**_IO_PROPS},
            "required": ["input_path", "output_path"],
        },
    ),
    Tool(
        name="run_cloudcompare_command",
        description=(
            "Run an arbitrary CloudCompare command line (advanced). Do NOT include the binary path, "
            "-SILENT or -LOG_FILE — they are added automatically. Note: CloudCompare saves after "
            "every step by default here; add [\"-AUTO_SAVE\", \"OFF\"] and an explicit -SAVE_CLOUDS to control output. "
            "Example: [\"-O\", \"/path/cloud.las\", \"-SOR\", \"8\", \"2\"]"
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "args": {"type": "array", "items": {"type": "string"}},
                "working_directory": {"type": "string"},
            },
            "required": ["args"],
        },
    ),
]


# ── Handler helpers ───────────────────────────────────────────────────────────

def _ok(data: dict | str) -> list[TextContent]:
    body = data if isinstance(data, str) else json.dumps(data, indent=2, ensure_ascii=False)
    return [TextContent(type="text", text=body)]


def _err(msg: str) -> list[TextContent]:
    return [TextContent(type="text", text=json.dumps({"error": msg}, indent=2, ensure_ascii=False))]


_LOG_NOISE = ("[Plugin]", "Plugin found:", "New file extensions registered", "[Global Shift] Max abs.")


def _tail(text: str, n: int = 60) -> str:
    lines = [ln for ln in text.strip().splitlines() if ln.strip() and not any(k in ln for k in _LOG_NOISE)]
    if len(lines) > n:
        lines = [f"… ({len(lines) - n} earlier lines omitted)"] + lines[-n:]
    return "\n".join(lines) or "(none)"


def _run_result(rc: int, stdout: str, stderr: str, output: str | None = None, extra: dict | None = None) -> list[TextContent]:
    success = rc == 0
    out: dict = {}
    if output is not None:
        exists = Path(output).is_file()
        # CloudCompare can exit 0 after logging an error; trust the file on disk.
        success = success and exists
        out["output"] = output
        out["output_written"] = exists
        if exists:
            out["output_size_mb"] = round(Path(output).stat().st_size / 1_048_576, 2)
    out.update({"success": success, "returncode": rc, "log": _tail(stdout)})
    if stderr.strip():
        out["stderr"] = _tail(stderr, 20)
    if extra:
        out.update(extra)
    return _ok(out)


def _check_inputs(*paths: str) -> str | None:
    for p in paths:
        if not Path(p).is_file():
            return f"File not found: {p}"
    return None


def _prepare_output(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    # Remove a stale result so success can be judged by the file's presence.
    if Path(path).is_file():
        Path(path).unlink()


def _run_and_save(inputs: list[str], ops: list[str], output: str, extra: dict | None = None) -> list[TextContent]:
    if err := _check_inputs(*inputs):
        return _err(err)
    _prepare_output(output)
    rc, stdout, stderr = cc_run(ops)
    return _run_result(rc, stdout, stderr, output, extra)


# ── Native handlers ───────────────────────────────────────────────────────────

def handle_read_metadata(args: dict) -> list[TextContent]:
    fp = args["file_path"]
    if err := _check_inputs(fp):
        return _err(err)
    try:
        xyz, _colors, _intensity, meta = _load_cloud(fp, int(args.get("max_points", _DEFAULT_SAMPLE)))
        _cloud_meta_stats(xyz, meta)
    except Exception as exc:  # noqa: BLE001
        return _err(f"Failed to read {fp}: {exc}")
    return _ok(meta)


def handle_visualize(args: dict) -> list[ImageContent | TextContent]:
    fp = args["file_path"]
    color_by = args.get("color_by", "height")
    max_pts = int(args.get("max_points", 400_000))
    if err := _check_inputs(fp):
        return _err(err)
    try:
        xyz, colors, intensity, meta = _load_cloud(fp, max_pts)
        _cloud_meta_stats(xyz, meta)
    except Exception as exc:  # noqa: BLE001
        return _err(f"Failed to read '{fp}': {exc}")

    try:
        fig = _build_figure(xyz, colors, intensity, meta, color_by, Path(fp).name)
        img_b64 = _fig_to_base64(fig)
    except Exception as exc:  # noqa: BLE001
        return _err(f"Visualization failed: {exc}")
    finally:
        try:
            import matplotlib.pyplot as plt
            plt.close("all")
        except Exception:  # noqa: BLE001
            pass

    ex, sp = meta["extent_m"], meta.get("spacing", {})
    desc = (
        f"Point cloud: {Path(fp).name} | {meta['point_count']:,} points | "
        f"extent {ex['x']:.1f} × {ex['y']:.1f} × {ex['z']:.1f} m | "
        f"median spacing {sp.get('median_point_spacing_m', 'N/A')} m | "
        f"RGB: {'yes' if meta['has_rgb'] else 'no'}"
    )
    return [
        ImageContent(type="image", data=img_b64, mimeType="image/png"),
        TextContent(type="text", text=desc + "\n\n" + json.dumps(meta, indent=2, ensure_ascii=False)),
    ]


# ── CloudCompare handlers ─────────────────────────────────────────────────────

def handle_get_cloudcompare_info(_args: dict) -> list[TextContent]:
    binary = find_cloudcompare()
    if not binary:
        return _err(
            "CloudCompare not found. Install it from https://www.cloudcompare.org "
            "and, if it is in a non-standard place, set CLOUDCOMPARE_PATH."
        )
    rc, stdout, stderr = cc_run([], timeout=120)
    version = next(
        (ln.strip() for ln in (stdout + "\n" + stderr).splitlines() if "version" in ln.lower()),
        "unknown (CloudCompare is installed; version not reported in silent mode)",
    )
    return _ok({"binary": binary, "platform": platform.system(), "version": version, "ready": True})


def handle_load_cloud_info(args: dict) -> list[TextContent]:
    fp = args["file_path"]
    if err := _check_inputs(fp):
        return _err(err)
    rc, stdout, stderr = cc_run(_open(fp))
    return _run_result(rc, stdout, stderr, extra={"file": fp, "size_mb": round(Path(fp).stat().st_size / 1_048_576, 2)})


def handle_subsample(args: dict) -> list[TextContent]:
    method = args.get("method", "SPATIAL").upper()
    param = args["parameter"]
    if method not in ("RANDOM", "SPATIAL", "OCTREE"):
        return _err(f"Unknown method {method}")
    value = str(int(param)) if method in ("RANDOM", "OCTREE") else str(param)
    out = args["output_path"]
    return _run_and_save(
        [args["input_path"]],
        [*_export_fmt(out), *_open(args["input_path"]), "-SS", method, value, *_save_clouds(out)],
        out,
    )


def handle_c2c_distances(args: dict) -> list[TextContent]:
    # CloudCompare: 1st loaded cloud = compared, 2nd = reference.
    out = args["output_path"]
    opts: list[str] = []
    if args.get("split_xyz"):
        opts.append("-SPLIT_XYZ")
    if args.get("max_distance") is not None:
        opts += ["-MAX_DIST", str(args["max_distance"])]
    return _run_and_save(
        [args["compared_path"], args["reference_path"]],
        [
            *_export_fmt(out),
            *_open(args["compared_path"]),
            *_open(args["reference_path"]),
            "-C2C_DIST", *opts,
            "-POP_CLOUDS",  # drop the reference (last loaded) so only the compared cloud is saved
            *_save_clouds(out),
        ],
        out,
    )


def handle_c2m_distances(args: dict) -> list[TextContent]:
    out = args["output_path"]
    return _run_and_save(
        [args["cloud_path"], args["mesh_path"]],
        [*_export_fmt(out), *_open(args["cloud_path"]), *_open(args["mesh_path"]), "-C2M_DIST", *_save_clouds(out)],
        out,
    )


def handle_icp(args: dict) -> list[TextContent]:
    # CloudCompare: 1st loaded entity = data (moves), 2nd = model (fixed).
    out = args["output_path"]
    opts = [
        "-OVERLAP", str(int(args.get("overlap", 100))),
        "-RANDOM_SAMPLING_LIMIT", str(int(args.get("random_sampling_limit", 50000))),
    ]
    if args.get("iterations"):
        opts += ["-ITER", str(int(args["iterations"]))]
    if args.get("min_error_diff"):
        opts += ["-MIN_ERROR_DIFF", str(args["min_error_diff"])]
    if args.get("adjust_scale"):
        opts.append("-ADJUST_SCALE")
    return _run_and_save(
        [args["data_path"], args["model_path"]],
        [
            *_export_fmt(out),
            *_open(args["data_path"]),
            *_open(args["model_path"]),
            "-ICP", *opts,
            "-POP_CLOUDS",  # drop the model (last loaded), keep the registered data cloud
            *_save_clouds(out),
        ],
        out,
        extra={"note": "The 4x4 transformation matrix and final RMS are reported in 'log'."},
    )


def handle_compute_normals(args: dict) -> list[TextContent]:
    out = args["output_path"]
    radius = str(args.get("radius", "AUTO")).upper()
    model = str(args.get("model", "QUADRIC")).upper()
    if model == "TRIANGULATION":
        model = "TRI"
    ops = ["-OCTREE_NORMALS", radius, "-MODEL", model]
    if args.get("orientation"):
        ops += ["-ORIENT", str(args["orientation"]).upper()]
    if args.get("mst_neighbors"):
        ops += ["-ORIENT_NORMS_MST", str(int(args["mst_neighbors"]))]
    return _run_and_save(
        [args["input_path"]],
        [*_export_fmt(out), *_open(args["input_path"]), *ops, *_save_clouds(out)],
        out,
        extra={"note": "Use .ply, .bin or .las (1.4) output to keep normals; .xyz/.asc keep them as extra columns."},
    )


def handle_filter_sf(args: dict) -> list[TextContent]:
    out = args["output_path"]
    sf = []
    if args.get("scalar_field") not in (None, ""):
        sf = ["-SET_ACTIVE_SF", str(args["scalar_field"])]
    return _run_and_save(
        [args["input_path"]],
        [
            *_export_fmt(out), *_open(args["input_path"]), *sf,
            "-FILTER_SF", str(args["min_val"]), str(args["max_val"]),
            *_save_clouds(out),
        ],
        out,
    )


def handle_sor(args: dict) -> list[TextContent]:
    out = args["output_path"]
    return _run_and_save(
        [args["input_path"]],
        [
            *_export_fmt(out), *_open(args["input_path"]),
            "-SOR", str(int(args.get("knn", 6))), str(args.get("n_sigma", 1.0)),
            *_save_clouds(out),
        ],
        out,
    )


def handle_merge(args: dict) -> list[TextContent]:
    out = args["output_path"]
    opens: list[str] = []
    for p in args["input_paths"]:
        opens += _open(p)
    return _run_and_save(
        list(args["input_paths"]),
        [*_export_fmt(out), *opens, "-MERGE_CLOUDS", *_save_clouds(out)],
        out,
    )


def handle_convert(args: dict) -> list[TextContent]:
    out = args["output_path"]
    return _run_and_save(
        [args["input_path"]],
        [*_export_fmt(out), *_open(args["input_path"]), *_save_clouds(out)],
        out,
        extra={"input": args["input_path"]},
    )


def handle_raw_command(args: dict) -> list[TextContent]:
    cleaned = [a for a in args["args"] if a.upper() != "-SILENT"]
    rc, stdout, stderr = cc_run(cleaned, cwd=args.get("working_directory"), managed=False)
    return _run_result(rc, stdout, stderr)


# ── Format helper ─────────────────────────────────────────────────────────────

_EXT_TO_CC: dict[str, str] = {
    ".las": "LAS", ".laz": "LAS", ".ply": "PLY", ".pcd": "PCD",
    ".xyz": "ASC", ".asc": "ASC", ".txt": "ASC", ".csv": "ASC", ".pts": "ASC",
    ".e57": "E57", ".bin": "BIN", ".vtk": "VTK", ".sbf": "SBF",
}


def _ext_flag(path: str) -> str:
    ext = Path(path).suffix.lower()
    if ext not in _EXT_TO_CC:
        raise ValueError(
            f"Unsupported output extension '{ext}'. Use one of: {', '.join(sorted(_EXT_TO_CC))}."
        )
    return _EXT_TO_CC[ext]


# ── MCP event handlers ────────────────────────────────────────────────────────

_DISPATCH = {
    "get_cloudcompare_info": handle_get_cloudcompare_info,
    "read_cloud_metadata": handle_read_metadata,
    "visualize_cloud": handle_visualize,
    "load_cloud_info": handle_load_cloud_info,
    "subsample": handle_subsample,
    "compute_cloud_to_cloud_distances": handle_c2c_distances,
    "compute_cloud_to_mesh_distances": handle_c2m_distances,
    "icp_registration": handle_icp,
    "compute_normals": handle_compute_normals,
    "filter_by_scalar_field": handle_filter_sf,
    "statistical_outlier_removal": handle_sor,
    "merge_clouds": handle_merge,
    "convert_format": handle_convert,
    "run_cloudcompare_command": handle_raw_command,
}


@server.list_tools()
async def list_tools() -> list[Tool]:
    return TOOLS


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent | ImageContent]:
    import asyncio

    handler = _DISPATCH.get(name)
    if handler is None:
        return _err(f"Unknown tool: {name}")
    try:
        # Run blocking work (CloudCompare, numpy) off the event loop.
        return await asyncio.to_thread(handler, arguments or {})
    except (FileNotFoundError, ValueError) as exc:
        return _err(str(exc))
    except Exception as exc:  # noqa: BLE001
        return _err(f"Unexpected error in '{name}': {exc}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    import asyncio
    asyncio.run(_run())


async def _run() -> None:
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    main()
