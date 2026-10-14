# cloudcompare-mcp (cave / 3D survey fork)

Model Context Protocol (MCP) server for [CloudCompare](https://www.cloudcompare.org) — lets Claude (Desktop or Code) inspect and process point clouds and meshes in natural language.

Fork of [yufeioptimal/cloudcompare-mcp](https://github.com/yufeioptimal/cloudcompare-mcp) (MIT), fixed and tuned for enclosed 3D surveys such as caves, tunnels and hypogea.

## What changed from upstream (v0.2.0 → v0.3.0)

**Bugs fixed (CloudCompare command line)** — checked against CloudCompare's own `ccCommandLineCommands.cpp`:

| Tool | Upstream problem | Fix |
|------|------------------|-----|
| all | `mcp` 2.x removed the decorator API, so a fresh `uvx` install crashed at startup | pinned `mcp>=1.3,<2` |
| all | CloudCompare auto-saves after every step by default → stray timestamped files written next to your inputs | `-AUTO_SAVE OFF`, explicit save only |
| all | `-SAVE_CLOUDS FILE` splits on spaces → paths like `C:\Users\Mario Rossi\…` broke | file names are quoted |
| all | no global shift → precision loss with UTM / national-grid coordinates | `-O -GLOBAL_SHIFT AUTO` |
| all | silent mode prints nothing → no feedback | CloudCompare log (`-LOG_FILE`) returned to Claude; success checked on the written file |
| `compute_cloud_to_cloud_distances` | reference and compared clouds swapped (CC: 1st = compared, 2nd = reference); save failed with 2 clouds loaded | correct order, reference dropped before saving, optional `split_xyz` |
| `icp_registration` | model and data swapped (CC: 1st = data that moves, 2nd = fixed model); save failed with 2 clouds loaded | correct order, model dropped before saving, `min_error_diff`, `adjust_scale` |
| `compute_normals` | used `-COMPUTE_NORMALS`, which is only a load-time flag for structured scans — never computed anything | `-OCTREE_NORMALS <radius\|AUTO> -MODEL LS\|QUADRIC\|TRI`, optional orientation and MST re-orientation |
| `filter_by_scalar_field` | always filtered the active field (often none) | `scalar_field` option (`-SET_ACTIVE_SF`) |
| `get_cloudcompare_info` | called a non-existent `--version` flag | reads the log |
| `convert_format` | `.obj` silently written as PLY | unsupported extensions are rejected with a clear message |

**Native tools (no CloudCompare needed):**

- **3D density.** Upstream reported points / XY bounding-box area — meaningless for a cave, a shaft or any enclosed space. Now: median / mean / p95 **nearest-neighbour spacing** and **surface density** (pts/m² of wall, same definition as CloudCompare's), robust to subsampling.
- **Large files.** LAS/LAZ are streamed in chunks with random sampling, so multi-GB scans don't need to fit in memory. Point count and bounding box come from the header (exact).
- **Real intensity colouring** (upstream used Z as a "proxy").
- Plan + two elevations at true scale (equal aspect), CRS shown when the LAS has one.
- Blocking work runs off the MCP event loop; long CloudCompare jobs time out after 2 h (`CLOUDCOMPARE_TIMEOUT` in seconds).

## Tools

| Tool | CloudCompare needed | Description |
|------|:---:|-------------|
| `read_cloud_metadata` | – | Point count, bbox, extent, 3D spacing & surface density, attributes |
| `visualize_cloud` | – | PNG with plan, two elevations and metadata panel, coloured by height / RGB / intensity |
| `get_cloudcompare_info` | ✓ | Check installation |
| `load_cloud_info` | ✓ | Open any CC-readable file (E57, BIN, PTX…) and return the log |
| `subsample` | ✓ | Random / spatial (min spacing) / octree |
| `compute_cloud_to_cloud_distances` | ✓ | C2C, e.g. survey vs re-survey |
| `compute_cloud_to_mesh_distances` | ✓ | C2M signed distances |
| `icp_registration` | ✓ | Align a cloud onto a fixed reference |
| `compute_normals` | ✓ | Octree normals (LS / quadric / triangulation) |
| `filter_by_scalar_field` | ✓ | Keep points within a value range of a chosen field |
| `statistical_outlier_removal` | ✓ | SOR noise filter |
| `merge_clouds` | ✓ | Merge several clouds |
| `convert_format` | ✓ | LAS/LAZ, PLY, PCD, XYZ/ASC/TXT, E57, BIN, VTK, SBF |
| `run_cloudcompare_command` | ✓ | Any CloudCompare command line (advanced) |

All CloudCompare tools work on files in batch (CloudCompare's command-line mode). They do not drive an open CloudCompare window.

## Install

Requirements: Python ≥ 3.10, [uv](https://docs.astral.sh/uv/) (recommended), CloudCompare ≥ 2.12 for the CloudCompare tools.

### Claude Desktop

Settings → Developer → Edit config, then add to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "cloudcompare": {
      "command": "uvx",
      "args": ["--from", "git+https://github.com/Stratos915/cloudcompare-mcp", "cloudcompare-mcp"]
    }
  }
}
```

Restart Claude Desktop. If CloudCompare is not in a standard location, add:

```json
      "env": { "CLOUDCOMPARE_PATH": "C:\\Program Files\\CloudCompare\\CloudCompare.exe" }
```

On macOS Claude Desktop may not see `uvx` on its PATH; use the full path (`which uvx`, e.g. `/Users/you/.local/bin/uvx`).

### Claude Code

```bash
claude mcp add cloudcompare -- uvx --from git+https://github.com/Stratos915/cloudcompare-mcp cloudcompare-mcp
```

### CloudCompare search order

1. `CLOUDCOMPARE_PATH`
2. `cloudcompare` / `CloudCompare` on PATH
3. Windows `C:\Program Files\CloudCompare\CloudCompare.exe` · macOS `/Applications/CloudCompare.app/Contents/MacOS/CloudCompare` · Linux `/usr/bin/cloudcompare`, `/snap/bin/cloudcompare`

## Examples

- "Show me `D:\Rilievi\grotta.laz` coloured by intensity and tell me the point spacing."
- "Subsample it to 2 cm and remove noise with SOR, save as `grotta_2cm.laz`."
- "Compare the 2026 survey with the 2024 one (reference) and keep only points that moved more than 5 cm."

## Testing status

Native tools are tested on synthetic tunnel clouds (spacing and surface density match the ground truth within 1 %, also from a 4 % subsample). CloudCompare command lines are generated per the CloudCompare 2.13 source and checked with a mock binary; they have not yet been run against a real CloudCompare install — please report issues.

## License

MIT — original work © yufeioptimal; modifications © Stratos915.
