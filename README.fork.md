# SHARPpy Point-Forecast Fork

This repository is a customized SHARPpy fork focused on interactive point-forecast soundings and workflow improvements for research and operational analysis.

It is based on upstream SHARPpy, with local changes that currently include:

- Point-click forecast soundings for `HRRR` and `GFS`
- `ERA5` point soundings using Zarr-backed data access
- AWS/remote-data forecast profile loading instead of station-only model sounding selection for supported models
- `Advanced Diagnostics` replacing the SARS inset in the sounding interface
- Startup, window-placement, and shutdown stability fixes for the GUI

## Scope

This fork is intended to preserve the SHARPpy analysis experience while extending the model sounding workflow toward arbitrary clicked points rather than station-bound forecast profiles.

Current point-forecast models:

- `HRRR`
- `GFS`
- `ERA5`

## Status

This is an actively modified fork and is not a drop-in representation of upstream SHARPpy behavior.

Some changes are substantial and may not yet be suitable for direct upstream merge without being broken into smaller pull requests.

## Installation

The recommended local setup for this fork is a Python virtual environment.

Example:

```bash
cd /path/to/your/sharppy-fork
python3.10 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

Additional dependencies used by the point-forecast backends may include:

- `cfgrib`
- `eccodes`
- `xarray`
- `zarr`
- `gcsfs`
- `dask`
- `PySide2`

## Running

From the repository root:

```bash
source .venv/bin/activate
QT_QPA_PLATFORM=xcb sharppy
```

If you are on a system where Qt/X11 defaults are problematic, forcing `QT_QPA_PLATFORM=xcb` is recommended.

## Major Differences From Upstream

### Point Forecast Workflow

For supported models, the map can be used as a true point picker. A user can click an arbitrary location and generate a sounding from interpolated gridded forecast or reanalysis data.

### ERA5 Support

ERA5 is exposed as a point-forecast option using Zarr-backed access for efficient retrieval.

### Diagnostics Panel

The old SARS inset has been removed from the sounding UI in this fork and replaced with `Advanced Diagnostics`, which currently shows:

- `SFC-100m SRH`
- `SFC-250m SRH`
- `SFC-500m SRH`

## Recommended GitHub Positioning

This repository is best treated as a maintained fork of SHARPpy.

If contributing back upstream, consider opening smaller pull requests for isolated changes such as:

- crash/stability fixes
- UI cleanup
- standalone diagnostics additions
- config migration fixes

For large workflow changes like point-forecast backends and datasource redesign, keeping them in the fork until they are better separated is the safer approach.

## Attribution

Upstream SHARPpy project:

- GitHub: https://github.com/sharppy/SHARPpy

If you publish research or products using this fork, keep the upstream SHARPpy citation and attribution intact where appropriate.
