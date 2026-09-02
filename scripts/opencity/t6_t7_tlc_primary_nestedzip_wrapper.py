#!/usr/bin/env python3
"""I/O compatibility wrapper for the official TLC taxi_zones.zip.

The official 2026-09-02 archive is valid but stores the shapefile under
`taxi_zones/taxi_zones.shp`.  Some pyogrio/GDAL builds do not auto-select a
nested shapefile when given only `zip:///path/taxi_zones.zip`.  Keep the
official ZIP bytes unchanged for provenance and SHA-256, but intercept that
single read operation, extract to a temporary directory, and pass the explicit
`.shp` path to GeoPandas.  All scientific logic remains in
`t6_t7_tlc_primary_public_registry.py`.
"""
from __future__ import annotations

import runpy
import tempfile
import zipfile
from pathlib import Path

import geopandas as gpd

_original_read_file = gpd.read_file
_tempdirs: list[tempfile.TemporaryDirectory] = []


def _read_file_compat(path, *args, **kwargs):
    if isinstance(path, str) and path.startswith("zip://"):
        archive = Path(path[len("zip://"):])
        if archive.is_file() and zipfile.is_zipfile(archive):
            with zipfile.ZipFile(archive) as zf:
                shp_names = [n for n in zf.namelist() if n.lower().endswith(".shp")]
                if len(shp_names) == 1:
                    td = tempfile.TemporaryDirectory(prefix="tlc_taxi_zones_")
                    _tempdirs.append(td)
                    zf.extractall(td.name)
                    shp = Path(td.name) / shp_names[0]
                    return _original_read_file(shp, *args, **kwargs)
    return _original_read_file(path, *args, **kwargs)


gpd.read_file = _read_file_compat
runpy.run_path(
    str(Path(__file__).with_name("t6_t7_tlc_primary_public_registry.py")),
    run_name="__main__",
)
