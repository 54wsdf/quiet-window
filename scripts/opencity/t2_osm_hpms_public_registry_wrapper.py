#!/usr/bin/env python3
"""Compatibility wrapper for the T2 OSM-HPMS public-registry rehearsal."""
from __future__ import annotations
import runpy
from pathlib import Path
import geopandas as gpd
from shapely.geometry import box

def _from_bbox(cls, bbox, crs=None):
    return gpd.GeoSeries([box(*bbox)], crs=crs)

if not hasattr(gpd.GeoSeries, "from_bbox"):
    gpd.GeoSeries.from_bbox = classmethod(_from_bbox)

runpy.run_path(str(Path(__file__).with_name("t2_osm_hpms_public_registry.py")), run_name="__main__")
