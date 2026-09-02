#!/usr/bin/env python3
"""T2 public_data registry rehearsal: OSM primary input + FHWA HPMS truth.

The public task card defines OpenStreetMap as the primary source and FHWA HPMS
as held-out truth after conflation.  This script constructs a deterministic
Southern California instance, conflates OSM highway ways to HPMS lines using a
strict spatial/orientation filter, masks 30% of each HPMS target independently,
and evaluates a fixed mode-imputation-by-OSM-class baseline.

Because both source layers are public, this is a reproducible public rehearsal,
not a recreation of organizer-secret held-out attributes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path

import geopandas as gpd
import numpy as np
import osmium
import pandas as pd
import pyogrio
from shapely.geometry import LineString
from sklearn.metrics import balanced_accuracy_score

SEED = 20260902
BBOX = (-118.70, 33.55, -117.45, 34.45)
ALLOWED_HIGHWAY = {
    "motorway", "motorway_link", "trunk", "trunk_link", "primary", "primary_link",
    "secondary", "secondary_link", "tertiary", "tertiary_link", "unclassified",
    "residential", "living_street", "service",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def parse_num(x):
    if x is None:
        return np.nan
    m = re.search(r"\d+(?:\.\d+)?", str(x))
    return float(m.group()) if m else np.nan


def parse_speed_mph(x):
    v = parse_num(x)
    if not np.isfinite(v):
        return np.nan
    s = str(x).lower()
    if "km" in s or ("mph" not in s and v > 75):
        return v / 1.609344
    return v


def orientation_deg(geom) -> float:
    if geom is None or geom.is_empty:
        return np.nan
    try:
        if geom.geom_type == "MultiLineString":
            geom = max(geom.geoms, key=lambda g: g.length)
        c = list(geom.coords)
        if len(c) < 2:
            return np.nan
        dx = c[-1][0] - c[0][0]
        dy = c[-1][1] - c[0][1]
        if dx == 0 and dy == 0:
            return np.nan
        return (math.degrees(math.atan2(dy, dx)) + 180.0) % 180.0
    except Exception:
        return np.nan


def orientation_diff(a, b) -> float:
    if not np.isfinite(a) or not np.isfinite(b):
        return np.nan
    d = abs(a - b) % 180.0
    return min(d, 180.0 - d)


class HighwayCollector(osmium.SimpleHandler):
    def __init__(self, bbox, max_ways=400000):
        super().__init__()
        self.minx, self.miny, self.maxx, self.maxy = bbox
        self.max_ways = max_ways
        self.rows = []

    def way(self, w):
        if len(self.rows) >= self.max_ways:
            return
        highway = w.tags.get("highway")
        if highway not in ALLOWED_HIGHWAY:
            return
        coords = []
        inside = False
        try:
            for n in w.nodes:
                lon = n.lon
                lat = n.lat
                coords.append((lon, lat))
                if self.minx <= lon <= self.maxx and self.miny <= lat <= self.maxy:
                    inside = True
        except osmium.InvalidLocationError:
            return
        if not inside or len(coords) < 2:
            return
        try:
            geom = LineString(coords)
        except Exception:
            return
        if geom.is_empty:
            return
        self.rows.append({
            "osm_way_id": int(w.id),
            "osm_highway": highway,
            "osm_lanes": parse_num(w.tags.get("lanes")),
            "osm_maxspeed_mph": parse_speed_mph(w.tags.get("maxspeed")),
            "osm_name": w.tags.get("name") or "",
            "osm_ref": w.tags.get("ref") or "",
            "geometry": geom,
        })


def read_osm(pbf: Path) -> gpd.GeoDataFrame:
    h = HighwayCollector(BBOX)
    h.apply_file(str(pbf), locations=True)
    if not h.rows:
        raise RuntimeError("no OSM highway ways collected in bbox")
    g = gpd.GeoDataFrame(h.rows, geometry="geometry", crs=4326)
    clip = gpd.GeoSeries.from_bbox(BBOX, crs=4326).iloc[0]
    g = g[g.intersects(clip)].copy()
    g["geometry"] = g.geometry.intersection(clip)
    g = g[~g.geometry.is_empty].copy().to_crs(32611)
    g["osm_length_m"] = g.geometry.length
    return g[g["osm_length_m"] >= 30].copy()


def read_hpms(path: Path) -> gpd.GeoDataFrame:
    cols = ["objectid", "route_id", "f_system", "facility_type", "speed_limit", "through_lanes"]
    h = pyogrio.read_dataframe(path, columns=cols, bbox=BBOX)
    if h.crs is None:
        h = h.set_crs(4326)
    else:
        h = h.to_crs(4326)
    h["hpms_objectid"] = pd.to_numeric(h["objectid"], errors="coerce")
    h["hpms_f_system"] = pd.to_numeric(h["f_system"], errors="coerce")
    h["hpms_through_lanes"] = pd.to_numeric(h["through_lanes"], errors="coerce")
    h["hpms_speed_limit"] = pd.to_numeric(h["speed_limit"], errors="coerce")
    h = h.dropna(subset=["hpms_objectid", "hpms_f_system", "hpms_through_lanes", "geometry"]).copy()
    h = h[h["hpms_f_system"].between(1, 7) & h["hpms_through_lanes"].between(1, 12)].copy()
    h["hpms_objectid"] = h["hpms_objectid"].astype(int)
    h["hpms_f_system"] = h["hpms_f_system"].astype(int)
    h["hpms_through_lanes"] = h["hpms_through_lanes"].astype(int)
    return h.to_crs(32611)


def mode_map(train: pd.DataFrame, group_col: str, target_col: str):
    global_mode = int(train[target_col].mode().iloc[0])
    m = {}
    for k, g in train.groupby(group_col, dropna=False):
        md = g[target_col].mode()
        if len(md):
            m[k] = int(md.iloc[0])
    return m, global_mode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--osm", type=Path, required=True)
    ap.add_argument("--hpms", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--target-matches", type=int, default=20000)
    ap.add_argument("--max-distance-m", type=float, default=15.0)
    ap.add_argument("--max-orientation-diff-deg", type=float, default=25.0)
    args = ap.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)

    osm = read_osm(args.osm)
    hpms = read_hpms(args.hpms)
    osm["osm_orientation_deg"] = osm.geometry.map(orientation_deg)
    hpms["hpms_orientation_deg"] = hpms.geometry.map(orientation_deg)

    left = osm.sort_values("osm_way_id").reset_index(drop=True)
    right = hpms[["hpms_objectid", "hpms_f_system", "hpms_through_lanes", "hpms_speed_limit", "hpms_orientation_deg", "geometry"]].copy()
    matched = gpd.sjoin_nearest(left, right, how="inner", max_distance=args.max_distance_m, distance_col="match_distance_m")
    matched["orientation_diff_deg"] = [orientation_diff(a, b) for a, b in zip(matched["osm_orientation_deg"], matched["hpms_orientation_deg"])]
    matched = matched[matched["orientation_diff_deg"].le(args.max_orientation_diff_deg)].copy()
    matched = matched.sort_values(["match_distance_m", "orientation_diff_deg", "osm_way_id", "hpms_objectid"])
    matched = matched.drop_duplicates("hpms_objectid", keep="first")
    matched = matched.sort_values(["osm_way_id", "hpms_objectid"]).head(args.target_matches).copy()
    if len(matched) < 500:
        raise RuntimeError(f"only {len(matched)} high-confidence OSM-HPMS matches; T2 minimum is 500 links")

    rng_lane = np.random.default_rng(SEED + 2)
    rng_class = np.random.default_rng(SEED + 3)
    n = len(matched)
    lane_mask = np.zeros(n, dtype=bool)
    class_mask = np.zeros(n, dtype=bool)
    lane_mask[rng_lane.choice(n, size=int(round(0.30 * n)), replace=False)] = True
    class_mask[rng_class.choice(n, size=int(round(0.30 * n)), replace=False)] = True
    matched["lane_truth_withheld"] = lane_mask
    matched["road_class_truth_withheld"] = class_mask

    lane_train = matched[~matched["lane_truth_withheld"]]
    class_train = matched[~matched["road_class_truth_withheld"]]
    lane_map, lane_global = mode_map(lane_train, "osm_highway", "hpms_through_lanes")
    class_map, class_global = mode_map(class_train, "osm_highway", "hpms_f_system")
    matched["pred_through_lanes"] = [lane_map.get(x, lane_global) for x in matched["osm_highway"]]
    matched["pred_f_system"] = [class_map.get(x, class_global) for x in matched["osm_highway"]]

    lane_y = matched.loc[lane_mask, "hpms_through_lanes"].astype(int)
    lane_p = matched.loc[lane_mask, "pred_through_lanes"].astype(int)
    class_y = matched.loc[class_mask, "hpms_f_system"].astype(int)
    class_p = matched.loc[class_mask, "pred_f_system"].astype(int)
    lane_bal = float(balanced_accuracy_score(lane_y, lane_p))
    class_bal = float(balanced_accuracy_score(class_y, class_p))
    lane_acc = float((lane_y.to_numpy() == lane_p.to_numpy()).mean())
    class_acc = float((class_y.to_numpy() == class_p.to_numpy()).mean())

    completed = matched[[
        "osm_way_id", "hpms_objectid", "osm_highway", "osm_lanes", "osm_maxspeed_mph",
        "hpms_f_system", "hpms_through_lanes", "hpms_speed_limit", "match_distance_m",
        "orientation_diff_deg", "lane_truth_withheld", "road_class_truth_withheld",
        "pred_through_lanes", "pred_f_system",
    ]].copy()
    completed["input_hpms_through_lanes"] = np.where(completed["lane_truth_withheld"], np.nan, completed["hpms_through_lanes"])
    completed["input_hpms_f_system"] = np.where(completed["road_class_truth_withheld"], np.nan, completed["hpms_f_system"])
    completed.to_csv(out / "completed_link_table.csv", index=False)
    matched[["osm_way_id", "hpms_objectid", "match_distance_m", "orientation_diff_deg", "osm_highway", "hpms_f_system", "hpms_through_lanes"]].to_csv(out / "conflation_audit.csv", index=False)

    summary = {
        "task": "T2 road-network attribute inference",
        "evidence_class": "public_data registry / official public source rehearsal",
        "primary_source": "OpenStreetMap Southern California extract",
        "truth_source": "FHWA HPMS California 2018 layer 0",
        "study_bbox_lonlat": list(BBOX),
        "source_sha256": {"osm_pbf": sha256(args.osm), "hpms_geojson": sha256(args.hpms)},
        "conflation": {
            "osm_candidate_ways": int(len(osm)),
            "hpms_candidate_lines": int(len(hpms)),
            "high_confidence_matches": int(len(matched)),
            "max_distance_m": args.max_distance_m,
            "max_orientation_diff_deg": args.max_orientation_diff_deg,
            "median_match_distance_m": float(matched["match_distance_m"].median()),
            "p95_match_distance_m": float(matched["match_distance_m"].quantile(.95)),
            "median_orientation_diff_deg": float(matched["orientation_diff_deg"].median()),
        },
        "masking": {"lane_truth_fraction": float(lane_mask.mean()), "road_class_truth_fraction": float(class_mask.mean()), "seed": SEED, "targets_masked_independently": True},
        "baseline": "mode_imputation_by_osm_highway_class_from_visible_HPMS_cells",
        "metrics": {
            "lane_balanced_accuracy": lane_bal,
            "road_class_balanced_accuracy": class_bal,
            "E_mean_balanced_accuracy": float((lane_bal + class_bal) / 2),
            "lane_raw_accuracy": lane_acc,
            "road_class_raw_accuracy": class_acc,
            "lane_masked_n": int(lane_mask.sum()),
            "road_class_masked_n": int(class_mask.sum()),
        },
        "status": "PUBLIC_OSM_HPMS_CONFLATION_REHEARSAL_CLOSED",
        "formal_boundary": "OSM and HPMS are both public and temporally mismatched (OSM 2026 snapshot versus HPMS 2018 truth). This public replay validates the conflation/masking/evaluation pipeline but does not reproduce organizer-secret held-out attributes or same-date ground truth.",
        "provenance_separation": {"teacher_or_asu_data_used": False, "self_added_data_used": False, "self_added_parameters": {"bbox": list(BBOX), "max_distance_m": args.max_distance_m, "max_orientation_diff_deg": args.max_orientation_diff_deg, "seed": SEED}},
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
