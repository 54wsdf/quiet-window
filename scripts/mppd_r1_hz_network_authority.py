from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts.mppd_r1_joint_validation_v2 import NETWORK_AUTHORITY_SCHEMA

PATHS: dict[str, dict[str, Any]] = {
    "B_main": {"line_id": "B", "stations": list(range(0, 28))},
    "B_branch": {"line_id": "B", "stations": list(range(0, 21)) + list(range(28, 34))},
    "C_main": {"line_id": "C", "stations": list(range(34, 67))},
    "A_main": {"line_id": "A", "stations": [67, 68, 69, 70, 71, 72, 73, 74, 5, 75, 76, 77, 46, 78, 79, 80, 15, 16]},
}


def option_tuple(x: dict[str, Any]) -> tuple[str, str]:
    return str(x["path_id"]), str(x["direction"])


def path_line(path_id: str) -> str:
    if path_id not in PATHS:
        raise ValueError(f"route support references unknown path_id={path_id}")
    return str(PATHS[path_id]["line_id"])


def build_authority(route_support: dict[str, Any]) -> dict[str, Any]:
    if route_support.get("status") != "QUALIFIED_LINE_AWARE_ROUTE_SUPPORT":
        raise ValueError("route support must be QUALIFIED_LINE_AWARE_ROUTE_SUPPORT")

    line_paths = []
    for path_id, meta in PATHS.items():
        base = [str(x) for x in meta["stations"]]
        for direction in ("Down", "Up"):
            line_paths.append(
                {
                    "path_id": path_id,
                    "line_id": str(meta["line_id"]),
                    "direction_id": direction,
                    "station_ids": base if direction == "Down" else list(reversed(base)),
                }
            )

    transfer_keys: dict[tuple[str, str, str, str, str], set[tuple[str, str]]] = {}
    route_count = 0
    route_support_map = route_support.get("route_support", {})
    for candidates in route_support_map.values():
        if not isinstance(candidates, list):
            continue
        for route in candidates:
            if not isinstance(route, dict):
                continue
            route_count += 1
            legs = route.get("ride_legs", [])
            for left, right in zip(legs, legs[1:]):
                if int(left["to_station"]) != int(right["from_station"]):
                    continue
                station = str(int(left["to_station"]))
                left_opts = [option_tuple(x) for x in left.get("compatible_service_options", [])]
                right_opts = [option_tuple(x) for x in right.get("compatible_service_options", [])]
                for p1, d1 in left_opts:
                    for p2, d2 in right_opts:
                        if (p1, d1) == (p2, d2):
                            # Remaining on the same service hypothesis is not a transfer movement.
                            continue
                        l1 = path_line(p1)
                        l2 = path_line(p2)
                        key = (station, l1, d1, l2, d2)
                        transfer_keys.setdefault(key, set()).add((p1, p2))

    transfer_movements = []
    for i, (key, path_pairs) in enumerate(sorted(transfer_keys.items())):
        station, from_line, from_direction, to_line, to_direction = key
        transfer_movements.append(
            {
                "movement_id": f"hz:m{i}:{station}:{from_line}:{from_direction}->{to_line}:{to_direction}",
                "station_id": station,
                "from_line_id": from_line,
                "from_direction_id": from_direction,
                "to_line_id": to_line,
                "to_direction_id": to_direction,
                "structural_path_pairs": [
                    {"from_path_id": a, "to_path_id": b} for a, b in sorted(path_pairs)
                ],
            }
        )

    authority = {
        "schema": NETWORK_AUTHORITY_SCHEMA,
        "dataset_id": "CN_HZ_Tianchi_2019",
        "authority_role": "NETWORK_AND_TRANSFER_PHYSICAL_DOMAIN_ONLY",
        "station_ids": [str(x) for x in range(81)],
        "line_paths": line_paths,
        "transfer_movements": transfer_movements,
        "source_route_support_status": route_support.get("status"),
        "source_route_candidate_count": route_count,
        "semantics": {
            "contains_service_event_inventory": False,
            "contains_expected_service_count": False,
            "contains_planned_timetable": False,
            "service_count_must_be_inferred": True,
            "transfer_path_component_count_must_be_inferred": True,
        },
    }
    return authority


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--route-support", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    raw = json.loads(a.route_support.read_text(encoding="utf-8"))
    authority = build_authority(raw)
    a.output.write_text(json.dumps(authority, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "schema": authority["schema"],
                "station_count": len(authority["station_ids"]),
                "line_path_direction_count": len(authority["line_paths"]),
                "transfer_movement_count": len(authority["transfer_movements"]),
                "source_route_candidate_count": authority["source_route_candidate_count"],
                "contains_service_count": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
