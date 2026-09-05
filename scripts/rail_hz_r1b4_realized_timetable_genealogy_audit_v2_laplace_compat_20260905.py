from __future__ import annotations

import gzip
import json

import scripts.rail_hz_r1b4_realized_timetable_genealogy_audit_20260905 as base


def load_event_map_compat(paths):
    out = {}
    for p in paths:
        with gzip.open(p, "rt", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                e = json.loads(line)
                if "realized_time_sd_s" not in e:
                    if "realized_time_sd_laplace_s" not in e:
                        raise SystemExit(f"service-event uncertainty field missing in {p}")
                    e["realized_time_sd_s"] = float(e["realized_time_sd_laplace_s"])
                key = (str(e["root_id"]), int(e["station"]))
                if key in out:
                    raise SystemExit(f"duplicate R1B3 root/station event: {key}")
                out[key] = e
    if len(out) != 43584:
        raise SystemExit(f"expected 43584 root-event rows, found {len(out)}")
    return out


def main():
    base.load_event_map = load_event_map_compat
    base.main()


if __name__ == "__main__":
    main()
