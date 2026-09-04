import argparse
import csv
import io
import json
import math
import statistics
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import scripts.mppd_g0_full_network_coverage_20260904 as g0

START = g0.START
END = g0.END
RAIL = g0.RAIL


def rep_time(vals):
    if not vals:
        return None
    xs = sorted(v.timestamp() for v in vals)
    return datetime.fromtimestamp(statistics.median(xs))


def pearson(x, y):
    if len(x) < 3:
        return 0.0
    mx = sum(x) / len(x)
    my = sum(y) / len(y)
    vx = sum((a - mx) ** 2 for a in x)
    vy = sum((b - my) ** 2 for b in y)
    if vx <= 0 or vy <= 0:
        return 0.0
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / math.sqrt(vx * vy)


def robust_z(v, vals):
    if not vals:
        return 0.0
    med = statistics.median(vals)
    mad = statistics.median(abs(x - med) for x in vals)
    return (v - med) / max(1.0, 1.4826 * mad)


def minute_floor(t):
    return t.replace(second=0, microsecond=0)


def nms(items, sep_sec):
    chosen = []
    for t, score in sorted(items, key=lambda z: (z[1], -z[0].timestamp()), reverse=True):
        if all(abs((t - u).total_seconds()) >= sep_sec for u, _ in chosen):
            chosen.append((t, score))
    return sorted(chosen)


def classify_direction(station_times, meta):
    pairs = []
    for st, t in station_times.items():
        m = meta.get(st)
        if m and m.get("seq") is not None:
            pairs.append((m["seq"], t.timestamp()))
    pairs.sort()
    if len(pairs) < 3:
        return None
    c = pearson([a for a, _ in pairs], [b for _, b in pairs])
    if abs(c) < 0.30:
        return None
    return "INC" if c > 0 else "DEC"


def load_service_events(service_path, meta):
    raw = defaultdict(lambda: defaultdict(list))
    with open(service_path, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            line = str(r.get("subway_id") or "").strip()
            tr = str(r.get("btrain_no") or "").strip()
            st = str(r.get("statn_id") or "").strip()
            a = g0.dt(r.get("arrival_detect_time"))
            d = g0.dt(r.get("departure_detect_time"))
            if not line or not tr or not st or not a or not d:
                continue
            n = g0.node(line, st)
            if n not in meta:
                continue
            raw[(line, tr)][n].append((a, d))
    trains = {}
    for key, by_st in raw.items():
        sts = {}
        for n, vals in by_st.items():
            arr = rep_time([a for a, _ in vals])
            dep = rep_time([d for _, d in vals])
            if arr and dep:
                sts[n] = {"arrival": arr, "departure": dep}
        if len(sts) >= 2:
            trains[key] = sts
    return trains


def load_afc_hist(taims_path, code_to_nodes, meta):
    entry_hist = defaultdict(Counter)
    exit_hist = defaultdict(Counter)
    entry_mass = Counter()
    exit_mass = Counter()
    eligible = 0
    mapped = 0
    with zipfile.ZipFile(taims_path) as z:
        f = g0.zr(z, "VW_KSCC_DX_CARD.csv")
        for r in csv.DictReader(f):
            if str(r.get("TRNS_MNS_CD") or "").strip() not in RAIL:
                continue
            t0 = g0.dt(r.get("RIDE_DTM"))
            t1 = g0.dt(r.get("ALGH_DTM"))
            if not t0 or not t1 or not (START <= t0 < END) or t1 <= t0 or t1 - t0 > timedelta(hours=3):
                continue
            eligible += 1
            oc = g0.z4(r.get("RIDE_BSST_ID"))
            dc = g0.z4(r.get("ALGH_BSST_ID"))
            ons = [n for n in code_to_nodes.get(oc, []) if n in meta]
            dns = [n for n in code_to_nodes.get(dc, []) if n in meta]
            if not ons or not dns:
                continue
            mapped += 1
            w0 = 1.0 / len(ons)
            for n in ons:
                entry_hist[n][minute_floor(t0)] += w0
                entry_mass[n] += w0
            if START <= t1 < END:
                w1 = 1.0 / len(dns)
                for n in dns:
                    exit_hist[n][minute_floor(t1)] += w1
                    exit_mass[n] += w1
        f.close()
    return entry_hist, exit_hist, entry_mass, exit_mass, {"eligible_rows": eligible, "mapped_endpoint_rows": mapped}


def build_line_models(trains, meta, exit_hist, exit_mass):
    by_line_train = defaultdict(dict)
    for (line, tr), sts in trains.items():
        station_times = {n: z["departure"] for n, z in sts.items()}
        direction = classify_direction(station_times, meta)
        by_line_train[line][tr] = {"stations": sts, "direction": direction}

    models = {}
    for line in sorted({m["line"] for m in meta.values()}):
        line_trains = by_line_train.get(line, {})
        by_dir = defaultdict(list)
        for tr, obj in line_trains.items():
            if obj["direction"]:
                by_dir[obj["direction"]].append(tr)
        dirs = {}
        for direction in ("INC", "DEC"):
            trs = by_dir.get(direction, [])
            coverage = Counter(n for tr in trs for n in line_trains[tr]["stations"])
            ref = coverage.most_common(1)[0][0] if coverage else None
            offsets_raw = defaultdict(list)
            if ref:
                for tr in trs:
                    sts = line_trains[tr]["stations"]
                    if ref not in sts:
                        continue
                    rt = sts[ref]["departure"]
                    for n, ev in sts.items():
                        offsets_raw[n].append((ev["departure"] - rt).total_seconds())
            offsets = {n: statistics.median(v) for n, v in offsets_raw.items() if len(v) >= 2}
            intercepts = {}
            for tr in trs:
                vals = []
                for n, ev in line_trains[tr]["stations"].items():
                    if n in offsets:
                        vals.append(ev["departure"] - timedelta(seconds=offsets[n]))
                x = rep_time(vals)
                if x:
                    intercepts[tr] = x

            lag = {}
            quality = {}
            for n in offsets:
                if exit_mass[n] < 20:
                    continue
                observed = Counter(minute_floor(line_trains[tr]["stations"][n]["arrival"]) for tr in trs if n in line_trains[tr]["stations"])
                vals_hist = [exit_hist[n].get(START + timedelta(minutes=i), 0.0) for i in range(180)]
                best = None
                for l in range(11):
                    xs, ys = [], []
                    for i in range(180):
                        m = START + timedelta(minutes=i)
                        xs.append(observed.get(m, 0.0))
                        ys.append(exit_hist[n].get(m + timedelta(minutes=l), 0.0))
                    c = pearson(xs, ys)
                    cand = (c, -l)
                    if best is None or cand > best[0]:
                        best = (cand, l)
                lag[n] = best[1]
                quality[n] = best[0][0]

            observed_refs = sorted(intercepts.values())
            gaps = [(b - a).total_seconds() for a, b in zip(observed_refs, observed_refs[1:]) if 60 <= (b - a).total_seconds() <= 1800]
            median_headway = statistics.median(gaps) if gaps else 360.0
            dirs[direction] = {
                "ref": ref,
                "offsets": offsets,
                "intercepts": intercepts,
                "lag": lag,
                "quality": quality,
                "observed_train_count": len(intercepts),
                "median_observed_headway_sec": median_headway,
            }
        models[line] = dirs
    return models, by_line_train


def infer_candidates(models, exit_hist, exit_mass):
    all_candidates = []
    line_summary = []
    for line, dirs in models.items():
        for direction, md in dirs.items():
            offsets = md["offsets"]
            usable = []
            for n, off in offsets.items():
                if n not in md["lag"] or exit_mass[n] < 20:
                    continue
                histvals = [exit_hist[n].get(START + timedelta(minutes=i), 0.0) for i in range(180)]
                weight = math.sqrt(max(1.0, exit_mass[n])) * max(0.05, md["quality"].get(n, 0.0))
                usable.append((n, off, md["lag"][n], histvals, weight))
            scores = []
            t = START + timedelta(seconds=30)
            while t < END:
                num = 0.0
                den = 0.0
                for n, off, lag, histvals, weight in usable:
                    em = minute_floor(t + timedelta(seconds=off)) + timedelta(minutes=lag)
                    z = robust_z(exit_hist[n].get(em, 0.0), histvals)
                    num += weight * max(0.0, z)
                    den += weight
                scores.append((t, num / den if den else 0.0))
                t += timedelta(seconds=30)

            positive = [s for _, s in scores if s > 0]
            threshold = None
            if positive:
                sp = sorted(positive)
                threshold = sp[round((len(sp) - 1) * 0.90)]
            sep = min(720.0, max(90.0, 0.55 * md["median_observed_headway_sec"]))
            peaks = nms([(t, s) for t, s in scores if threshold is not None and s >= threshold], sep)
            observed = list(md["intercepts"].values())
            inferred = []
            for t0, score in peaks:
                if any(abs((t0 - x).total_seconds()) <= 90 for x in observed):
                    continue
                inferred.append((t0, score))
                events = []
                for n, off in offsets.items():
                    tt = t0 + timedelta(seconds=off)
                    if START - timedelta(minutes=15) <= tt < END + timedelta(minutes=15):
                        events.append({"node": n, "time": tt.isoformat()})
                all_candidates.append({
                    "line": line,
                    "direction": direction,
                    "candidate_id": f"AFC_{line}_{direction}_{t0.strftime('%H%M%S')}",
                    "intercept_time": t0.isoformat(),
                    "ridge_score": score,
                    "evidence_class": "AFC_INFERRED_SERVICE_FIELD",
                    "station_events": events,
                })
            line_summary.append({
                "line": line,
                "direction": direction,
                "observed_train_count": md["observed_train_count"],
                "usable_station_count": len(usable),
                "median_observed_headway_sec": md["median_observed_headway_sec"],
                "ridge_threshold": threshold,
                "inferred_candidate_count": len(inferred),
                "status": "BOOTSTRAPPED" if usable else "UNRESOLVED_LATENT_SERVICE_FIELD",
            })
    return all_candidates, line_summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--taims", required=True)
    ap.add_argument("--p1c", required=True)
    ap.add_argument("--service", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    G, meta, code_to_nodes, transfer_groups, ambiguous_seq, ambiguous_codes, graph_build = g0.build_network(args.p1c)
    trains = load_service_events(args.service, meta)
    entry_hist, exit_hist, entry_mass, exit_mass, afc_stats = load_afc_hist(args.taims, code_to_nodes, meta)
    models, by_line_train = build_line_models(trains, meta, exit_hist, exit_mass)
    inferred, line_summary = infer_candidates(models, exit_hist, exit_mass)

    observed_rows = []
    for (line, tr), sts in trains.items():
        direction = by_line_train[line][tr]["direction"]
        observed_rows.append({
            "line": line,
            "train": tr,
            "direction": direction,
            "evidence_class": "PARTIAL_DIRECT_SERVICE_ANCHOR",
            "station_event_count": len(sts),
            "station_events": [
                {"node": n, "arrival": ev["arrival"].isoformat(), "departure": ev["departure"].isoformat()}
                for n, ev in sorted(sts.items())
            ],
        })

    all_lines = sorted({m["line"] for m in meta.values()})
    unresolved = [x for x in line_summary if x["status"] != "BOOTSTRAPPED"]
    result = {
        "schema": "mppd.g1-full-network-service-field-bootstrap.v1",
        "date": "2026-09-04",
        "status": "G1_FULL_NETWORK_SERVICE_FIELD_BOOTSTRAP_COMPLETED",
        "time_window": "2026-08-29 07:00-10:00",
        "scope_assertions": {
            "line_filter_applied": False,
            "segment_filter_applied": False,
            "all_network_lines_entered_bootstrap": True,
            "missing_direct_anchor_line_dropped": False,
            "inferred_service_evidence_labeled": "AFC_INFERRED_SERVICE_FIELD",
        },
        "network": {
            "nodes": G.number_of_nodes(),
            "edges": G.number_of_edges(),
            "lines": len(all_lines),
            "line_ids": all_lines,
            "transfer_groups": len(transfer_groups),
        },
        "afc": afc_stats,
        "observed_service": {
            "train_keys": len(observed_rows),
            "lines_with_direct_anchor": len({x["line"] for x in observed_rows}),
        },
        "afc_inferred_service": {
            "candidate_count": len(inferred),
            "lines_with_inferred_candidates": len({x["line"] for x in inferred}),
        },
        "line_direction_summary": line_summary,
        "unresolved_line_direction_states": unresolved,
        "hard_boundary": [
            "This is an initialization/bootstrap field, not final realized ATS reconstruction.",
            "PARTIAL_DIRECT_SERVICE_ANCHOR remains partial direct operational evidence.",
            "AFC-derived candidates are explicitly AFC_INFERRED_SERVICE_FIELD and never OBSERVED_ATS.",
            "No network line is removed because direct service anchors are missing.",
            "The ridge bootstrap is line-conditional initialization only; G3 must jointly update all service states with full-network passenger route/transfer loops.",
        ],
        "next_gate": "Build G2 arbitrary-transfer route-family and passenger-chain posterior over the complete network using this full-network service-field initialization, then jointly update S and station-time kernels in G3.",
        "no_email_notification_logic": True,
    }

    (outdir / "g1_full_network_service_field_bootstrap_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (outdir / "g1_full_network_inferred_service_field.json").write_text(json.dumps({"schema": "mppd.afc-inferred-service-field.v1", "evidence_class": "AFC_INFERRED_SERVICE_FIELD", "candidates": inferred}, ensure_ascii=False), encoding="utf-8")
    (outdir / "g1_full_network_direct_service_anchors.json").write_text(json.dumps({"schema": "mppd.partial-direct-service-anchors.v1", "evidence_class": "PARTIAL_DIRECT_SERVICE_ANCHOR", "trains": observed_rows}, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
