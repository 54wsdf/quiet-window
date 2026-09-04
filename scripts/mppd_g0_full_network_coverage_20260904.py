import argparse
import csv
import io
import json
import math
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import networkx as nx

STRICT_TIERS = {"A_TRANSFER_CODE_LINE_NAME", "B_UNIQUE_NAME", "C_NAME_PREFIX_LINE"}
TRANSFER_TIER = "A_TRANSFER_CODE_LINE_NAME"
RAIL = {"201", "202", "203", "204", "205", "206", "208", "209", "210", "231", "232", "235", "236", "237", "290"}
START = datetime(2026, 8, 29, 7, 0)
END = datetime(2026, 8, 29, 10, 0)


def dt(v):
    d = "".join(c for c in str(v or "") if c.isdigit())
    try:
        return datetime.strptime(d[:14], "%Y%m%d%H%M%S") if len(d) >= 14 else None
    except Exception:
        return None


def z4(v):
    d = "".join(c for c in str(v or "") if c.isdigit())
    return d.zfill(4) if d and len(d) <= 4 else (d[-4:] if d else "")


def num(v):
    try:
        return int(str(v).strip())
    except Exception:
        return None


def node(line, station):
    return f"{line}|{station}"


def detect(raw):
    for enc in ("utf-8-sig", "utf-8", "cp949", "euc-kr"):
        try:
            raw.decode(enc)
            return enc
        except UnicodeDecodeError:
            pass
    return "cp949"


def zr(z, name):
    f = z.open(name)
    head = f.read(262144)
    f.close()
    return io.TextIOWrapper(z.open(name), encoding=detect(head), errors="replace", newline="")


def build_network(p1c_path):
    p = json.loads(Path(p1c_path).read_text(encoding="utf-8"))
    entries = [
        x for x in p["canonical_entries"]
        if x.get("tier") in STRICT_TIERS and x.get("service_subway_id") and x.get("service_statn_id")
    ]

    meta = {}
    by_line_seq = defaultdict(lambda: defaultdict(set))
    by_dv = defaultdict(dict)
    code_candidates = defaultdict(dict)

    for x in entries:
        line = str(x["service_subway_id"])
        st = str(x["service_statn_id"])
        n = node(line, st)
        seq = num(x.get("service_statn_sn"))
        dv = str(x.get("dv_name") or "").strip()
        service_name = str(x.get("service_name") or "").strip()
        tier = str(x.get("tier") or "")
        current = meta.get(n)
        if current is None or tier == TRANSFER_TIER:
            meta[n] = {
                "line": line,
                "station": st,
                "seq": seq,
                "dv_name": dv,
                "service_name": service_name,
                "tier": tier,
            }
        if seq is not None:
            by_line_seq[line][seq].add(n)
        if tier == TRANSFER_TIER and dv:
            by_dv[dv][n] = x
        code = str(x.get("out_stn_num") or "").strip()
        if code:
            old = code_candidates[code].get(n)
            rank = 0 if tier == TRANSFER_TIER else (1 if tier == "B_UNIQUE_NAME" else 2)
            if old is None or rank < old:
                code_candidates[code][n] = rank

    G = nx.Graph()
    for n, m in meta.items():
        G.add_node(n, **m)

    ambiguous_sequence_groups = []
    uncertain_inline_edges = 0
    certain_inline_edges = 0
    for line, groups in by_line_seq.items():
        seqs = sorted(groups)
        for seq in seqs:
            if len(groups[seq]) > 1:
                ambiguous_sequence_groups.append({"line": line, "seq": seq, "nodes": sorted(groups[seq])})
        for a, b in zip(seqs, seqs[1:]):
            if b - a != 1:
                continue
            uncertain = len(groups[a]) != 1 or len(groups[b]) != 1
            for u in groups[a]:
                for v in groups[b]:
                    if u == v:
                        continue
                    G.add_edge(
                        u,
                        v,
                        kind="inline_uncertain" if uncertain else "inline",
                        weight=1.15 if uncertain else 1.0,
                        line=line,
                        evidence="P1C_SERVICE_STATN_SN_AMBIGUOUS" if uncertain else "P1C_SERVICE_STATN_SN",
                    )
                    if uncertain:
                        uncertain_inline_edges += 1
                    else:
                        certain_inline_edges += 1
        if line == "1002" and seqs:
            for u in groups[seqs[0]]:
                for v in groups[seqs[-1]]:
                    if u != v:
                        G.add_edge(u, v, kind="inline", weight=1.0, line=line, evidence="LINE2_CIRCULAR_CLOSURE")

    transfer_groups = []
    transfer_edges = 0
    for dv, nodes_map in by_dv.items():
        vals = [n for n in nodes_map if n in G]
        lines = {meta[n]["line"] for n in vals}
        if len(lines) < 2:
            continue
        transfer_groups.append({"physical_station": dv, "members": sorted(vals), "lines": sorted(lines)})
        for i in range(len(vals)):
            for j in range(i + 1, len(vals)):
                u, v = vals[i], vals[j]
                if meta[u]["line"] == meta[v]["line"]:
                    continue
                if not G.has_edge(u, v):
                    transfer_edges += 1
                G.add_edge(u, v, kind="transfer", weight=2.5, evidence=TRANSFER_TIER, physical_station=dv)

    code_to_nodes = {}
    ambiguous_codes = []
    for code, cand in code_candidates.items():
        if not cand:
            continue
        best_rank = min(cand.values())
        nodes = sorted(n for n, rank in cand.items() if rank == best_rank)
        code_to_nodes[code] = nodes
        if len(nodes) > 1:
            ambiguous_codes.append({"external_station_code": code, "candidate_nodes": nodes})

    return G, meta, code_to_nodes, transfer_groups, ambiguous_sequence_groups, ambiguous_codes, {
        "certain_inline_edges_constructed": certain_inline_edges,
        "uncertain_inline_edges_constructed": uncertain_inline_edges,
        "transfer_edges_constructed": transfer_edges,
    }


def load_service(service_path):
    by_line_train = defaultdict(set)
    by_line_station = Counter()
    rows = 0
    with open(service_path, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            line = str(r.get("subway_id") or "").strip()
            st = str(r.get("statn_id") or "").strip()
            tr = str(r.get("btrain_no") or "").strip()
            a = dt(r.get("arrival_detect_time"))
            d = dt(r.get("departure_detect_time"))
            if line and st and tr and a and d:
                rows += 1
                by_line_train[line].add(tr)
                by_line_station[(line, st)] += 1
    return rows, by_line_train, by_line_station


def load_afc(taims_path, code_to_nodes):
    code_pair_bins = Counter()
    eligible = 0
    mapped_codes = 0
    unmapped_origin = 0
    unmapped_destination = 0
    unmapped_both = 0
    duration_bins = Counter()

    with zipfile.ZipFile(taims_path) as z:
        f = zr(z, "VW_KSCC_DX_CARD.csv")
        for r in csv.DictReader(f):
            if str(r.get("TRNS_MNS_CD") or "").strip() not in RAIL:
                continue
            t0 = dt(r.get("RIDE_DTM"))
            t1 = dt(r.get("ALGH_DTM"))
            if not t0 or not t1 or not (START <= t0 < END) or t1 <= t0 or t1 - t0 > timedelta(hours=3):
                continue
            eligible += 1
            oc = z4(r.get("RIDE_BSST_ID"))
            dc = z4(r.get("ALGH_BSST_ID"))
            oh = oc in code_to_nodes
            dh = dc in code_to_nodes
            if not oh and not dh:
                unmapped_both += 1
                continue
            if not oh:
                unmapped_origin += 1
                continue
            if not dh:
                unmapped_destination += 1
                continue
            mapped_codes += 1
            bin5 = (t0.hour * 60 + t0.minute) // 5
            dur5 = int((t1 - t0).total_seconds() // 300)
            duration_bins[dur5] += 1
            code_pair_bins[(oc, dc, bin5)] += 1
        f.close()

    return code_pair_bins, {
        "eligible_rows": eligible,
        "mapped_endpoint_rows": mapped_codes,
        "unmapped_origin_rows": unmapped_origin,
        "unmapped_destination_rows": unmapped_destination,
        "unmapped_both_rows": unmapped_both,
        "duration_5min_bin_counts": dict(duration_bins),
    }


def path_line_sequence(path, meta):
    seq = []
    for n in path:
        line = meta[n]["line"]
        if not seq or seq[-1] != line:
            seq.append(line)
    return tuple(seq)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--taims", required=True)
    ap.add_argument("--p1c", required=True)
    ap.add_argument("--service", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    G, meta, code_to_nodes, transfer_groups, ambiguous_seq, ambiguous_codes, graph_build = build_network(args.p1c)
    service_rows, service_trains_by_line, service_station_rows = load_service(args.service)
    code_pair_bins, afc_stats = load_afc(args.taims, code_to_nodes)

    components = list(nx.connected_components(G))
    component_of = {}
    for i, comp in enumerate(components):
        for n in comp:
            component_of[n] = i

    # Cache complete single-source weighted paths. The network is small enough that this
    # avoids per-OD repeated Dijkstra while preserving the full spatial graph.
    path_cache = {}
    def paths_from(o):
        if o not in path_cache:
            path_cache[o] = nx.single_source_dijkstra_path(G, o, weight="weight")
        return path_cache[o]

    od_mass = Counter()
    od_bins = Counter()
    for (oc, dc, b), mass in code_pair_bins.items():
        od_mass[(oc, dc)] += mass
        od_bins[(oc, dc, b)] += mass

    resolved_codepair = {}
    route_gap_pairs = []
    route_family_mass = Counter()
    route_family_od = Counter()
    transfer_count_mass = Counter()
    line_usage_mass = Counter()
    line_sequence_examples = defaultdict(list)
    routed_mass = 0
    unrouted_mass = 0
    ambiguous_endpoint_choice_pairs = 0

    for (oc, dc), mass in od_mass.items():
        origins = code_to_nodes.get(oc, [])
        dests = code_to_nodes.get(dc, [])
        if len(origins) > 1 or len(dests) > 1:
            ambiguous_endpoint_choice_pairs += 1
        best = None
        for o in origins:
            pmap = paths_from(o)
            for d in dests:
                path = pmap.get(d)
                if not path:
                    continue
                cost = 0.0
                for u, v in zip(path, path[1:]):
                    cost += float((G.get_edge_data(u, v) or {}).get("weight", 1.0))
                key = (cost, len(path), o, d, path)
                if best is None or key[:4] < best[:4]:
                    best = key
        if best is None:
            unrouted_mass += mass
            route_gap_pairs.append({
                "origin_code": oc,
                "destination_code": dc,
                "passenger_mass": mass,
                "origin_candidates": "|".join(origins),
                "destination_candidates": "|".join(dests),
                "origin_components": "|".join(str(component_of.get(x, -1)) for x in origins),
                "destination_components": "|".join(str(component_of.get(x, -1)) for x in dests),
            })
            continue
        _, _, o, d, path = best
        routed_mass += mass
        seq = path_line_sequence(path, meta)
        rf = ">".join(seq)
        resolved_codepair[(oc, dc)] = (o, d, path, seq)
        route_family_mass[rf] += mass
        route_family_od[rf] += 1
        tc = max(0, len(seq) - 1)
        transfer_count_mass[tc] += mass
        for line in set(seq):
            line_usage_mass[line] += mass
        if len(line_sequence_examples[rf]) < 5:
            line_sequence_examples[rf].append({"origin_code": oc, "destination_code": dc, "mass": mass})

    route_rows = []
    for rf, mass in route_family_mass.most_common():
        route_rows.append({
            "line_sequence": rf,
            "transfer_count": max(0, len(rf.split(">")) - 1),
            "od_pairs": route_family_od[rf],
            "passenger_mass": mass,
            "examples": json.dumps(line_sequence_examples[rf], ensure_ascii=False),
        })

    line_rows = []
    all_lines = sorted({m["line"] for m in meta.values()})
    station_count_by_line = Counter(m["line"] for m in meta.values())
    for line in all_lines:
        line_rows.append({
            "line": line,
            "station_nodes": station_count_by_line[line],
            "partial_direct_anchor_trains": len(service_trains_by_line.get(line, set())),
            "partial_direct_anchor_rows": sum(v for (l, _), v in service_station_rows.items() if l == line),
            "routed_passenger_mass_using_line": line_usage_mass[line],
        })

    def write_csv(path, rows, fallback):
        fields = list(rows[0]) if rows else fallback
        with path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            if rows:
                w.writerows(rows)

    write_csv(outdir / "g0_full_network_route_families.csv", route_rows, ["line_sequence", "transfer_count", "od_pairs", "passenger_mass", "examples"])
    write_csv(outdir / "g0_full_network_unrouted_codepairs.csv", route_gap_pairs, ["origin_code", "destination_code", "passenger_mass", "origin_candidates", "destination_candidates", "origin_components", "destination_components"])
    write_csv(outdir / "g0_full_network_line_coverage.csv", line_rows, ["line", "station_nodes", "partial_direct_anchor_trains", "partial_direct_anchor_rows", "routed_passenger_mass_using_line"])

    category_mass = {
        "same_line": int(transfer_count_mass.get(0, 0)),
        "one_transfer": int(transfer_count_mass.get(1, 0)),
        "two_transfer": int(transfer_count_mass.get(2, 0)),
        "three_or_more_transfer": int(sum(v for k, v in transfer_count_mass.items() if k >= 3)),
    }

    requested_examples = {}
    for pattern in ("1002", "1002>1007", "1002>1005", "1007>1002", "1005>1002", "1007>1002>1005", "1005>1002>1007"):
        requested_examples[pattern] = {
            "od_pairs": int(route_family_od.get(pattern, 0)),
            "passenger_mass": int(route_family_mass.get(pattern, 0)),
        }

    result = {
        "schema": "mppd.g0-full-network-coverage.v1",
        "date": "2026-09-04",
        "status": "G0_FULL_NETWORK_COVERAGE_COMPLETED",
        "time_window": "2026-08-29 07:00-10:00",
        "scope_assertions": {
            "spatial_line_filter_applied": False,
            "segment_filter_applied": False,
            "all_eligible_rail_rows_counted": True,
            "unmapped_rows_silently_dropped": False,
            "unrouted_rows_silently_dropped": False,
            "same_line_one_transfer_double_transfer_higher_transfer_in_scope": True,
        },
        "network": {
            "nodes": G.number_of_nodes(),
            "edges": G.number_of_edges(),
            "lines": len(all_lines),
            "line_ids": all_lines,
            "transfer_groups": len(transfer_groups),
            "connected_components": len(components),
            "largest_component_nodes": max((len(c) for c in components), default=0),
            "ambiguous_sequence_groups_retained_as_uncertain_edges": len(ambiguous_seq),
            "ambiguous_external_station_codes": len(ambiguous_codes),
            **graph_build,
        },
        "afc": {
            **afc_stats,
            "mapped_unique_code_pairs": len(od_mass),
            "routed_passenger_mass": routed_mass,
            "unrouted_passenger_mass": unrouted_mass,
            "routed_share_of_mapped_endpoint_mass": routed_mass / (routed_mass + unrouted_mass) if routed_mass + unrouted_mass else 0.0,
            "ambiguous_endpoint_choice_codepairs": ambiguous_endpoint_choice_pairs,
        },
        "service_anchor": {
            "partial_direct_event_rows": service_rows,
            "lines_with_anchor_trains": sum(1 for line in all_lines if service_trains_by_line.get(line)),
            "anchor_train_keys": int(sum(len(v) for v in service_trains_by_line.values())),
        },
        "route_family": {
            "unique_line_sequences": len(route_family_mass),
            "transfer_count_mass": {str(k): int(v) for k, v in sorted(transfer_count_mass.items())},
            "category_mass": category_mass,
            "top_50": route_rows[:50],
            "explicit_pattern_census": requested_examples,
        },
        "coverage_gaps": {
            "unmapped_endpoint_rows": afc_stats["unmapped_origin_rows"] + afc_stats["unmapped_destination_rows"] + afc_stats["unmapped_both_rows"],
            "unrouted_code_pairs": len(route_gap_pairs),
            "unrouted_mass": unrouted_mass,
            "rule": "Coverage gaps are retained as G0 evidence and must be repaired or represented as uncertain-route state before full-network joint certification; they are not removed by selecting a better-connected subnetwork.",
        },
        "scientific_boundary": [
            "This is complete-network input/route-family coverage qualification, not yet service inversion.",
            "Ambiguous same-line sequence groups are connected with explicitly uncertain P1C sequence edges rather than being spatially discarded.",
            "Physical transfer edges require highest-tier P1C transfer-code/name identity.",
            "The current route-family census uses one minimum-cost structural path per mapped OD code pair as a bootstrap; G2 must replace this with route-family posterior alternatives.",
            "BMS is partial direct service-anchor evidence, not exhaustive ATS truth.",
            "No raw card identifiers are retained.",
        ],
        "next_gate": "Use the complete-network G0 census to build G1 service-field bootstrap on every line, then G2 route-family posterior with multiple path alternatives. No spatial subnetwork certification is permitted.",
        "no_email_notification_logic": True,
    }

    (outdir / "g0_full_network_coverage_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
