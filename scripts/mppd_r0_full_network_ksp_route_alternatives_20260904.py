import argparse
import csv
import gzip
import hashlib
import json
import math
import time
from collections import Counter
from pathlib import Path

import networkx as nx

import scripts.mppd_g0_full_network_coverage_20260904 as g0
import scripts.mppd_r0_full_network_factor_engine_20260904 as r0


def shard_of(oc, dc, nshards):
    h = hashlib.sha1(f"{oc}|{dc}".encode()).digest()
    return int.from_bytes(h[:4], 'big') % nshards


def load_od_mass(cohorts_path):
    out = Counter()
    with gzip.open(cohorts_path, 'rt', encoding='utf-8', newline='') as f:
        for row in csv.DictReader(f):
            out[(row['origin_code'], row['destination_code'])] += int(row['passenger_mass'])
    return out


def path_cost(G, path):
    return sum(float((G.get_edge_data(u, v) or {}).get('weight', 1.0)) for u, v in zip(path, path[1:]))


def enumerate_pair(G, src, dst, meta, max_candidates, detour_ratio, detour_abs):
    out = []
    truncated = False
    next_cost = None
    try:
        gen = nx.shortest_simple_paths(G, src, dst, weight='weight')
        base = None
        for path in gen:
            cost = path_cost(G, path)
            if base is None:
                base = cost
            if out and cost > base * detour_ratio and cost > base + detour_abs:
                next_cost = cost
                break
            rf, tc = r0.route_signature(path, meta)
            out.append({
                'path': path,
                'line_sequence': rf,
                'transfer_count': tc,
                'base_cost': cost,
                'generator': 'K_SHORTEST_SIMPLE_PATH',
            })
            if len(out) >= max_candidates:
                try:
                    nxt = next(gen)
                    next_cost = path_cost(G, nxt)
                    if not (next_cost > base * detour_ratio and next_cost > base + detour_abs):
                        truncated = True
                except StopIteration:
                    pass
                break
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return [], False, None
    return out, truncated, next_cost


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--p1c', required=True)
    ap.add_argument('--cohorts', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--shard-index', type=int, required=True)
    ap.add_argument('--shard-count', type=int, required=True)
    ap.add_argument('--max-candidates', type=int, default=16)
    ap.add_argument('--detour-ratio', type=float, default=1.60)
    ap.add_argument('--detour-abs', type=float, default=8.0)
    args = ap.parse_args()

    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    wall0 = time.perf_counter()
    G, meta, code_to_nodes, transfer_groups, ambiguous_seq, ambiguous_codes, graph_build = g0.build_network(args.p1c)
    od_mass = load_od_mass(args.cohorts)
    selected = [(od, m) for od, m in od_mass.items() if shard_of(od[0], od[1], args.shard_count) == args.shard_index]
    selected.sort()

    stats = Counter(); cand_hist = Counter(); tc_mass = Counter(); trunc_mass = 0
    outpath = outdir / f"ksp_routes_shard_{args.shard_index:02d}_of_{args.shard_count:02d}.jsonl.gz"
    with gzip.open(outpath, 'wt', encoding='utf-8') as fout:
        for (oc, dc), mass in selected:
            stats['od'] += 1; stats['mass'] += mass
            allc = {}; truncated_any = False; next_costs = []
            origins = code_to_nodes.get(oc, [])
            dests = code_to_nodes.get(dc, [])
            for src in origins:
                for dst in dests:
                    cands, trunc, next_cost = enumerate_pair(
                        G, src, dst, meta, args.max_candidates,
                        args.detour_ratio, args.detour_abs
                    )
                    truncated_any = truncated_any or trunc
                    if next_cost is not None: next_costs.append(next_cost)
                    for c in cands:
                        sig = tuple(c['path'])
                        old = allc.get(sig)
                        if old is None or c['base_cost'] < old['base_cost']:
                            allc[sig] = c
            cands = sorted(allc.values(), key=lambda x:(x['base_cost'], x['transfer_count'], len(x['path']), x['line_sequence']))
            # Endpoint ambiguity can multiply candidate sets. Apply the same safety
            # bound only after unioning endpoint choices and report truncation.
            endpoint_truncated = False
            if len(cands) > args.max_candidates:
                endpoint_truncated = True
                cands = cands[:args.max_candidates]
            truncated = truncated_any or endpoint_truncated
            if truncated: trunc_mass += mass; stats['truncated_od'] += 1
            if not cands: stats['unrouted_od'] += 1
            cand_hist[len(cands)] += mass
            for c in cands: tc_mass[c['transfer_count']] += mass
            fout.write(json.dumps({
                'origin_code':oc,'destination_code':dc,'passenger_mass':mass,
                'candidate_set_status':'TRUNCATED_CANDIDATE_SET' if truncated else 'DET0UR_BOUND_EXHAUSTED',
                'next_unretained_cost_min':min(next_costs) if next_costs else None,
                'candidates':cands
            },ensure_ascii=False)+'\n')

    result={
      'schema':'mppd.r0-full-network-ksp-route-alternatives-shard.v1','date':'2026-09-04',
      'status':'R0_KSP_ROUTE_ALTERNATIVES_SHARD_COMPLETED',
      'authority':'00_CURRENT_CORE_CLOSURE_WORKPLAN_V6_FULL_NETWORK_STATE_RECONSTRUCTION_20260904.md',
      'shard':{'index':args.shard_index,'count':args.shard_count},
      'scope_assertions':{'full_network':True,'line_filter':False,'segment_filter':False,'transfer_count_cap':False,'all_selected_od_use_complete_network':True},
      'route_search':{'max_candidates_safety':args.max_candidates,'detour_ratio':args.detour_ratio,'detour_abs':args.detour_abs,'algorithm':'networkx.shortest_simple_paths'},
      'od':{'count':stats['od'],'passenger_mass':stats['mass'],'unrouted_od':stats['unrouted_od'],'truncated_od':stats['truncated_od'],'truncated_passenger_mass':trunc_mass},
      'passenger_mass_by_candidate_count':dict(sorted(cand_hist.items())),
      'candidate_support_mass_by_transfer_count':dict(sorted(tc_mass.items())),
      'performance':{'wall_sec':time.perf_counter()-wall0},
      'scientific_boundary':['KSP paths are structural route hypotheses, not observed route truth.','No transfer-count cap is applied.','The max-candidates parameter is a computational safety bound only; any OD hitting it is explicitly labeled TRUNCATED_CANDIDATE_SET and cannot be treated as route-set complete.','Detour stopping is cost-based, not line/segment selection.'],
      'no_email_notification_logic':True
    }
    (outdir/f"ksp_routes_shard_{args.shard_index:02d}_summary.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
