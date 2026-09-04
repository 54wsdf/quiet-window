import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import scripts.mppd_g0_full_network_coverage_20260904 as g0
import scripts.mppd_g1_full_network_service_bootstrap_20260904 as g1


def root_score(line, train, direction, salt):
    key = f"{salt}|{line}|{train}|{direction or 'UNKNOWN'}".encode()
    return int(hashlib.sha256(key).hexdigest()[:16], 16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--service', required=True)
    ap.add_argument('--p1c', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--holdout-rate', type=float, default=0.10)
    ap.add_argument('--min-train-per-stratum', type=int, default=3)
    ap.add_argument('--salt', default='R1D_DIRECT_SERVICE_ROOT_HOLDOUT_V1')
    args = ap.parse_args()
    if not 0 < args.holdout_rate < 0.5:
        raise ValueError('holdout-rate must be in (0,0.5)')

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    G, meta, *_ = g0.build_network(args.p1c)
    trains = g1.load_service_events(args.service, meta)
    by_stratum = defaultdict(list)
    root_meta = {}
    for (line, train), sts in trains.items():
        direction = g1.classify_direction({n: ev['departure'] for n, ev in sts.items()}, meta)
        key = (line, direction or 'UNKNOWN')
        by_stratum[key].append((line, train))
        root_meta[(line, train)] = {'line': line, 'train': train, 'direction': direction, 'station_event_count': len(sts)}

    holdout = set(); stratum_rows = []
    for (line, direction), roots in sorted(by_stratum.items()):
        roots = sorted(roots, key=lambda rt: root_score(rt[0], rt[1], direction, args.salt))
        n = len(roots)
        cap = max(0, n - args.min_train_per_stratum)
        desired = max(1, round(args.holdout_rate * n)) if n >= args.min_train_per_stratum + 1 else 0
        k = min(cap, desired)
        chosen = roots[:k]
        holdout.update(chosen)
        stratum_rows.append({'line': line, 'direction': direction, 'root_count': n, 'holdout_root_count': k, 'training_root_count': n-k})

    fields = None; train_rows = 0; held_rows = 0
    train_path = out/'service_training_roots.csv'; held_path = out/'service_heldout_roots.csv'
    with open(args.service, encoding='utf-8', newline='') as src:
        reader = csv.DictReader(src); fields = reader.fieldnames
        if not fields: raise RuntimeError('service CSV has no header')
        with train_path.open('w', encoding='utf-8', newline='') as tf, held_path.open('w', encoding='utf-8', newline='') as hf:
            tw = csv.DictWriter(tf, fieldnames=fields); hw = csv.DictWriter(hf, fieldnames=fields); tw.writeheader(); hw.writeheader()
            for row in reader:
                root = (str(row.get('subway_id') or '').strip(), str(row.get('btrain_no') or '').strip())
                if root in holdout:
                    hw.writerow(row); held_rows += 1
                else:
                    tw.writerow(row); train_rows += 1

    held_anchors = []
    for root in sorted(holdout):
        line, train = root; sts = trains[root]; md = root_meta[root]
        held_anchors.append({
            'line': line, 'train': train, 'direction': md['direction'],
            'evidence_class': 'HELD_OUT_PARTIAL_DIRECT_SERVICE_ANCHOR_R1D',
            'station_event_count': len(sts),
            'station_events': [
                {'node': n, 'arrival': ev['arrival'].isoformat(), 'departure': ev['departure'].isoformat()}
                for n, ev in sorted(sts.items())
            ],
        })

    held_by_line = Counter(x['line'] for x in held_anchors); held_events_by_line = Counter()
    for x in held_anchors: held_events_by_line[x['line']] += x['station_event_count']
    manifest = {
        'schema': 'mppd.r1d-direct-service-root-holdout-split.v1', 'date': '2026-09-05',
        'status': 'R1D_DIRECT_SERVICE_ROOT_HOLDOUT_SPLIT_COMPLETED_NO_RECONSTRUCTION_YET',
        'policy': {
            'unit': 'WHOLE_LINE_TRAIN_ROOT', 'stratification': 'LINE_X_DIRECTION_WITH_UNKNOWN_AS_EXPLICIT_STRATUM',
            'holdout_rate': args.holdout_rate, 'min_training_roots_per_stratum': args.min_train_per_stratum,
            'salt': args.salt, 'event_level_leakage_forbidden': True,
        },
        'counts': {
            'eligible_direct_train_root_count': len(trains), 'holdout_train_root_count': len(holdout),
            'training_train_root_count': len(trains)-len(holdout), 'holdout_station_event_count': sum(x['station_event_count'] for x in held_anchors),
            'training_csv_row_count': train_rows, 'heldout_csv_row_count': held_rows,
        },
        'holdout_train_count_by_line': dict(sorted(held_by_line.items())),
        'holdout_event_count_by_line': dict(sorted(held_events_by_line.items())),
        'strata': stratum_rows,
        'outputs': {'training_service_csv': train_path.name, 'heldout_service_csv': held_path.name, 'heldout_anchor_json': 'heldout_direct_service_anchors.json'},
        'scientific_boundary': [
            'The holdout is applied before G1 service bootstrap so held train roots cannot enter training as PARTIAL_DIRECT_SERVICE_ANCHOR.',
            'AFC remains available to reconstruct a held train root; recovery from AFC/passenger constraints is the object of the held-out validation and is not leakage.',
            'All raw CSV rows belonging to a held line/train root are removed together; station-event-level partial withholding is forbidden.',
            'This split alone does not qualify Gate A. The complete G1->G1F->R1B/R1C chain must be rebuilt from the training service CSV before withheld events are scored.'
        ],
        'no_email_notification_logic': True,
    }
    (out/'heldout_direct_service_anchors.json').write_text(json.dumps({'schema':'mppd.r1d-heldout-direct-service-anchors.v1','evidence_class':'HELD_OUT_PARTIAL_DIRECT_SERVICE_ANCHOR_R1D','trains':held_anchors}, ensure_ascii=False), encoding='utf-8')
    (out/'holdout_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == '__main__': main()
