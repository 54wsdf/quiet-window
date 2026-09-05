from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

from scripts.rail_hz_r1b2_event_service_posterior_merge_v2_20260905 import merge_global


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--event-file', type=Path, action='append', required=True)
    ap.add_argument('--summary', type=Path, action='append', required=True)
    ap.add_argument('--roots', type=Path, required=True)
    ap.add_argument('--genealogy-global', type=Path, required=True)
    ap.add_argument('--out-events', type=Path, required=True)
    ap.add_argument('--out-summary', type=Path, required=True)
    ap.add_argument('--diagnostic', type=Path, required=True)
    a = ap.parse_args()
    diagnostic = {
        'schema': 'rail.hz-r1b2-merge-diagnostic.v1',
        'event_file_count': len(a.event_file),
        'summary_file_count': len(a.summary),
        'event_files': [str(p) for p in a.event_file],
        'summary_files': [str(p) for p in a.summary],
        'status': 'STARTED'
    }
    a.diagnostic.write_text(json.dumps(diagnostic, indent=2), encoding='utf-8')
    try:
        result = merge_global(a.event_file, a.summary, a.roots, a.genealogy_global, a.out_events, a.out_summary)
        diagnostic['status'] = 'SUCCESS'
        diagnostic['result_status'] = result.get('status')
        diagnostic['candidate_root_event_state_count'] = result.get('candidate_root_event_state_count')
        diagnostic['qualification_gates'] = result.get('qualification_gates')
        a.diagnostic.write_text(json.dumps(diagnostic, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except BaseException as exc:
        diagnostic['status'] = 'FAILED'
        diagnostic['exception_type'] = type(exc).__name__
        diagnostic['exception_repr'] = repr(exc)
        diagnostic['traceback'] = traceback.format_exc()
        a.diagnostic.write_text(json.dumps(diagnostic, ensure_ascii=False, indent=2), encoding='utf-8')
        print('R1B2_MERGE_DIAGNOSTIC_FAILURE')
        print(json.dumps(diagnostic, ensure_ascii=False, indent=2))
        raise


if __name__ == '__main__':
    main()
