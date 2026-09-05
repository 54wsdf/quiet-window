from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import pyarrow.parquet as pq


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args()
    table = pq.read_table(a.input)
    rows = table.to_pylist()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(a.output, 'wt', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n')
    print(json.dumps({'status':'CONVERTED_R1B2_EVENT_AGGREGATE','rows':len(rows),'output':str(a.output)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
