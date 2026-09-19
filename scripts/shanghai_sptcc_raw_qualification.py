#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

POST_ANCHOR = [
    "11号线迪士尼","11号线康新公路","11号线秀沿路",
    "12号线漕宝路","12号线大木桥路","12号线东兰路","12号线顾戴路","12号线桂林公园",
    "12号线汉中路","12号线虹漕路","12号线虹梅路","12号线虹莘路","12号线嘉善路","12号线龙漕路",
    "12号线龙华","12号线龙华中路","12号线南京西路","12号线七莘路","12号线陕西南路",
    "13号线汉中路","13号线淮海中路","13号线江宁路","13号线马当路","13号线南京西路",
    "13号线世博大道","13号线世博会博物馆","13号线新天地","13号线自然博物馆"
]
HEADER_TOKENS={"卡ID","卡号","刷卡日期","交易日期","刷卡时间","刷卡站点","刷卡乘车类型","card_id"}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--raw",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--expected-bytes",type=int,required=True)
    ap.add_argument("--expected-data-rows",type=int,required=True)
    ap.add_argument("--expected-metro-rows",type=int,required=True)
    a=ap.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)

    h=hashlib.sha256()
    with a.raw.open("rb") as f:
        for block in iter(lambda:f.read(8*1024*1024),b""):
            h.update(block)

    post=set(POST_ANCHOR)
    physical_rows=data_rows=metro_rows=malformed=header_rows=0
    stations=Counter(); lines=Counter(); fare=Counter(); dates=Counter(); post_counts=Counter()
    card_ids=set(); min_dt=max_dt=None

    with a.raw.open("r",encoding="gb18030",errors="strict",newline="") as f:
        rd=csv.reader(f)
        for row in rd:
            physical_rows+=1
            if not row:
                continue
            if physical_rows==1 and HEADER_TOKENS.intersection(x.strip() for x in row):
                header_rows=1
                continue
            data_rows+=1
            if len(row)!=7:
                malformed+=1
                continue
            card,date,time,station,vehicle,money,prop=[x.strip().replace("\ufeff","") for x in row]
            if vehicle!="地铁":
                continue
            metro_rows+=1
            card_ids.add(card)
            stations[station]+=1
            m=re.match(r"^(\d+)号线",station)
            lines[m.group(1).zfill(2) if m else "UNPARSED"]+=1
            dates[date]+=1
            if station in post:
                post_counts[station]+=1
            try:
                x=float(money)
                fare["zero" if x==0 else "positive" if x>0 else "negative"]+=1
            except Exception:
                fare["unparsed"]+=1
            dt=f"{date} {time}"
            min_dt=dt if min_dt is None or dt<min_dt else min_dt
            max_dt=dt if max_dt is None or dt>max_dt else max_dt

    with (a.out/"station_token_counts.csv").open("w",encoding="utf-8",newline="") as f:
        w=csv.writer(f); w.writerow(["station_token","events"])
        for k,v in sorted(stations.items()): w.writerow([k,v])
    with (a.out/"post_anchor_token_counts.csv").open("w",encoding="utf-8",newline="") as f:
        w=csv.writer(f); w.writerow(["station_token","events","present"])
        for k in POST_ANCHOR: w.writerow([k,post_counts.get(k,0),bool(post_counts.get(k,0))])

    result={
        "schema":"mppd.shanghai.sptcc.raw-qualification.v1",
        "source_file":a.raw.name,
        "source_bytes":a.raw.stat().st_size,
        "sha256":h.hexdigest(),
        "encoding":"gb18030_strict",
        "physical_csv_rows":physical_rows,
        "header_rows":header_rows,
        "data_rows":data_rows,
        "malformed_7col_rows":malformed,
        "metro_rows":metro_rows,
        "unique_metro_cards":len(card_ids),
        "metro_station_tokens":len(stations),
        "metro_line_prefix_counts":dict(sorted(lines.items())),
        "metro_date_counts":dict(sorted(dates.items())),
        "metro_time_range":[min_dt,max_dt],
        "metro_fare_sign_counts":dict(fare),
        "post_anchor_registry_tokens_expected":len(POST_ANCHOR),
        "post_anchor_registry_tokens_present":sum(1 for k in POST_ANCHOR if post_counts.get(k,0)>0),
        "post_anchor_registry_event_mass":sum(post_counts.values()),
        "post_anchor_present_tokens":[k for k in POST_ANCHOR if post_counts.get(k,0)>0],
        "post_anchor_absent_tokens":[k for k in POST_ANCHOR if post_counts.get(k,0)==0],
        "expected_checks":{
            "bytes":{"expected":a.expected_bytes,"actual":a.raw.stat().st_size,"pass":a.raw.stat().st_size==a.expected_bytes},
            "data_rows":{"expected":a.expected_data_rows,"actual":data_rows,"pass":data_rows==a.expected_data_rows},
            "metro_rows":{"expected":a.expected_metro_rows,"actual":metro_rows,"pass":metro_rows==a.expected_metro_rows}
        },
        "privacy":"No card identifier, raw row, or cross-event passenger linkage is persisted in outputs.",
        "claim_boundary":"Raw-source qualification only; no pairing, stitching, route inference or passenger-service posterior."
    }
    (a.out/"qualification.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(result,ensure_ascii=False,indent=2))
    if not all(x["pass"] for x in result["expected_checks"].values()):
        raise SystemExit("expected source-scale check failed")

if __name__=="__main__":
    main()
