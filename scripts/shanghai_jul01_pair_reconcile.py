#!/usr/bin/env python3
"""Exact raw-row reconciliation between legacy strict and hardened formal pairing.

No card identifier or pair-level row is persisted. Card keys are full SHA-256
inside runner-local SQLite only. The output is aggregate counts sufficient to
explain the Jul01 gate-segment delta.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,sqlite3
from collections import Counter
from datetime import datetime
from pathlib import Path

FIELDS=["card_id","date","time","station","vehicle","money","property"]

def clean(x): return (x or "").replace("\ufeff","").strip()

def card_hash(x): return hashlib.sha256(clean(x).encode()).hexdigest()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input",type=Path,required=True)
    ap.add_argument("--db",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--source-date",required=True)
    ap.add_argument("--expected-strict",type=int,required=True)
    ap.add_argument("--expected-formal",type=int,required=True)
    a=ap.parse_args()

    con=sqlite3.connect(a.db); cur=con.cursor()
    cur.executescript(
      "PRAGMA journal_mode=WAL;PRAGMA synchronous=NORMAL;PRAGMA temp_store=FILE;"
      "DROP TABLE IF EXISTS e;"
      "CREATE TABLE e(card TEXT,ts INTEGER,raw_row INTEGER,metro INTEGER,station TEXT,fare REAL);"
    )
    epoch=datetime(1970,1,1); batch=[]; source=Counter()
    with a.input.open("r",encoding="gb18030",errors="strict",newline="") as f:
        rd=csv.reader(f); first=True
        for physical,row in enumerate(rd,start=1):
            if first:
                first=False
                if row and "刷卡日期" in row:
                    continue
            source["data_rows"]+=1
            if len(row)!=7:
                source["malformed"]+=1; continue
            card,date,time,station,vehicle,money,prop=[clean(x) for x in row]
            if date!=a.source_date:
                source["other_date"]+=1; continue
            ts=int((datetime.strptime(date+" "+time,"%Y-%m-%d %H:%M:%S")-epoch).total_seconds())
            metro=int(vehicle=="地铁")
            fare=float(money) if metro else None
            if metro: source["metro_rows"]+=1
            batch.append((card_hash(card),ts,physical,metro,station,fare))
            if len(batch)>=50000:
                cur.executemany("INSERT INTO e VALUES(?,?,?,?,?,?)",batch);con.commit();batch.clear()
    if batch:
        cur.executemany("INSERT INTO e VALUES(?,?,?,?,?,?)",batch);con.commit();batch.clear()
    cur.execute("CREATE INDEX idx_e_card_ts_row ON e(card,ts,raw_row)");con.commit()

    q=cur.execute("SELECT card,ts,raw_row,metro,station,fare FROM e ORDER BY card,ts,raw_row")
    total=Counter()
    current=None; events=[]
    def process(ev):
        if not ev:return
        # ev tuple: card,ts,row,metro,station,fare, already all-mode time/raw-row order
        all_ts=Counter(x[1] for x in ev)
        metro=[x for x in ev if x[3]]
        metro_order=sorted(metro,key=lambda x:(x[1],x[4],x[5],x[2]))
        metro_ts=Counter(x[1] for x in metro_order)
        strict=set()
        for x,y in zip(metro_order,metro_order[1:]):
            if x[5]==0 and y[5] is not None and y[5]>0 and y[1]>x[1] and y[1]-x[1]<=21600 and metro_ts[x[1]]==1 and metro_ts[y[1]]==1:
                strict.add((x[2],y[2]))
        formal=set(); open_e=None
        for x in ev:
            if all_ts[x[1]]>1:
                open_e=None; continue
            if not x[3]:
                open_e=None; continue
            fare=x[5]
            if fare==0:
                open_e=x
            elif fare is not None and fare>0:
                if open_e is None: continue
                if x[1]>open_e[1] and x[1]-open_e[1]<=21600:
                    formal.add((open_e[2],x[2]))
                open_e=None
        total["strict"]+=len(strict); total["formal"]+=len(formal)
        inter=strict&formal; so=strict-formal; fo=formal-strict
        total["intersection"]+=len(inter); total["strict_only"]+=len(so); total["formal_only"]+=len(fo)
        pos={x[2]:i for i,x in enumerate(ev)}
        byrow={x[2]:x for x in ev}
        for er,xr in so:
            e=byrow[er]; x=byrow[xr]
            pe,px=pos[er],pos[xr]
            between=ev[pe+1:px] if pe<px else []
            has_nonmetro=any(not z[3] for z in between)
            endpoint_allmode_collision=(all_ts[e[1]]>1 or all_ts[x[1]]>1)
            if has_nonmetro and endpoint_allmode_collision:
                total["strict_only_nonmetro_between_and_endpoint_collision"]+=1
            elif has_nonmetro:
                total["strict_only_nonmetro_between"]+=1
            elif endpoint_allmode_collision:
                total["strict_only_endpoint_allmode_same_timestamp_collision"]+=1
            else:
                total["strict_only_unexplained"]+=1

    for row in q:
        if current is not None and row[0]!=current:
            process(events);events=[]
        current=row[0];events.append(row)
    process(events)

    explained=(total["strict_only_nonmetro_between"]+
      total["strict_only_endpoint_allmode_same_timestamp_collision"]+
      total["strict_only_nonmetro_between_and_endpoint_collision"])
    result={
      "schema":"mppd.shanghai.jul01.strict-vs-formal-pair-reconciliation.v1",
      "source_date":a.source_date,
      "source":dict(source),
      "pair_counts":dict(total),
      "explained_strict_only":explained,
      "strict_only_explained_share":explained/total["strict_only"] if total["strict_only"] else 1.0,
      "checks":{
        "strict_matches_expected":total["strict"]==a.expected_strict,
        "formal_matches_expected":total["formal"]==a.expected_formal,
        "formal_is_subset_of_strict":total["formal_only"]==0,
        "all_strict_only_explained":total["strict_only_unexplained"]==0 and explained==total["strict_only"],
        "delta_identity":total["strict"]-total["formal"]==total["strict_only"]-total["formal_only"]
      },
      "privacy":"No card hash, card ID or pair-level row is written to output.",
      "claim_boundary":"Pairing reconciliation only; virtual-transfer stitching and passenger-service posterior are outside this audit."
    }
    a.out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(result,ensure_ascii=False,indent=2))
    if not all(result["checks"].values()):
        raise SystemExit("pair reconciliation gate failed")
    con.close()

if __name__=="__main__":main()
