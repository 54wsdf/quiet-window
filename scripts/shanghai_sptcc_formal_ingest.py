#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,gzip,hashlib,json,re,sqlite3
from collections import Counter
from datetime import datetime,timedelta
from pathlib import Path

FIELDS=["card_id","date","time","station","vehicle","money","property"]
TARGET_DATES={"2016-07-01","2016-08-02","2016-09-01"}
VIRTUAL_RULES=[
 {"id":"SH_RAILWAY_STATION_L1_L34_VIRTUAL","station":"上海火车站","pairs":[("01","03"),("01","04")],"from":"2008-06-01","to":None,"max_gap_s":1800},
 {"id":"HONGQIAO_T2_L2_L10_VIRTUAL","station":"虹桥2号航站楼","pairs":[("02","10")],"from":"2010-11-30","to":"2017-12-29","max_gap_s":1800},
 {"id":"NANJING_WEST_L2_L12_L13_VIRTUAL","station":"南京西路","pairs":[("02","12"),("02","13"),("12","13")],"from":"2015-12-19","to":None,"max_gap_s":1800},
 {"id":"LONGHUA_L11_L12_VIRTUAL","station":"龙华","pairs":[("11","12")],"from":"2015-12-19","to":"2018-12-29","max_gap_s":1800},
]

def open_text(path:Path):
    raw=gzip.open if path.suffix.lower()==".gz" else open
    last=None
    for enc in ("utf-8-sig","gb18030","gbk"):
        try:
            f=raw(path,"rt",encoding=enc,newline="");f.read(8192);f.seek(0);return f
        except UnicodeDecodeError as e:
            last=e
            try:f.close()
            except Exception:pass
    raise last

def iter_rows(path:Path):
    with open_text(path) as f:
        rd=csv.reader(f); first=next(rd,None)
        if first is None:return
        headers={"卡ID","卡号","刷卡日期","交易日期","刷卡时间","刷卡站点","刷卡乘车类型","card_id"}
        if not headers.intersection(x.strip() for x in first):yield dict(zip(FIELDS,first))
        for row in rd:
            if row:yield dict(zip(FIELDS,row))

def parse_station(raw:str):
    token=(raw or "").replace("\ufeff","").strip()
    m=re.match(r"^(\d+)号线(.+)$",token)
    if not m:return None
    line=m.group(1).zfill(2); station=m.group(2).strip()
    return line,station,token

def group_token(card:str):
    return hashlib.sha256(("group:"+card).encode("utf-8")).hexdigest()

def split_name(token:str):
    x=int(hashlib.sha256(("split:"+token).encode()).hexdigest()[:8],16)%10
    return "validation" if x==0 else ("test" if x in (1,2) else "train")

def date_ok(d,rule):
    return d>=rule["from"] and (rule["to"] is None or d<=rule["to"])

def virtual_rule(exit_line,exit_station,entry_line,entry_station,gap,date_text):
    if exit_station!=entry_station or gap<=0:return None
    pair=frozenset((exit_line,entry_line))
    for rule in VIRTUAL_RULES:
        if rule["station"]==exit_station and gap<=rule["max_gap_s"] and date_ok(date_text,rule):
            if any(pair==frozenset(x) for x in rule["pairs"]):return rule
    return None

def ingest(raw:Path,db:Path,out:Path,audit_path:Path,max_gate_duration_s=21600):
    con=sqlite3.connect(db);cur=con.cursor()
    cur.executescript(
      "PRAGMA journal_mode=WAL;PRAGMA synchronous=NORMAL;"
      "DROP TABLE IF EXISTS events;DROP TABLE IF EXISTS blocked;"
      "CREATE TABLE events(card TEXT,ts INTEGER,datestr TEXT,line TEXT,station TEXT,raw_token TEXT,money REAL,raw_row INTEGER);"
      "CREATE TABLE blocked(card TEXT PRIMARY KEY);")
    audit=Counter();rejects=Counter();batch=[];blocked=[];epoch=datetime(1970,1,1);seen_dates=set()
    for i,r in enumerate(iter_rows(raw),start=1):
        audit["input_rows"]+=1
        if i % 1000000 == 0:
            print(json.dumps({"phase":"scan","rows":i,"metro_events":audit["metro_events"]}),flush=True)
        if (r.get("vehicle") or "").strip()!="地铁":
            audit["non_metro_rows"]+=1;continue
        raw_card=(r.get("card_id") or "").strip()
        if not raw_card:
            rejects["missing_card_id"]+=1;continue
        card=group_token(raw_card)
        try:
            datestr=r["date"].strip()
            if datestr not in TARGET_DATES:raise ValueError("unexpected date")
            dt=datetime.strptime(datestr+" "+r["time"].strip(),"%Y-%m-%d %H:%M:%S")
        except Exception:
            rejects["datetime_or_date"]+=1;blocked.append((card,));continue
        seen_dates.add(datestr);ts=int((dt-epoch).total_seconds())
        p=parse_station(r.get("station",""))
        try:money=float(r["money"])
        except Exception:money=None
        if p is None:
            rejects["station_parse"]+=1
            batch.append((card,ts,datestr,"","","",None,i));audit["station_barrier_events"]+=1;audit["metro_events"]+=1
        elif money is None:
            rejects["money_parse"]+=1
            batch.append((card,ts,datestr,p[0],p[1],p[2],None,i));audit["money_barrier_events"]+=1;audit["metro_events"]+=1
        elif money<0:
            rejects["negative_money"]+=1
            batch.append((card,ts,datestr,p[0],p[1],p[2],None,i));audit["negative_money_barrier_events"]+=1;audit["metro_events"]+=1
        else:
            batch.append((card,ts,datestr,p[0],p[1],p[2],money,i));audit["metro_events"]+=1
        if len(batch)>=100000:
            cur.executemany("INSERT INTO events VALUES(?,?,?,?,?,?,?,?)",batch);batch.clear();con.commit()
        if len(blocked)>=5000:
            cur.executemany("INSERT OR IGNORE INTO blocked VALUES(?)",blocked);blocked.clear();con.commit()
    if batch:cur.executemany("INSERT INTO events VALUES(?,?,?,?,?,?,?,?)",batch)
    if blocked:cur.executemany("INSERT OR IGNORE INTO blocked VALUES(?)",blocked)
    con.commit();print(json.dumps({"phase":"index","events":audit["metro_events"]}),flush=True);cur.execute("CREATE INDEX idx_events_card_ts ON events(card,ts,raw_row)");con.commit()
    blocked_cards={x[0] for x in cur.execute("SELECT card FROM blocked")}
    q=cur.execute("SELECT card,ts,datestr,line,station,raw_token,money,raw_row FROM events ORDER BY card,ts,raw_row")
    def groups():
        current=None;g=[]
        for row in q:
            if current is not None and row[0]!=current:
                yield current,g;g=[]
            current=row[0];g.append(row)
        if current is not None:yield current,g

    opener=gzip.open if out.suffix.lower()==".gz" else open
    with opener(out,"wt",encoding="utf-8",newline="") as f:
        fields=["split","entry_time","exit_time","origin_token","destination_token","origin_line","origin_station",
                "destination_line","destination_station","duration_s","virtual_transfer_count","virtual_transfer_stations","virtual_transfer_policy_ids"]
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for card,events in groups():
            audit["unique_hashed_cards"]+=1
            if audit["unique_hashed_cards"] % 500000 == 0:
                print(json.dumps({"phase":"pair","cards":audit["unique_hashed_cards"],"gate_segments":audit["gate_segments"]}),flush=True)
            if card in blocked_cards:
                audit["cards_blocked_by_unordered_reject"]+=1;audit["events_on_blocked_cards"]+=len(events);continue
            ts_counts=Counter(e[1] for e in events);segments=[];open_entry=None
            for e in events:
                _card,ts,datestr,line,station,raw_token,money,raw_row=e
                if ts_counts[ts]>1:
                    audit["duplicate_timestamp_events"]+=1;open_entry=None;continue
                if money is None:
                    audit["barrier_events"]+=1;open_entry=None;continue
                if money==0:
                    if open_entry is not None:audit["entry_before_exit"]+=1
                    open_entry=e
                elif money>0:
                    if open_entry is None:
                        audit["exit_without_entry"]+=1;continue
                    if ts<=open_entry[1]:
                        audit["nonpositive_duration"]+=1;open_entry=None;continue
                    if ts-open_entry[1]>max_gate_duration_s:
                        audit["gate_segment_over_max_duration"]+=1;open_entry=None;continue
                    segments.append([open_entry,e]);open_entry=None
            if open_entry is not None:audit["open_entry_at_end"]+=1
            audit["gate_segments"]+=len(segments)
            i=0
            while i<len(segments):
                a,b=segments[i];transfers=[];policies=[];j=i+1
                while j<len(segments):
                    na,nb=segments[j];gap=na[1]-b[1]
                    rule=virtual_rule(b[3],b[4],na[3],na[4],gap,na[2])
                    if rule is None:break
                    transfers.append(rule["station"]);policies.append(rule["id"])
                    audit["virtual_transfer_stitches"]+=1;audit["vt_"+rule["id"]]+=1
                    b=nb;j+=1
                rec={"split":split_name(card),
                    "entry_time":(epoch+timedelta(seconds=a[1])).isoformat(sep=" "),
                    "exit_time":(epoch+timedelta(seconds=b[1])).isoformat(sep=" "),
                    "origin_token":a[5],"destination_token":b[5],"origin_line":a[3],"origin_station":a[4],
                    "destination_line":b[3],"destination_station":b[4],"duration_s":b[1]-a[1],
                    "virtual_transfer_count":len(transfers),"virtual_transfer_stations":"|".join(transfers),
                    "virtual_transfer_policy_ids":"|".join(policies)}
                w.writerow(rec);audit["journeys"]+=1;audit["split_"+rec["split"]]+=1;i=j
    payload={k:v for k,v in sorted(audit.items())}
    payload.update({"rejects":dict(sorted(rejects.items())),"input_dates":sorted(seen_dates),
      "source_file":raw.name,"max_gate_duration_s":max_gate_duration_s,
      "identity_mode":"SOURCE_TOKEN_LINE_SCOPED","card_grouping":"TRANSIENT_SHA256_GROUP_KEY_NOT_OUTPUT",
      "active_virtual_policy_ids":[x["id"] for x in VIRTUAL_RULES],
      "privacy":"Raw card IDs and transient grouping hashes are not written to the journey output; only stable split labels are persisted.",
      "status":"INGEST_EXECUTED"})
    audit_path.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding="utf-8")
    con.close();return payload

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--raw",type=Path,required=True);ap.add_argument("--db",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True);ap.add_argument("--audit-out",type=Path,required=True)
    ap.add_argument("--max-gate-duration-s",type=int,default=21600)
    a=ap.parse_args()
    payload=ingest(a.raw,a.db,a.out,a.audit_out,a.max_gate_duration_s)
    print(json.dumps(payload,ensure_ascii=False))

if __name__=="__main__":main()
