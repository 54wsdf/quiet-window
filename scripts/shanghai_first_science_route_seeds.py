#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,gzip,json,re
from collections import Counter,defaultdict
from pathlib import Path
import duckdb
from shanghai_route_frontier import build_adjacency,solve_frontier,target_frontier

ALIASES={"淞浜路":"淞滨路","李子园路":"李子园","上海野生动物园":"野生动物园","外高桥保税区北":"外高桥保税区北站","外高桥保税区南":"外高桥保税区南站","上海大学站":"上海大学"}
HUAQIAO={"花桥","光明路","兆丰路","安亭","上海汽车城","昌吉东路","上海赛车场"}
JIADINGBEI={"嘉定北","嘉定西","白银路"}
L2_INNER={"徐泾东","虹桥火车站","虹桥2号航站楼","淞虹路","北新泾","威宁路","娄山关路","中山公园","江苏路","静安寺","南京西路","人民广场","南京东路","陆家嘴","东昌路","世纪大道","上海科技馆","世纪公园","龙阳路","张江高科","金科路","广兰路"}
L2_EAST={"唐镇","创新中路","华夏东路","川沙","凌空路","远东大道","海天三路","浦东国际机场"}

def node(line,name): return f"{int(line):02d}::{name.strip()}"
def physical(key):
    line,name=key.split("::",1)
    return key if name=="浦电路" else name

def parse_token(token):
    token=(token or "").replace("\ufeff","").strip()
    m=re.fullmatch(r"(\d+)号线(.*)",token)
    if not m: raise ValueError(token)
    line=m.group(1).zfill(2); name=ALIASES.get(m.group(2).strip(),m.group(2).strip())
    return line,name,(f"{line}::{name}" if name=="浦电路" else name)

def build_worlds(station_path,connections_path):
    with station_path.open(encoding="utf-8-sig",newline="") as f: stations=list(csv.DictReader(f))
    connections=json.loads(connections_path.read_text(encoding="utf-8"))
    all_nodes={node(r["LINE_NO"],r["ST_NAME"]) for r in stations}
    rides={(min(node(l,a),node(l,b)),max(node(l,a),node(l,b)),"ride") for a,b,l in connections}
    active={n for e in rides for n in e[:2]}
    assert all_nodes-active=={"11::严御路"}
    extensions={11:["罗山路","秀沿路","康新公路","迪士尼"],12:["七莘路","虹莘路","顾戴路","东兰路","虹梅路","虹漕路","桂林公园","漕宝路","龙漕路","龙华","龙华中路","大木桥路","嘉善路","陕西南路","南京西路","汉中路","曲阜路"],13:["长寿路","江宁路","汉中路","自然博物馆","南京西路","淮海中路","新天地","马当路","世博会博物馆","世博大道"]}
    worlds={}
    for epoch in ["2015_snapshot","2016_expanded"]:
        nodes=set(active); edges=set(rides)
        if epoch=="2016_expanded":
            for line,seq in extensions.items():
                nodes.update(node(line,s) for s in seq)
                edges.update((min(node(line,a),node(line,b)),max(node(line,a),node(line,b)),"ride") for a,b in zip(seq,seq[1:]))
        groups=defaultdict(list)
        for n in nodes: groups[physical(n)].append(n)
        virtual=set()
        def vpair(st,a,b): virtual.add(tuple(sorted((node(a,st),node(b,st)))))
        for a,b in [(1,3),(1,4)]: vpair("上海火车站",a,b)
        vpair("虹桥2号航站楼",2,10)
        if epoch=="2015_snapshot": vpair("陕西南路",1,10)
        else:
            for a,b in [(2,12),(2,13),(12,13)]: vpair("南京西路",a,b)
            vpair("龙华",11,12)
        for include in [False,True]:
            e=set(edges)
            for states in groups.values():
                ss=sorted(states)
                for i in range(len(ss)):
                    for j in range(i+1,len(ss)):
                        pair=(ss[i],ss[j])
                        if include or pair not in virtual: e.add((pair[0],pair[1],"transfer"))
            ident=epoch+("_all_physical" if include else "_paid_only")
            worlds[ident]={"nodes":sorted(nodes),"edges":sorted(e),"endpoint_groups":{p:sorted(v) for p,v in sorted(groups.items())}}
    return worlds

def l11_period(t):
    if "07:00:00"<=t<"08:30:00": return "AM_0700_0830"
    if "09:00:00"<=t<"16:00:00": return "OFF_0900_1600"
    if "17:00:00"<=t<"19:30:00": return "PM_1700_1930"
    return "OTHER"

def l2_period(t):
    if "07:00:00"<=t<"09:00:00": return "AM_0700_0900"
    if "09:00:00"<=t<"16:00:00": return "OFF_0900_1600"
    if "17:30:00"<=t<"19:00:00": return "PM_DOC_1730_1900"
    if ("17:00:00"<=t<"17:30:00") or ("19:00:00"<=t<"19:30:00"): return "PM_SHOULDER"
    return "OTHER"

def classify(o,d,trunk):
    if o in HUAQIAO and d in trunk: return "L11_HUAQIAO_TO_TRUNK"
    if o in JIADINGBEI and d in trunk: return "L11_JIADINGBEI_TO_TRUNK"
    if o in trunk and d in HUAQIAO: return "L11_TRUNK_TO_HUAQIAO"
    if o in trunk and d in JIADINGBEI: return "L11_TRUNK_TO_JIADINGBEI"
    if (o in HUAQIAO and d in JIADINGBEI) or (o in JIADINGBEI and d in HUAQIAO): return "L11_CROSS_BRANCH"
    if o in L2_INNER and d in L2_EAST: return "L2_INNER_TO_EAST"
    if o in L2_EAST and d in L2_INNER: return "L2_EAST_TO_INNER"
    return None

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--journey-dir",type=Path,required=True); ap.add_argument("--station",type=Path,required=True)
    ap.add_argument("--connections",type=Path,required=True); ap.add_argument("--out-dir",type=Path,required=True)
    a=ap.parse_args(); a.out_dir.mkdir(parents=True,exist_ok=True)
    worlds=build_worlds(a.station,a.connections)
    l11_names={physical(n) for n in worlds["2015_snapshot_all_physical"]["nodes"] if n.startswith("11::")}
    trunk=l11_names-HUAQIAO-JIADINGBEI

    con=duckdb.connect(); con.execute("PRAGMA threads=4"); con.execute("PRAGMA memory_limit='6GB'")
    parts=[]
    for day in ["20160701","20160802","20160901"]:
        p=a.journey_dir/f"SPTCC-{day}.formal_mppd_journeys.parquet"
        if not p.is_file():
            raise FileNotFoundError(f"qualified Parquet mirror missing: {p}")
        ps=str(p).replace("'","''")
        parts.append("""
          SELECT '"""+day+"""' AS day,
                 origin_line,origin_station,destination_line,destination_station,
                 CASE
                   WHEN origin_line='11' AND destination_line='11' THEN
                     CASE
                       WHEN substr(entry_time,12,8)>='07:00:00' AND substr(entry_time,12,8)<'08:30:00' THEN 'AM_0700_0830'
                       WHEN substr(entry_time,12,8)>='09:00:00' AND substr(entry_time,12,8)<'16:00:00' THEN 'OFF_0900_1600'
                       WHEN substr(entry_time,12,8)>='17:00:00' AND substr(entry_time,12,8)<'19:30:00' THEN 'PM_1700_1930'
                       ELSE 'OTHER' END
                   WHEN origin_line='02' AND destination_line='02' THEN
                     CASE
                       WHEN substr(entry_time,12,8)>='07:00:00' AND substr(entry_time,12,8)<'09:00:00' THEN 'AM_0700_0900'
                       WHEN substr(entry_time,12,8)>='09:00:00' AND substr(entry_time,12,8)<'16:00:00' THEN 'OFF_0900_1600'
                       WHEN substr(entry_time,12,8)>='17:30:00' AND substr(entry_time,12,8)<'19:00:00' THEN 'PM_DOC_1730_1900'
                       WHEN (substr(entry_time,12,8)>='17:00:00' AND substr(entry_time,12,8)<'17:30:00')
                         OR (substr(entry_time,12,8)>='19:00:00' AND substr(entry_time,12,8)<'19:30:00') THEN 'PM_SHOULDER'
                       ELSE 'OTHER' END
                   ELSE NULL END AS period,
                 count(*)::BIGINT AS n
          FROM read_parquet('"""+ps+"""')
          WHERE (origin_line='11' AND destination_line='11')
             OR (origin_line='02' AND destination_line='02')
          GROUP BY 1,2,3,4,5,6
        """)
    rows=con.execute(" UNION ALL ".join(parts)).fetchall(); con.close()

    support=Counter(); od_mass=Counter()
    for day,ol,o,dl,d,period,n in rows:
        fam=classify(o,d,trunk)
        if not fam: continue
        support[(fam,day,period,o,d)]+=int(n); od_mass[(fam,o,d)]+=int(n)

    with (a.out_dir/"first_science_od_support.csv").open("w",encoding="utf-8",newline="") as f:
        w=csv.writer(f); w.writerow(["family","day","period","origin","destination","journeys"])
        for key,n in sorted(support.items()): w.writerow([*key,n])

    ods=sorted(od_mass)
    stats=Counter(); fstats=defaultdict(Counter)
    with gzip.open(a.out_dir/"first_science_structural_route_seeds.jsonl.gz","wt",encoding="utf-8") as out:
        for world_name,world in worlds.items():
            adj=build_adjacency(world["nodes"],world["edges"]); groups=world["endpoint_groups"]
            edge_cost={}
            for x,y,kind in world["edges"]:
                edge_cost[frozenset((x,y))]=(1,0) if kind=="ride" else (0,1)
            by_origin=defaultdict(list)
            for fam,o,d in ods:
                if o in groups and d in groups: by_origin[o].append((fam,d))
                else: fstats[fam]["missing_endpoint"]+=1
            total_origins=len(by_origin)
            for oi,(o,targets) in enumerate(sorted(by_origin.items()),start=1):
                labels=solve_frontier(adj,groups[o])
                for fam,d in sorted(targets):
                    front=target_frontier(labels,groups[d]); vectors=[]
                    for (rides,changes),rec in front.items():
                        path=list(rec["path"])
                        rr=tt=0
                        for x,y in zip(path,path[1:]):
                            dr,dt=edge_cost[frozenset((x,y))]; rr+=dr; tt+=dt
                        if (rr,tt)!=(rides,changes):
                            raise AssertionError((world_name,o,d,(rides,changes),(rr,tt),path))
                        vectors.append({"ride_intervals":rides,"line_changes":changes,"path_count":rec["count"],"representative_path":path})
                        stats["cost_vectors"]+=1; stats["path_instances_counted_not_expanded"]+=rec["count"]
                        stats["max_line_changes"]=max(stats["max_line_changes"],changes); stats["max_path_ties"]=max(stats["max_path_ties"],rec["count"])
                    out.write(json.dumps({"world":world_name,"family":fam,"origin":o,"destination":d,"formal_journey_mass_all_days":od_mass[(fam,o,d)],"vectors":vectors},ensure_ascii=False,separators=(",",":"))+"\n")
                    stats["od_world_records"]+=1; fstats[fam]["od_world_records"]+=1
                print(json.dumps({"phase":"frontier","world":world_name,"origin_index":oi,"origin_total":total_origins,"origin":o,"od_records":stats["od_world_records"]},ensure_ascii=False),flush=True)

    families={}
    for fam in sorted({x[0] for x in ods}):
        families[fam]={"unique_od_pairs":len({(o,d) for f,o,d in ods if f==fam}),"formal_journey_mass_all_days":sum(n for (f,o,d),n in od_mass.items() if f==fam),"od_world_records":fstats[fam]["od_world_records"],"missing_endpoint":fstats[fam]["missing_endpoint"]}
    q={"schema":"mppd.shanghai.first-science-structural-route-seeds.v2","families":families,"candidate_od_count":len(ods),"worlds":sorted(worlds),"route_stats":dict(stats),"boundaries":["Structural Pareto seeds only; not the final service-aware universe.","All tied-path multiplicities are retained exactly as counts, but tied geometries are intentionally not expanded in this bounded stage.","The representative path is for geometry/service diagnostics only and is not assumed to represent all tied paths.","Station-token line prefixes are not treated as route truth.","Structurally dominated routes are not assigned zero probability.","Only aggregate OD/period support is emitted."]}
    (a.out_dir/"qualification.json").write_text(json.dumps(q,ensure_ascii=False,indent=2),encoding="utf-8"); print(json.dumps(q,ensure_ascii=False,indent=2))
if __name__=="__main__": main()
