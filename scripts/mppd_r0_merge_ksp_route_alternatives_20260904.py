import argparse,csv,gzip,json
from collections import Counter
from pathlib import Path

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dir',required=True);ap.add_argument('--out',required=True);ap.add_argument('--shard-count',type=int,required=True);args=ap.parse_args()
    d=Path(args.dir);out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    cand_hist=Counter();tc_mass=Counter();status_mass=Counter();od_count=mass=trunc_od=trunc_mass=unrouted=0
    merged=out/'r0_ksp_full_network_route_candidates.jsonl.gz'
    with gzip.open(merged,'wt',encoding='utf-8') as fo:
        for i in range(args.shard_count):
            p=d/f'ksp_routes_shard_{i:02d}_of_{args.shard_count:02d}.jsonl.gz'
            if not p.exists(): raise FileNotFoundError(p)
            with gzip.open(p,'rt',encoding='utf-8') as fi:
                for line in fi:
                    obj=json.loads(line);fo.write(line)
                    m=int(obj['passenger_mass']);mass+=m;od_count+=1
                    cs=obj.get('candidates',[]);cand_hist[len(cs)]+=m
                    if not cs:unrouted+=1
                    if obj.get('candidate_set_status')=='TRUNCATED_CANDIDATE_SET':trunc_od+=1;trunc_mass+=m
                    status_mass[obj.get('candidate_set_status','UNKNOWN')]+=m
                    for c in cs:tc_mass[int(c.get('transfer_count',0))]+=m
    result={'schema':'mppd.r0-full-network-ksp-route-alternatives.v1','date':'2026-09-04','status':'R0_FULL_NETWORK_KSP_ROUTE_ALTERNATIVES_MERGED',
      'authority':'00_CURRENT_CORE_CLOSURE_WORKPLAN_V6_FULL_NETWORK_STATE_RECONSTRUCTION_20260904.md',
      'scope_assertions':{'full_network':True,'line_filter':False,'segment_filter':False,'transfer_count_cap':False,'all_shards_merged':True},
      'od':{'count':od_count,'passenger_mass':mass,'unrouted_od':unrouted,'truncated_od':trunc_od,'truncated_passenger_mass':trunc_mass,'truncated_mass_share':trunc_mass/mass if mass else 0},
      'passenger_mass_by_candidate_count':dict(sorted(cand_hist.items())),'candidate_support_mass_by_transfer_count':dict(sorted(tc_mass.items())),'candidate_set_status_mass':dict(status_mass),
      'scientific_boundary':['Merged KSP candidates are structural route hypotheses, not observed passenger routes.','No transfer-count cap is applied.','ODs labeled TRUNCATED_CANDIDATE_SET require expansion or explicit posterior-tail accounting before final route inference.'],
      'next_gate':'Compare candidate diversity against R0 four-policy generator; if truncation is controlled, use KSP cache for G2 route posterior.','no_email_notification_logic':True}
    (out/'r0_ksp_full_network_route_alternatives_summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
