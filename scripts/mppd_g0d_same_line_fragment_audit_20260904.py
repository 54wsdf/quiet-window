import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import networkx as nx

import scripts.mppd_g0_full_network_coverage_20260904 as g0
from scripts.mppd_topology_patch_20260904 import apply_topology_patch, load_patch


def num(v):
    try:
        return int(str(v).strip())
    except Exception:
        return None


def read_remaining(path):
    rows=[]
    with open(path,encoding='utf-8',newline='') as f:
        for r in csv.DictReader(f):
            rows.append({'origin_code':str(r.get('origin_code') or '').strip(),'destination_code':str(r.get('destination_code') or '').strip(),'passenger_mass':int(r.get('passenger_mass') or 0)})
    return rows


def comp_map(G):
    comps=list(nx.connected_components(G));out={}
    for i,c in enumerate(comps):
        for n in c:out[n]=i
    return comps,out


def all_entry_index(p1c):
    by_line_seq=defaultdict(list);by_node=defaultdict(list)
    for x in p1c.get('canonical_entries',[]):
        line=str(x.get('service_subway_id') or '').strip();st=str(x.get('service_statn_id') or '').strip();seq=num(x.get('service_statn_sn'))
        if not line or not st:continue
        obj={'node':g0.node(line,st),'line':line,'station':st,'seq':seq,'tier':str(x.get('tier') or ''),'dv_name':str(x.get('dv_name') or '').strip(),'service_name':str(x.get('service_name') or '').strip(),'out_stn_num':str(x.get('out_stn_num') or '').strip()}
        by_node[obj['node']].append(obj)
        if seq is not None:by_line_seq[(line,seq)].append(obj)
    return by_node,by_line_seq


def best_name(entries,meta_row):
    names=[]
    for x in entries:
        for k in ('dv_name','service_name'):
            v=str(x.get(k) or '').strip()
            if v and v not in names:names.append(v)
    for k in ('dv_name','service_name'):
        v=str((meta_row or {}).get(k) or '').strip()
        if v and v not in names:names.append(v)
    return '|'.join(names[:4])


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--p1c',required=True);ap.add_argument('--remaining',required=True);ap.add_argument('--patch',required=True);ap.add_argument('--out',required=True);args=ap.parse_args()
    outdir=Path(args.out);outdir.mkdir(parents=True,exist_ok=True)
    p1c=json.loads(Path(args.p1c).read_text(encoding='utf-8'))
    G,meta,code_to_nodes,transfer_groups,ambiguous_seq,ambiguous_codes,graph_build=g0.build_network(args.p1c)
    patch=load_patch(args.patch);patch_result=apply_topology_patch(G,meta,patch)
    comps,component_of=comp_map(G);rows=read_remaining(args.remaining);by_node_all,by_line_seq_all=all_entry_index(p1c)

    pair_mass=Counter();pair_od=Counter()
    for r in rows:
        os={component_of[n] for n in code_to_nodes.get(r['origin_code'],[]) if n in component_of};ds={component_of[n] for n in code_to_nodes.get(r['destination_code'],[]) if n in component_of}
        for a in os:
            for b in ds:
                if a==b:continue
                cp=tuple(sorted((a,b)));pair_mass[cp]+=r['passenger_mass'];pair_od[cp]+=1

    comp_lines={i:defaultdict(list) for i in range(len(comps))}
    for i,c in enumerate(comps):
        for n in c:
            comp_lines[i][meta[n]['line']].append(n)

    candidates=[]
    for (a,b),mass in pair_mass.most_common():
        shared=sorted(set(comp_lines[a]) & set(comp_lines[b]))
        if not shared:continue
        for line in shared:
            pairs=[]
            for u in comp_lines[a][line]:
                su=meta[u].get('seq')
                if su is None:continue
                for v in comp_lines[b][line]:
                    sv=meta[v].get('seq')
                    if sv is None:continue
                    pairs.append((abs(su-sv),min(su,sv),max(su,sv),u,v,su,sv))
            pairs.sort()
            for rank,item in enumerate(pairs[:5],start=1):
                gap,lo,hi,u,v,su,sv=item
                intermediate=[];intermediate_outside_graph=[]
                if gap>1:
                    for seq in range(lo+1,hi):
                        for x in by_line_seq_all.get((line,seq),[]):
                            intermediate.append(f"{seq}:{x['node']}:{x['tier']}:{x['dv_name'] or x['service_name']}")
                            if x['node'] not in G:intermediate_outside_graph.append(f"{seq}:{x['node']}:{x['tier']}:{x['dv_name'] or x['service_name']}")
                candidates.append({
                    'component_a':a,'component_b':b,'residual_passenger_mass':mass,'residual_od_count':pair_od[(a,b)],'line':line,'candidate_rank':rank,
                    'node_u':u,'node_v':v,'seq_u':su,'seq_v':sv,'absolute_seq_gap':gap,
                    'name_u':best_name(by_node_all.get(u,[]),meta.get(u)),'name_v':best_name(by_node_all.get(v,[]),meta.get(v)),
                    'intermediate_entry_count':len(intermediate),'intermediate_entries':'|'.join(intermediate[:30]),
                    'intermediate_outside_graph_count':len(intermediate_outside_graph),'intermediate_outside_graph':'|'.join(intermediate_outside_graph[:30]),
                    'diagnostic_class':('ADJACENT_SEQ_COMPONENT_BREAK' if gap==1 else ('LOWER_TIER_OR_MISSING_NODE_GAP' if intermediate_outside_graph else 'SEQUENCE_GAP_OR_BRANCH_JUNCTION'))
                })

    comp_rows=[]
    incident_mass=Counter()
    for (a,b),mass in pair_mass.items():incident_mass[a]+=mass;incident_mass[b]+=mass
    for i,c in enumerate(comps):
        lines=sorted(comp_lines[i]);seq_ranges=[]
        for line,nodes in comp_lines[i].items():
            ss=sorted(meta[n]['seq'] for n in nodes if meta[n].get('seq') is not None)
            seq_ranges.append(f"{line}:{ss[0] if ss else 'NA'}-{ss[-1] if ss else 'NA'}")
        comp_rows.append({'component':i,'node_count':len(c),'lines':'|'.join(lines),'seq_ranges':'|'.join(seq_ranges),'remaining_residual_incident_mass':incident_mass[i],'nodes':'|'.join(sorted(c))})
    comp_rows.sort(key=lambda x:(-x['remaining_residual_incident_mass'],-x['node_count'],x['component']))

    diag_mass=Counter()
    seen_cp=set()
    for c in sorted(candidates,key=lambda x:(-x['residual_passenger_mass'],x['candidate_rank'],x['absolute_seq_gap'])):
        cp=(c['component_a'],c['component_b'])
        if cp in seen_cp:continue
        seen_cp.add(cp);diag_mass[c['diagnostic_class']]+=c['residual_passenger_mass']

    result={
      'schema':'mppd.g0d-same-line-fragment-audit.v1','date':'2026-09-04','status':'G0D_POST_G0C_SAME_LINE_FRAGMENT_AUDIT_COMPLETED','authority':'00_CURRENT_CORE_CLOSURE_WORKPLAN_V6_FULL_NETWORK_STATE_RECONSTRUCTION_20260904.md',
      'scope':{'remaining_unrouted_od_count':len(rows),'remaining_unrouted_passenger_mass':sum(r['passenger_mass'] for r in rows),'patched_graph_components':len(comps),'patched_graph_nodes':G.number_of_nodes(),'patched_graph_edges':G.number_of_edges()},
      'same_line_fragment_candidate_count':len(candidates),'component_pairs_with_same_line_candidate':len({(x['component_a'],x['component_b']) for x in candidates}),
      'best_candidate_diagnostic_mass':dict(diag_mass),
      'top_candidates':[x for x in sorted(candidates,key=lambda x:(-x['residual_passenger_mass'],x['candidate_rank'],x['absolute_seq_gap'])) if x['candidate_rank']==1][:30],
      'scientific_boundary':['G0D is a diagnostic audit only; it does not insert same-line edges.','A shared line identifier across disconnected components is not by itself sufficient evidence of physical adjacency because branch/junction topology may exist.','Sequence gaps, lower-tier omitted entries and external network topology must be reconciled before any second topology patch is qualified.','Remaining AFC mass is retained as structural evidence and is not reclassified as anomalous passenger behavior.'],
      'next_gate':'Externally and internally qualify the high-mass same-line fragment boundaries; create a second explicit patch only for supported adjacency/junction relations, then reroute the remaining AFC residual.','no_email_notification_logic':True
    }
    fields=list(candidates[0]) if candidates else ['component_a','component_b']
    with open(outdir/'g0d_same_line_fragment_candidates.csv','w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(candidates)
    with open(outdir/'g0d_post_g0c_component_summary.csv','w',encoding='utf-8',newline='') as f:
        fields2=list(comp_rows[0]) if comp_rows else ['component'];w=csv.DictWriter(f,fieldnames=fields2);w.writeheader();w.writerows(comp_rows)
    with open(outdir/'g0d_remaining_component_pair_mass.csv','w',encoding='utf-8',newline='') as f:
        rr=[{'component_a':a,'component_b':b,'passenger_mass':m,'od_count':pair_od[(a,b)],'shared_lines':'|'.join(sorted(set(comp_lines[a])&set(comp_lines[b])))} for (a,b),m in pair_mass.most_common()]
        fields3=list(rr[0]) if rr else ['component_a','component_b'];w=csv.DictWriter(f,fieldnames=fields3);w.writeheader();w.writerows(rr)
    (outdir/'g0d_same_line_fragment_audit_summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
