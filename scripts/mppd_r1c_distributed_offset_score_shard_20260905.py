import argparse
import gzip
import json
import math
from collections import defaultdict
from pathlib import Path

import scripts.mppd_r1b_first_bidirectional_joint_update_20260904 as base
import scripts.mppd_r1c_hierarchical_temporal_kernels_first_pass_20260904 as r1c


def load_kernels(path):
    payload=json.loads(Path(path).read_text(encoding='utf-8')); return payload.get('kernels',payload)


def choose_kernel(row,kernels):
    if row['type']=='EGRESS':
        hour=row.get('time_bin_hour'); ctx=row.get('station_context')
        return r1c.choose_context_kernel(kernels,'EGRESS',ctx,int(hour)) if hour is not None else kernels['egress']
    if row.get('kind')=='ACCESS':
        hour=row.get('time_bin_hour'); ctx=row.get('station_context')
        return r1c.choose_context_kernel(kernels,'ACCESS',ctx,int(hour)) if hour is not None else kernels['access']
    return base.kernel_for(row.get('kind'),row.get('movement'),kernels)


def row_loglik(row,delta,kernels):
    kern=choose_kernel(row,kernels)
    if row['type']=='EGRESS':
        mean=float(row['x_sec'])+float(row.get('x_shift_coeff',0))*delta
        return math.log(max(base.EPS,base.expected_kernel_pdf_normal_difference(mean,float(row.get('arr_sd',0.0)),kern)))
    upper=float(row['upper_rel_sec'])+float(row.get('upper_shift_coeff',0))*delta
    usd=math.sqrt(float(row.get('upper_sd',0.0))**2+float(row.get('ready_sd',0.0))**2)
    fu=base.expected_kernel_cdf_normal_difference(upper,usd,kern)
    if row.get('lower_rel_sec') is None:
        fl=0.0
    else:
        lower=float(row['lower_rel_sec'])+float(row.get('lower_shift_coeff',0))*delta
        lsd=math.sqrt(float(row.get('lower_sd',0.0))**2+float(row.get('ready_sd',0.0))**2)
        fl=base.expected_kernel_cdf_normal_difference(lower,lsd,kern)
    return math.log(max(base.EPS,fu-fl))


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--offset-stats',required=True); ap.add_argument('--kernels',required=True); ap.add_argument('--grid',default='-90,-60,-30,0,30,60,90'); ap.add_argument('--out',required=True); args=ap.parse_args()
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True); kernels=load_kernels(args.kernels); grid=sorted(set(int(x) for x in args.grid.split(',') if x.strip()))
    if 0 not in grid: grid.append(0); grid.sort()
    scores=defaultdict(lambda:{d:{'ll_sum':0.0,'used_weight':0.0} for d in grid}); row_count=0; weight_total=0.0
    with gzip.open(args.offset_stats,'rt',encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            r=json.loads(line); root=r['root']; w=float(r['weight']); row_count+=1; weight_total+=w
            if w<=0: continue
            for d in grid:
                lp=row_loglik(r,d,kernels)
                if math.isfinite(lp): scores[root][d]['ll_sum']+=w*lp; scores[root][d]['used_weight']+=w
    payload={'schema':'mppd.r1c-distributed-offset-score-shard.v1','date':'2026-09-05','status':'R1C_DISTRIBUTED_CONTEXT_AWARE_OFFSET_SCORE_SHARD_COMPLETED','grid_sec':grid,'source_offset_stat_row_count':row_count,'source_offset_stat_weight_total':weight_total,'root_count':len(scores),'scores':{root:{str(d):v for d,v in ds.items()} for root,ds in scores.items()},'scientific_boundary':['Score sums are additive shard contributions to the same context-aware R1C service-offset likelihood used in the one-time consistency patch.','No offset is selected and no service timing or support is mutated at shard level.'],'no_email_notification_logic':True}
    (out/'offset_score_sums.json').write_text(json.dumps(payload,ensure_ascii=False),encoding='utf-8'); print(json.dumps({'root_count':len(scores),'row_count':row_count,'weight_total':weight_total},ensure_ascii=False,indent=2))


if __name__=='__main__': main()
