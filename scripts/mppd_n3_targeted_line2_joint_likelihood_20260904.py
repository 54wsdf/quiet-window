import math
from collections import defaultdict

import scripts.mppd_n3_shared_kernel_joint_likelihood_20260904 as m


def fit_transfer_targeted(rows, transfer_pairs, get_paths, A_par, E_par):
    eligible=[r for r in rows if r['ol']!=r['dl'] and transfer_pairs.get((r['ol'],r['dl']))]
    line2=[r for r in eligible if r['ol']==m.L2 or r['dl']==m.L2]
    other=[r for r in eligible if r['ol']!=m.L2 and r['dl']!=m.L2]
    line2=m.stable_sample(line2,16000,'k-l2')
    other=m.stable_sample(other,4000,'k-other')
    cross=line2+other
    K_global={'mu':math.log(180.0),'sigma':0.65,'n':0,'fitted':False}; K_by={}; history=[]
    for it in range(3):
        by=defaultdict(list); allints=[]; assigned=0; line2_assigned=0
        for r in cross:
            chains=m.candidate_transfer_chains(r,transfer_pairs,get_paths,A_par,E_par,K_global,K_by)
            if not chains:
                continue
            best=max(chains,key=lambda x:x[0]); assigned+=1
            if r['ol']==m.L2 or r['dl']==m.L2:
                line2_assigned+=1
            l,u=best[2]; by[best[1]].append((l,u,1.0)); allints.append((l,u,1.0))
        K_global=m.fit_lognorm_intervals(allints,(K_global['mu'],K_global['sigma']))
        new={}
        for k,ints in by.items():
            if len(ints)>=20:
                new[k]=m.fit_lognorm_intervals(ints,(K_global['mu'],K_global['sigma']))
        K_by=new
        history.append({'iter':it+1,'sample_total':len(cross),'sample_line2_adjacent':len(line2),'assigned':assigned,'line2_assigned':line2_assigned,'global':K_global.copy(),'specific_count':len(K_by),'top_specific':sorted([(k,v['n'],math.exp(v['mu']),v['sigma']) for k,v in K_by.items()],key=lambda x:x[1],reverse=True)[:30]})
    return K_global,K_by,history


m.fit_transfer=fit_transfer_targeted
m.main()
