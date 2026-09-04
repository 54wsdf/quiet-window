import argparse, csv, io, json, math, zipfile, hashlib
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy.stats import lognorm

L2 = "1002"
A_ST = "1002000237"
B_ST = "1002000238"
STRICT_TIERS = {"A_TRANSFER_CODE_LINE_NAME", "B_UNIQUE_NAME", "C_NAME_PREFIX_LINE"}
AT_TIER = "A_TRANSFER_CODE_LINE_NAME"
RAIL = {"201","202","203","204","205","206","208","209","210","231","232","235","236","237","290"}
START = datetime(2026,8,29,7)
END = datetime(2026,8,29,10)


def dt(v):
    d = ''.join(c for c in str(v or '') if c.isdigit())
    try:
        return datetime.strptime(d[:14], '%Y%m%d%H%M%S') if len(d) >= 14 else None
    except Exception:
        return None


def z4(v):
    d = ''.join(c for c in str(v or '') if c.isdigit())
    return d.zfill(4) if d and len(d) <= 4 else (d[-4:] if d else '')


def detect(raw):
    for e in ('utf-8-sig','utf-8','cp949','euc-kr'):
        try:
            raw.decode(e)
            return e
        except UnicodeDecodeError:
            pass
    return 'cp949'


def zr(z, name):
    f = z.open(name)
    h = f.read(262144)
    f.close()
    return io.TextIOWrapper(z.open(name), encoding=detect(h), errors='replace', newline='')


def sec(x):
    return float(x.total_seconds())


def stable_sample(rows, cap, salt):
    if len(rows) <= cap:
        return rows
    scored = []
    for i, r in enumerate(rows):
        key = f"{salt}|{r['ol']}|{r['os']}|{r['dl']}|{r['ds']}|{r['t0'].isoformat()}|{r['t1'].isoformat()}|{i}"
        h = int(hashlib.sha1(key.encode()).hexdigest()[:16], 16)
        scored.append((h, r))
    scored.sort(key=lambda x: x[0])
    return [r for _, r in scored[:cap]]


def rep_event(evs):
    if not evs:
        return None
    evs = sorted(evs)
    return evs[len(evs)//2]


def fit_lognorm_intervals(intervals, init=(math.log(240.0), 0.7)):
    clean = [(max(0.0,l), max(1e-3,u), max(1.0,w)) for l,u,w in intervals if u > max(0.0,l)+1e-6]
    if len(clean) < 20:
        return {"mu":float(init[0]),"sigma":float(init[1]),"n":len(clean),"fitted":False}
    def obj(x):
        mu, lsig = float(x[0]), float(x[1])
        sigma = math.exp(lsig)
        if sigma < 0.12 or sigma > 2.5:
            return 1e12
        val = 0.0
        for l,u,w in clean:
            Fu = lognorm.cdf(u, s=sigma, scale=math.exp(mu))
            Fl = lognorm.cdf(l, s=sigma, scale=math.exp(mu)) if l > 0 else 0.0
            val -= w*math.log(max(Fu-Fl, 1e-12))
        return val
    res = minimize(obj, np.array([init[0], math.log(init[1])]), method='Nelder-Mead', options={"maxiter":800,"xatol":1e-5,"fatol":1e-4})
    return {"mu":float(res.x[0]),"sigma":float(math.exp(res.x[1])),"n":len(clean),"fitted":bool(res.success)}


def fit_lognorm_points(values, init=(math.log(120.0),0.7)):
    vals = np.array([max(1e-3,float(v)) for v in values if 0 < float(v) <= 3600], dtype=float)
    if len(vals) < 20:
        return {"mu":float(init[0]),"sigma":float(init[1]),"n":int(len(vals)),"fitted":False}
    logs = np.log(vals)
    return {"mu":float(np.mean(logs)),"sigma":float(max(0.12,np.std(logs))),"n":int(len(vals)),"fitted":True}


def log_interval_prob(par, lower, upper):
    if upper <= max(0.0,lower):
        return -np.inf
    sigma = max(0.12, float(par['sigma']))
    scale = math.exp(float(par['mu']))
    Fu = lognorm.cdf(max(1e-6,upper), s=sigma, scale=scale)
    Fl = lognorm.cdf(max(0.0,lower), s=sigma, scale=scale) if lower > 0 else 0.0
    return math.log(max(Fu-Fl,1e-14))


def log_point_density(par, value):
    if value <= 0 or value > 3600:
        return -np.inf
    return float(lognorm.logpdf(value, s=max(0.12,float(par['sigma'])), scale=math.exp(float(par['mu']))))


def load_inputs(taims, p1c, service):
    p = json.loads(Path(p1c).read_text(encoding='utf-8'))
    entries = [x for x in p['canonical_entries'] if x.get('tier') in STRICT_TIERS]
    cw = {}
    for x in entries:
        k = str(x.get('out_stn_num') or '')
        if k and (k not in cw or x.get('tier') == AT_TIER):
            cw[k] = x
    groups = defaultdict(dict)
    for x in entries:
        if x.get('tier') == AT_TIER and x.get('dv_name') and x.get('service_subway_id') and x.get('service_statn_id'):
            groups[str(x['dv_name'])][str(x['service_subway_id'])] = str(x['service_statn_id'])
    transfer_pairs = defaultdict(list)
    for dv,g in groups.items():
        lines = sorted(g)
        for la in lines:
            for lb in lines:
                if la != lb:
                    transfer_pairs[(la,lb)].append((g[la],g[lb],dv))
    raw_trains = defaultdict(lambda:defaultdict(list))
    with open(service, encoding='utf-8', newline='') as f:
        for r in csv.DictReader(f):
            line = str(r.get('subway_id') or '').strip(); tr = str(r.get('btrain_no') or '').strip(); st = str(r.get('statn_id') or '').strip()
            a = dt(r.get('arrival_detect_time')); d = dt(r.get('departure_detect_time'))
            if line and tr and st and a and d:
                raw_trains[(line,tr)][st].append((a,d))
    train_events = {}; by_line = defaultdict(list)
    for k,sts in raw_trains.items():
        c = {}
        for st,evs in sts.items():
            e = rep_event(sorted(set(evs)))
            if e:
                c[st] = e
        if len(c) >= 2:
            train_events[k] = c; by_line[k[0]].append(k[1])
    for line in by_line:
        by_line[line] = sorted(set(by_line[line]))
    rows = []
    with zipfile.ZipFile(taims) as z:
        f = zr(z,'VW_KSCC_DX_CARD.csv')
        for r in csv.DictReader(f):
            if str(r.get('TRNS_MNS_CD') or '').strip() not in RAIL:
                continue
            t0 = dt(r.get('RIDE_DTM')); t1 = dt(r.get('ALGH_DTM'))
            if not t0 or not t1 or not (START <= t0 < END) or t1 <= t0 or t1-t0 > timedelta(hours=3):
                continue
            a = cw.get(z4(r.get('RIDE_BSST_ID'))); b = cw.get(z4(r.get('ALGH_BSST_ID')))
            if not a or not b:
                continue
            ol = str(a.get('service_subway_id') or ''); os = str(a.get('service_statn_id') or '')
            dl = str(b.get('service_subway_id') or ''); ds = str(b.get('service_statn_id') or '')
            if all([ol,os,dl,ds]):
                rows.append({'ol':ol,'os':os,'dl':dl,'ds':ds,'t0':t0,'t1':t1})
        f.close()
    return transfer_pairs, train_events, by_line, rows


def make_path_cache(train_events, by_line, excluded):
    cache = {}
    def get(line,o,d):
        key = (line,o,d)
        if key in cache:
            return cache[key]
        out = []
        for tr in by_line.get(line,[]):
            if (line,tr) in excluded:
                continue
            sts = train_events.get((line,tr),{})
            if o not in sts or d not in sts:
                continue
            dep = sts[o][1]; arr = sts[d][0]
            if dep < arr:
                out.append((dep,arr,tr))
        out.sort(key=lambda x:x[0]); cache[key] = out
        return out
    return get


def prev_dep(paths, idx, t0):
    for j in range(idx-1,-1,-1):
        d = paths[j][0]
        if d > t0:
            return d
        break
    return None


def assign_direct(row, get_paths, A_par, E_par):
    paths = get_paths(row['ol'],row['os'],row['ds']); best = None
    for i,(dep,arr,tr) in enumerate(paths):
        if dep <= row['t0'] or arr >= row['t1']:
            continue
        upper = sec(dep-row['t0'])
        if upper > 2400:
            continue
        pd = prev_dep(paths,i,row['t0']); lower = max(0.0,sec(pd-row['t0'])) if pd else 0.0
        eg = sec(row['t1']-arr)
        if eg <= 0 or eg > 1800:
            continue
        lp = log_interval_prob(A_par,lower,upper) + log_point_density(E_par,eg)
        if best is None or lp > best[0]:
            best = (lp,(lower,upper),eg,(dep,arr,tr))
    return best


def fit_access_egress(rows, get_paths):
    same = stable_sample([r for r in rows if r['ol']==r['dl'] and r['os']!=r['ds']], 30000, 'ae')
    A_par = {'mu':math.log(240.0),'sigma':0.8,'n':0,'fitted':False}; E_par = {'mu':math.log(120.0),'sigma':0.8,'n':0,'fitted':False}
    history = []
    for it in range(3):
        ints=[]; vals=[]; assigned=0
        for r in same:
            z = assign_direct(r,get_paths,A_par,E_par)
            if not z:
                continue
            assigned += 1; ints.append((z[1][0],z[1][1],1.0)); vals.append(z[2])
        A_par = fit_lognorm_intervals(ints,(A_par['mu'],A_par['sigma']))
        E_par = fit_lognorm_points(vals,(E_par['mu'],E_par['sigma']))
        history.append({'iter':it+1,'assigned':assigned,'A':A_par.copy(),'E':E_par.copy()})
    return A_par,E_par,history


def candidate_transfer_chains(row, transfer_pairs, get_paths, A_par, E_par, K_global, K_by_key):
    out=[]; ol,os,dl,ds = row['ol'],row['os'],row['dl'],row['ds']
    if ol == dl:
        return out
    for osx,dsx,dv in transfer_pairs.get((ol,dl),[]):
        up = get_paths(ol,os,osx); down = get_paths(dl,dsx,ds)
        if not up or not down:
            continue
        ups=[(i,x) for i,x in enumerate(up) if row['t0']<x[0]<row['t1'] and x[1]<row['t1']]
        dns=[(j,x) for j,x in enumerate(down) if row['t0']<x[0]<row['t1'] and x[1]<row['t1']]
        for i,(udep,uarr,utr) in ups[-5:]:
            au=sec(udep-row['t0'])
            if au<=0 or au>2400:
                continue
            pud=prev_dep(up,i,row['t0']); al=max(0.0,sec(pud-row['t0'])) if pud else 0.0
            la=log_interval_prob(A_par,al,au)
            if not np.isfinite(la):
                continue
            for j,(ddep,darr,dtr) in dns[:8]:
                if ddep<=uarr:
                    continue
                ku=sec(ddep-uarr)
                if ku<=0 or ku>2400:
                    continue
                pdd=prev_dep(down,j,uarr); kl=max(0.0,sec(pdd-uarr)) if pdd else 0.0
                key=f"{ol}->{dl}@{dv}"; kp=K_by_key.get(key,K_global)
                lk=log_interval_prob(kp,kl,ku); eg=sec(row['t1']-darr); le=log_point_density(E_par,eg)
                if np.isfinite(la+lk+le):
                    out.append((la+lk+le,key,(kl,ku),(al,au),eg,(utr,dtr,dv)))
    return out


def fit_transfer(rows, transfer_pairs, get_paths, A_par, E_par):
    cross = stable_sample([r for r in rows if r['ol']!=r['dl'] and transfer_pairs.get((r['ol'],r['dl']))], 18000, 'k')
    K_global={'mu':math.log(180.0),'sigma':0.65,'n':0,'fitted':False}; K_by={}; history=[]
    for it in range(2):
        by=defaultdict(list); allints=[]; assigned=0
        for r in cross:
            chains=candidate_transfer_chains(r,transfer_pairs,get_paths,A_par,E_par,K_global,K_by)
            if not chains:
                continue
            best=max(chains,key=lambda x:x[0]); assigned+=1; l,u=best[2]; by[best[1]].append((l,u,1.0)); allints.append((l,u,1.0))
        K_global=fit_lognorm_intervals(allints,(K_global['mu'],K_global['sigma'])); new={}
        for k,ints in by.items():
            if len(ints)>=35:
                new[k]=fit_lognorm_intervals(ints,(K_global['mu'],K_global['sigma']))
        K_by=new
        history.append({'iter':it+1,'assigned':assigned,'global':K_global.copy(),'specific_count':len(K_by),'top_specific':sorted([(k,v['n'],math.exp(v['mu']),v['sigma']) for k,v in K_by.items()],key=lambda x:x[1],reverse=True)[:20]})
    return K_global,K_by,history


def target_trains(train_events):
    rows=[]
    for (line,tr),sts in train_events.items():
        if line!=L2 or A_ST not in sts or B_ST not in sts:
            continue
        ta=sts[A_ST][1]; tb=sts[B_ST][1]; ref=min(ta,tb)
        if START<=ref<END:
            rows.append((ref,tr,'A_TO_B' if ta<tb else 'B_TO_A'))
    return sorted(rows)


def select_targets(tt):
    by=defaultdict(list)
    for x in tt:
        by[x[2]].append(x)
    selected=[]
    for direction,arr in by.items():
        if not arr:
            continue
        for i in sorted(set([len(arr)//3,(2*len(arr))//3])):
            i=min(max(1,i),len(arr)-2) if len(arr)>=3 else min(i,len(arr)-1)
            selected.append(arr[i])
    return sorted(selected)


def trajectory_offsets(train_events, direction, excluded):
    vals=defaultdict(lambda:[[],[]])
    for (line,tr),sts in train_events.items():
        if line!=L2 or (line,tr) in excluded or A_ST not in sts or B_ST not in sts:
            continue
        d='A_TO_B' if sts[A_ST][1]<sts[B_ST][1] else 'B_TO_A'
        if d!=direction:
            continue
        ref=sts[A_ST][1] if direction=='A_TO_B' else sts[B_ST][1]
        for st,(arr,dep) in sts.items():
            vals[st][0].append(sec(arr-ref)); vals[st][1].append(sec(dep-ref))
    return {st:(float(np.median(aa)),float(np.median(dd)),len(aa)) for st,(aa,dd) in vals.items() if len(aa)>=5}


def target_path(offsets, ref, o, d):
    if o not in offsets or d not in offsets:
        return None
    dep=ref+timedelta(seconds=offsets[o][1]); arr=ref+timedelta(seconds=offsets[d][0])
    return (dep,arr) if dep<arr else None


def target_direct_log(row, ref, offsets, get_paths, A_par, E_par):
    tp=target_path(offsets,ref,row['os'],row['ds'])
    if not tp or row['ol']!=L2 or row['dl']!=L2:
        return -np.inf
    dep,arr=tp
    if dep<=row['t0'] or arr>=row['t1']:
        return -np.inf
    paths=get_paths(L2,row['os'],row['ds']); pds=[x[0] for x in paths if row['t0']<x[0]<dep]
    lower=max(0.0,sec(max(pds)-row['t0'])) if pds else 0.0; upper=sec(dep-row['t0']); eg=sec(row['t1']-arr)
    return log_interval_prob(A_par,lower,upper)+log_point_density(E_par,eg)


def visible_direct_log(row,get_paths,A_par,E_par):
    z=assign_direct(row,get_paths,A_par,E_par)
    return z[0] if z else -np.inf


def target_inbound_logs(row,ref,offsets,transfer_pairs,get_paths,A_par,E_par,K_global,K_by):
    if row['ol']==L2 or row['dl']!=L2:
        return []
    out=[]
    for ost,l2st,dv in transfer_pairs.get((row['ol'],L2),[]):
        tp=target_path(offsets,ref,l2st,row['ds'])
        if not tp:
            continue
        tdep,tarr=tp
        if tarr>=row['t1']:
            continue
        up=get_paths(row['ol'],row['os'],ost); downvis=get_paths(L2,l2st,row['ds'])
        prev_down=[x[0] for x in downvis if x[0]<tdep]; prev_target_dep=max(prev_down) if prev_down else None
        for i,(udep,uarr,utr) in enumerate(up):
            if udep<=row['t0'] or uarr>=tdep:
                continue
            au=sec(udep-row['t0'])
            if au>2400:
                continue
            pud=prev_dep(up,i,row['t0']); al=max(0.0,sec(pud-row['t0'])) if pud else 0.0
            ku=sec(tdep-uarr); kl=max(0.0,sec(prev_target_dep-uarr)) if prev_target_dep and prev_target_dep>uarr else 0.0; eg=sec(row['t1']-tarr)
            key=f"{row['ol']}->{L2}@{dv}"; lp=log_interval_prob(A_par,al,au)+log_interval_prob(K_by.get(key,K_global),kl,ku)+log_point_density(E_par,eg)
            if np.isfinite(lp):
                out.append(lp)
    return out


def target_outbound_logs(row,ref,offsets,transfer_pairs,get_paths,A_par,E_par,K_global,K_by):
    if row['ol']!=L2 or row['dl']==L2:
        return []
    out=[]
    for l2st,dst,dv in transfer_pairs.get((L2,row['dl']),[]):
        tp=target_path(offsets,ref,row['os'],l2st)
        if not tp:
            continue
        tdep,tarr=tp
        if tdep<=row['t0']:
            continue
        upvis=get_paths(L2,row['os'],l2st); prev_up=[x[0] for x in upvis if row['t0']<x[0]<tdep]
        al=max(0.0,sec(max(prev_up)-row['t0'])) if prev_up else 0.0; au=sec(tdep-row['t0']); la=log_interval_prob(A_par,al,au)
        if not np.isfinite(la):
            continue
        down=get_paths(row['dl'],dst,row['ds'])
        for j,(ddep,darr,dtr) in enumerate(down):
            if ddep<=tarr or darr>=row['t1']:
                continue
            ku=sec(ddep-tarr)
            if ku>2400:
                continue
            pdd=prev_dep(down,j,tarr); kl=max(0.0,sec(pdd-tarr)) if pdd else 0.0; eg=sec(row['t1']-darr)
            key=f"{L2}->{row['dl']}@{dv}"; lp=la+log_interval_prob(K_by.get(key,K_global),kl,ku)+log_point_density(E_par,eg)
            if np.isfinite(lp):
                out.append(lp)
    return out


def visible_transfer_log(row,transfer_pairs,get_paths,A_par,E_par,K_global,K_by):
    ch=candidate_transfer_chains(row,transfer_pairs,get_paths,A_par,E_par,K_global,K_by)
    return float(logsumexp(sorted([x[0] for x in ch],reverse=True)[:20])) if ch else -np.inf


def score_target(rowsets,ref,offsets,transfer_pairs,get_paths,A_par,E_par,K_global,K_by,vis_cache):
    scores={}
    for tier in ('DIRECT','DIRECT_PLUS_INBOUND','DIRECT_PLUS_BOTH'):
        total=0.0; used=0; improved=0
        groups=['direct'] if tier=='DIRECT' else (['direct','inbound'] if tier=='DIRECT_PLUS_INBOUND' else ['direct','inbound','outbound'])
        for g in groups:
            for idx,row in enumerate(rowsets[g]):
                lv=vis_cache.get((g,idx),-np.inf)
                if g=='direct':
                    lt=target_direct_log(row,ref,offsets,get_paths,A_par,E_par); lts=[lt] if np.isfinite(lt) else []
                elif g=='inbound':
                    lts=target_inbound_logs(row,ref,offsets,transfer_pairs,get_paths,A_par,E_par,K_global,K_by)
                else:
                    lts=target_outbound_logs(row,ref,offsets,transfer_pairs,get_paths,A_par,E_par,K_global,K_by)
                if not np.isfinite(lv) and not lts:
                    continue
                if lts:
                    lt=float(logsumexp(sorted(lts,reverse=True)[:30])); joint=np.logaddexp(lv,lt) if np.isfinite(lv) else lt
                    if np.isfinite(lv) and joint>lv+1e-9:
                        improved+=1
                else:
                    joint=lv
                if np.isfinite(joint):
                    total+=joint; used+=1
        scores[tier]={'loglik':float(total),'used':used,'target_improved_rows':improved}
    return scores


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--taims',required=True); ap.add_argument('--p1c',required=True); ap.add_argument('--service',required=True); ap.add_argument('--out',required=True); args=ap.parse_args()
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    transfer_pairs,train_events,by_line,rows=load_inputs(args.taims,args.p1c,args.service)
    tt=target_trains(train_events); selected=select_targets(tt)
    result={'schema':'mppd.n3-shared-kernel-joint-likelihood-pilot.v1','date':'2026-09-04','status':'N3_SHARED_KERNEL_JOINT_LIKELIHOOD_PILOT_COMPLETED','scientific_boundary':['Known-neighbor-gap conditional hidden-service pilot, not free-search reconstruction.','Waiting is not a free kernel: access and transfer enter as first-feasible-service interval probabilities induced by discrete service gates.','Transfer kernels are fitted from visible-service passenger chains with direction/physical-station-specific kernels where support is sufficient and pooled fallback otherwise.','BMS is partial service-anchor evidence, not exhaustive ATS.','Direct and one-transfer factors only in this pilot; double-transfer and pulse factors are deferred.','No card identifiers are retained.'],'targets':[]}
    for truth_ref,tr,direction in selected:
        excluded={(L2,tr)}; get_paths=make_path_cache(train_events,by_line,excluded)
        A_par,E_par,ae_hist=fit_access_egress(rows,get_paths); K_global,K_by,k_hist=fit_transfer(rows,transfer_pairs,get_paths,A_par,E_par); offsets=trajectory_offsets(train_events,direction,excluded)
        direct=[r for r in rows if r['ol']==L2 and r['dl']==L2 and target_path(offsets,truth_ref,r['os'],r['ds'])]
        inbound=[r for r in rows if r['ol']!=L2 and r['dl']==L2 and transfer_pairs.get((r['ol'],L2))]
        outbound=[r for r in rows if r['ol']==L2 and r['dl']!=L2 and transfer_pairs.get((L2,r['dl']))]
        rowsets={'direct':stable_sample(direct,5000,f'{tr}-d'),'inbound':stable_sample(inbound,5000,f'{tr}-i'),'outbound':stable_sample(outbound,5000,f'{tr}-o')}
        vis_cache={}
        for g in rowsets:
            for idx,r in enumerate(rowsets[g]):
                vis_cache[(g,idx)]=visible_direct_log(r,get_paths,A_par,E_par) if g=='direct' else visible_transfer_log(r,transfer_pairs,get_paths,A_par,E_par,K_global,K_by)
        direction_rows=[x for x in tt if x[2]==direction and x[1]!=tr]; before=[x for x in direction_rows if x[0]<truth_ref]; after=[x for x in direction_rows if x[0]>truth_ref]
        if not before or not after:
            continue
        prev=max(before,key=lambda x:x[0]); nxt=min(after,key=lambda x:x[0]); start=prev[0]+timedelta(seconds=30); stop=nxt[0]-timedelta(seconds=30)
        if start>=stop:
            continue
        grid=[]; cur=start
        while cur<=stop:
            grid.append(cur); cur+=timedelta(seconds=30)
        tiers=defaultdict(list)
        for cand in grid:
            sc=score_target(rowsets,cand,offsets,transfer_pairs,get_paths,A_par,E_par,K_global,K_by,vis_cache)
            for tier,v in sc.items():
                tiers[tier].append({'candidate':cand.isoformat(),**v})
        tier_summary={}
        for tier,vals in tiers.items():
            best=max(vals,key=lambda x:x['loglik']); bt=datetime.fromisoformat(best['candidate']); err=abs(sec(bt-truth_ref)); sv=sorted(vals,key=lambda x:x['loglik'],reverse=True); second=sv[1]['loglik'] if len(sv)>1 else None
            tier_summary[tier]={'best_candidate':best['candidate'],'abs_error_sec':err,'within_60':bool(err<=60),'within_120':bool(err<=120),'best_loglik':best['loglik'],'second_best_gap':None if second is None else float(best['loglik']-second),'used_rows':best['used'],'target_improved_rows':best['target_improved_rows']}
        midpoint=prev[0]+(nxt[0]-prev[0])/2
        result['targets'].append({'train':tr,'direction':direction,'truth_ref':truth_ref.isoformat(),'neighbor_gap':{'prev':prev[0].isoformat(),'next':nxt[0].isoformat(),'midpoint_abs_error_sec':abs(sec(midpoint-truth_ref)),'gap_selection_uses_target_truth':True},'kernel_fit':{'A':A_par,'E':E_par,'K_global':K_global,'K_specific_count':len(K_by),'A_median_sec':math.exp(A_par['mu']),'E_median_sec':math.exp(E_par['mu']),'K_global_median_sec':math.exp(K_global['mu']),'access_egress_history':ae_hist,'transfer_history':k_hist},'cohort_sizes':{k:len(v) for k,v in rowsets.items()},'tiers':tier_summary})
    agg={}
    for tier in ('DIRECT','DIRECT_PLUS_INBOUND','DIRECT_PLUS_BOTH'):
        errs=[t['tiers'][tier]['abs_error_sec'] for t in result['targets'] if tier in t['tiers']]
        agg[tier]={'n':len(errs),'median_abs_error_sec':float(np.median(errs)) if errs else None,'within_60_share':float(np.mean([e<=60 for e in errs])) if errs else None,'within_120_share':float(np.mean([e<=120 for e in errs])) if errs else None}
    result['aggregate']=agg; result['next_gate']='If shared-kernel probabilistic tiers show stable network gain, expand to double-transfer and pulse factors; otherwise retain the negative and inspect route-mixture/kernel misspecification before free-search.'
    (out/'n3_shared_kernel_joint_likelihood_pilot_summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8'); print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
