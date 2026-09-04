import scripts.mppd_n3_segment_conditioned_normalized_likelihood_20260904 as s

orig_sample=s.m.stable_sample

def small(rows,cap,salt):
    return orig_sample(rows,min(cap,1000),salt)

def one(tt):
    x=[r for r in tt if r[1]=='3060']
    return x[:1] if x else s.m.select_targets(tt)[:1]

s.m.stable_sample=small
s.m.select_targets=one
s.run()
