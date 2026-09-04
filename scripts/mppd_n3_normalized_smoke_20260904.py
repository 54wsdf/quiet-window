import scripts.mppd_n3_normalized_schedule_likelihood_20260904 as n

orig_sample = n.m.stable_sample

def small_sample(rows, cap, salt):
    return orig_sample(rows, min(cap, 1000), salt)


def one_target(tt):
    picked = [x for x in tt if x[1] == '3060']
    if picked:
        return picked[:1]
    return n.m.select_targets(tt)[:1]

n.m.stable_sample = small_sample
n.m.select_targets = one_target
n.run()
