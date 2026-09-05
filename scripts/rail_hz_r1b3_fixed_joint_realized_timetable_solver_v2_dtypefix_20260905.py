from __future__ import annotations

import numpy as np

import scripts.rail_hz_r1b3_fixed_joint_realized_timetable_solver_20260905 as base


def structural_factors_fixed(roots, service_init, model):
    ctx = base.prior.contexts_by_edge(service_init)
    pu_i=[]; pu_y=[]; pu_c=[]
    tr_i=[]; tr_j=[]; tr_lag=[]; tr_c=[]
    for r in roots:
        rid=str(r["root_id"]); fw=float(model["family_weight"][rid]); evs=list(r["events"])
        for ev in evs:
            if bool(ev.get("matched_observed_pulse",False)):
                ni=model["root_station_to_node"][(rid,int(ev["station"]))]
                sd=max(1e-6,float(ev["sd_s"]))
                pu_i.append(ni); pu_y.append(float(ev["time_s"])); pu_c.append(fw/(sd*sd))
        for left,right in zip(evs[:-1],evs[1:]):
            lag,sd,_eclass=base.prior.transition_factor(str(r["path_id"]),str(r["direction"]),left,right,ctx)
            i=model["root_station_to_node"][(rid,int(left["station"]))]
            j=model["root_station_to_node"][(rid,int(right["station"]))]
            tr_i.append(i); tr_j.append(j); tr_lag.append(lag); tr_c.append(fw/(sd*sd))
    return (
        np.asarray(pu_i,np.int32), np.asarray(pu_y,float), np.asarray(pu_c,float),
        np.asarray(tr_i,np.int32), np.asarray(tr_j,np.int32), np.asarray(tr_lag,float), np.asarray(tr_c,float),
    )


def main():
    base.structural_factors = structural_factors_fixed
    base.main()


if __name__ == "__main__":
    main()
