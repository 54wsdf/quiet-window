from __future__ import annotations

import scripts.mppd_r1_hz_birth_lineage_audit as a


def tr(tid,line,direction,path,t,offset=0.0,support=8,evidence=1.0):
    station=a.birth.LINE_ANCHOR_STATION[line]
    events=[]
    nodes=[station,station+1,station+2,station+3]
    for i,s in enumerate(nodes):
        events.append({'station':s,'sequence_index':i,'time_s':t+i*100+offset})
    return {
        'trajectory_id':tid,'afc_line':line,'direction':direction,'path_id':path,
        'reference_time_s':t,'support_station_count':support,'support_event_count':support,
        'evidence_score':evidence,'path_ambiguous':False,'events':events
    }


def test_lineage_finds_nearest_parent_and_parallel_offset():
    p=tr('p','A','Down','A_main',100.0)
    c=tr('c','A','Down','A_main',145.0)
    row=a.lineage_row([p],c)
    assert row['nearest_parent_trajectory_id']=='p'
    assert row['nearest_parent_anchor_distance_s']==45.0
    assert row['nearest_parent_same_path'] is True
    assert row['common_station_count']==4
    assert row['parallel_signed_offset_median_s']==45.0
    assert row['parallel_offset_mad_s']==0.0


def test_headway_audit_counts_short_gaps():
    rows=[tr('a','A','Down','A_main',100.0),tr('b','A','Down','A_main',125.0),tr('c','A','Down','A_main',200.0)]
    h=a.headway_audit(rows)
    assert h['gap_lt_30_count']==1
    assert h['gap_lt_45_count']==1
    assert h['gap_lt_60_count']==1


def test_build_audit_does_not_call_parallel_sibling_qualified():
    parent={'trajectories':[tr('p','A','Down','A_main',100.0)]}
    births=[tr('b','A','Down','A_main',160.0,evidence=1.2)]
    aug={'trajectories':parent['trajectories']+births}
    out=a.build_audit(parent,births,aug)
    assert out['birth_count']==1
    assert out['lineage_class_counts']['DISTINCT_PARALLEL_SIBLING_SAME_PATH']==1
    assert out['semantics']['parallel_sibling_is_not_automatically_qualified_as_distinct_train'] is True
