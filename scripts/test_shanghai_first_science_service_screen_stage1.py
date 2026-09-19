from __future__ import annotations
import unittest
from shanghai_first_science_service_screen_stage1 import progression,classify_slice,summarise_slice

PRIOR={
 ("11",frozenset(("A","B"))):{"p10":90.0,"median":100.0,"p90":110.0,"n":3},
 ("11",frozenset(("B","C"))):{"p10":80.0,"median":90.0,"p90":100.0,"n":3},
}
WORLDS={
 "L11":{"periods":{
   "AM_0700_0830":{"phase_worlds":[{"id":"u"}]},
   "OFF_0900_1600":{"phase_worlds":[{"id":"u"}]},
   "PM_1700_1930":{"phase_worlds":[{"id":"u"}]}
 }},
 "L02":{
   "date_classes":{"2016-07-01":"PRE_20160812","2016-09-01":"POST_20160812"},
   "AM_0700_0900":{"all_dates":{"stream_specific_headways":"UNRESOLVED"}},
   "PM":{
     "PRE_20160812":{"stream_specific_headways":"UNRESOLVED"},
     "POST_20160812":{"stream_specific_headways":"UNRESOLVED","stream_share":"UNRESOLVED"}
   }
 }
}
class T(unittest.TestCase):
    def test_progression(self):
        p=progression(["11::A","11::B","11::C"],PRIOR)
        self.assertEqual(p["ride_edges"],2);self.assertEqual(p["prior_missing_ride_edges"],0)
        self.assertEqual(p["t_prog_median_s"],190.0)
    def test_l11_qualified_period_is_phase_dependent(self):
        v={"path_count":1,"line_changes":0};p={"prior_missing_ride_edges":0,"transfer_edges":0}
        self.assertEqual(classify_slice("L11_HUAQIAO_TO_TRUNK","20160701","AM_0700_0830",v,p,WORLDS)[0],"PHASE_DEPENDENT")
    def test_l11_other_is_unresolved_not_phase(self):
        v={"path_count":1,"line_changes":0};p={"prior_missing_ride_edges":0,"transfer_edges":0}
        self.assertEqual(classify_slice("L11_HUAQIAO_TO_TRUNK","20160701","OTHER",v,p,WORLDS)[1],"L11_PERIOD_SERVICE_WORLD_NOT_QUALIFIED")
    def test_tied_unresolved(self):
        v={"path_count":2,"line_changes":0};p={"prior_missing_ride_edges":0,"transfer_edges":0}
        self.assertEqual(classify_slice("L11_HUAQIAO_TO_TRUNK","20160701","AM_0700_0830",v,p,WORLDS)[1],"TIED_GEOMETRY_NOT_EXPANDED")
    def test_l2_post_pm_share_unresolved(self):
        v={"path_count":1,"line_changes":0};p={"prior_missing_ride_edges":0,"transfer_edges":0}
        self.assertEqual(classify_slice("L2_INNER_TO_EAST","20160901","PM_DOC_1730_1900",v,p,WORLDS)[1],"L2_PM_POST_STREAM_SHARE_UNRESOLVED")
    def test_slice_summary(self):
        self.assertEqual(summarise_slice({"PHASE_DEPENDENT"}),"ONLY_PHASE_DEPENDENT")
        self.assertEqual(summarise_slice({"SERVICE_UNRESOLVED"}),"ONLY_SERVICE_UNRESOLVED")
        self.assertEqual(summarise_slice({"SERVICE_UNRESOLVED","PHASE_DEPENDENT"}),"MIXED")
if __name__=="__main__": unittest.main()
