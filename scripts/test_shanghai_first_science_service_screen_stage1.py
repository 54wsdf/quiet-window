from __future__ import annotations
import unittest
from shanghai_first_science_service_screen_stage1 import progression,classify_vector

PRIOR={
 ("11",frozenset(("A","B"))):{"p10":90.0,"median":100.0,"p90":110.0,"n":3},
 ("11",frozenset(("B","C"))):{"p10":80.0,"median":90.0,"p90":100.0,"n":3},
}
class T(unittest.TestCase):
    def test_progression(self):
        p=progression(["11::A","11::B","11::C"],PRIOR)
        self.assertEqual(p["ride_edges"],2);self.assertEqual(p["prior_missing_ride_edges"],0)
        self.assertEqual(p["t_prog_median_s"],190.0)
    def test_l11_phase(self):
        v={"path_count":1,"line_changes":0}
        p={"prior_missing_ride_edges":0,"transfer_edges":0}
        self.assertEqual(classify_vector("L11_HUAQIAO_TO_TRUNK",v,p)[0],"PHASE_DEPENDENT")
    def test_tied_unresolved(self):
        v={"path_count":2,"line_changes":0}
        p={"prior_missing_ride_edges":0,"transfer_edges":0}
        self.assertEqual(classify_vector("L11_HUAQIAO_TO_TRUNK",v,p)[1],"TIED_GEOMETRY_NOT_EXPANDED")
    def test_l2_unresolved(self):
        v={"path_count":1,"line_changes":0};p={"prior_missing_ride_edges":0,"transfer_edges":0}
        self.assertEqual(classify_vector("L2_INNER_TO_EAST",v,p)[0],"SERVICE_UNRESOLVED")
if __name__=="__main__":unittest.main()
