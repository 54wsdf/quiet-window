from __future__ import annotations
import unittest
from shanghai_route_clear_afc_residual_stage3 import route_clear_record

def rec(world,path=("11::A","11::B"),count=1,changes=0,missing=0,prog=100.0):
    return {
      "world":world,
      "vectors":[{
        "representative_path":list(path),
        "path_count":count,
        "line_changes":changes,
        "prior_missing_ride_edges":missing,
        "t_prog_median_s":prog,
        "ride_intervals":1,
        "support_slices":[{"day":"20160701","period":"AM_0700_0830","journeys":10}]
      }]
    }

class T(unittest.TestCase):
    def test_stable_clear(self):
        x=route_clear_record([rec("w1"),rec("w2")],["w1","w2"])
        self.assertIsNotNone(x); self.assertEqual(x["t_prog_median_s"],100.0)
    def test_tied_not_clear(self):
        self.assertIsNone(route_clear_record([rec("w1",count=2),rec("w2",count=2)],["w1","w2"]))
    def test_world_path_change_not_clear(self):
        self.assertIsNone(route_clear_record([rec("w1"),rec("w2",path=("11::A","11::C"))],["w1","w2"]))
    def test_transfer_not_clear(self):
        self.assertIsNone(route_clear_record([rec("w1",changes=1),rec("w2",changes=1)],["w1","w2"]))
if __name__=="__main__":unittest.main()
