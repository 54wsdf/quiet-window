from __future__ import annotations
import unittest
from shanghai_route_frontier import build_adjacency,solve_frontier
from shanghai_first_science_tied_geometry_stage2 import label_progression_bounds

class T(unittest.TestCase):
    def labels(self):
        nodes=["11::A","11::B","11::C","11::D"]
        edges=[
          ("11::A","11::B","ride"),("11::B","11::D","ride"),
          ("11::A","11::C","ride"),("11::C","11::D","ride")
        ]
        return solve_frontier(build_adjacency(nodes,edges),["11::A"])
    def test_two_tied_paths_bounds(self):
        prior={
          ("11",frozenset(("A","B"))):{"p10":9.0,"median":10.0,"p90":11.0,"n":1},
          ("11",frozenset(("B","D"))):{"p10":9.0,"median":10.0,"p90":11.0,"n":1},
          ("11",frozenset(("A","C"))):{"p10":19.0,"median":20.0,"p90":21.0,"n":1},
          ("11",frozenset(("C","D"))):{"p10":19.0,"median":20.0,"p90":21.0,"n":1},
        }
        b=label_progression_bounds(self.labels(),["11::D"],(2,0),prior)
        self.assertEqual(b["total_paths"],2)
        self.assertEqual(b["fully_progression_covered_paths"],2)
        self.assertEqual(b["covered_path_median_sum_min_s"],20.0)
        self.assertEqual(b["covered_path_median_sum_max_s"],40.0)
        self.assertFalse(b["all_tied_paths_progression_equivalent"])
    def test_partial_coverage(self):
        prior={
          ("11",frozenset(("A","B"))):{"p10":9.0,"median":10.0,"p90":11.0,"n":1},
          ("11",frozenset(("B","D"))):{"p10":9.0,"median":10.0,"p90":11.0,"n":1},
          ("11",frozenset(("A","C"))):{"p10":19.0,"median":20.0,"p90":21.0,"n":1},
        }
        b=label_progression_bounds(self.labels(),["11::D"],(2,0),prior)
        self.assertEqual(b["total_paths"],2)
        self.assertEqual(b["fully_progression_covered_paths"],1)
        self.assertFalse(b["all_paths_progression_covered"])
if __name__=="__main__":unittest.main()
