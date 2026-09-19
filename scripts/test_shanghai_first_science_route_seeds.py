from __future__ import annotations
import unittest
from shanghai_first_science_route_seeds import classify,l11_period,l2_period,parse_token

class T(unittest.TestCase):
    def test_token_identity(self):
        self.assertEqual(parse_token("4号线浦电路")[2],"04::浦电路")
        self.assertEqual(parse_token("6号线浦电路")[2],"06::浦电路")
        self.assertEqual(parse_token("11号线李子园路")[2],"李子园")
    def test_periods(self):
        self.assertEqual(l11_period("08:00:00"),"AM_0700_0830")
        self.assertEqual(l2_period("18:00:00"),"PM_DOC_1730_1900")
    def test_family(self):
        trunk={"嘉定新城","马陆","南翔"}
        self.assertEqual(classify("花桥","南翔",trunk),"L11_HUAQIAO_TO_TRUNK")
        self.assertEqual(classify("嘉定北","花桥",trunk),"L11_CROSS_BRANCH")
        self.assertEqual(classify("人民广场","唐镇",trunk),"L2_INNER_TO_EAST")
if __name__=="__main__": unittest.main()
