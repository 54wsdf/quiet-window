from __future__ import annotations
import csv,tempfile,unittest
from pathlib import Path
from shanghai_sptcc_formal_ingest import ingest
HEADER="卡ID,刷卡日期,刷卡时间,刷卡站点,刷卡乘车类型,刷卡扣钱,是否优惠\n"
class T(unittest.TestCase):
    def run_case(self,rows):
        td=tempfile.TemporaryDirectory();root=Path(td.name);raw=root/"x.csv"
        raw.write_text(HEADER+"".join(rows),encoding="utf-8");out=root/"j.csv.gz";audit=root/"a.json"
        payload=ingest(raw,root/"e.sqlite",out,audit)
        import gzip
        with gzip.open(out,"rt",encoding="utf-8",newline="") as f:data=list(csv.DictReader(f))
        return td,data,payload
    def test_basic(self):
        td,d,p=self.run_case(["c,2016-07-01,08:00:00,2号线陆家嘴,地铁,0,否\n","c,2016-07-01,08:20:00,2号线南京西路,地铁,4,否\n"]);self.assertEqual(len(d),1);self.assertNotIn("card_token",d[0]);td.cleanup()
    def test_station_barrier(self):
        td,d,p=self.run_case(["c,2016-07-01,08:00:00,2号线陆家嘴,地铁,0,否\n","c,2016-07-01,08:10:00,坏token,地铁,4,否\n","c,2016-07-01,08:20:00,2号线南京西路,地铁,4,否\n"]);self.assertEqual(d,[]);td.cleanup()
    def test_duplicate_barrier(self):
        td,d,p=self.run_case(["c,2016-07-01,08:00:00,2号线陆家嘴,地铁,0,否\n","c,2016-07-01,08:20:00,2号线南京西路,地铁,4,否\n","c,2016-07-01,08:20:00,2号线陆家嘴,地铁,0,否\n","c,2016-07-01,09:00:00,2号线南京西路,地铁,4,否\n"]);self.assertEqual(d,[]);self.assertEqual(p["duplicate_timestamp_events"],2);td.cleanup()
    def test_over_six_hours(self):
        td,d,p=self.run_case(["c,2016-07-01,01:00:00,2号线陆家嘴,地铁,0,否\n","c,2016-07-01,08:00:01,2号线南京西路,地铁,4,否\n"]);self.assertEqual(d,[]);td.cleanup()
    def test_virtual_stitch(self):
        td,d,p=self.run_case(["c,2016-07-01,07:50:00,2号线陆家嘴,地铁,0,否\n","c,2016-07-01,08:10:00,2号线虹桥2号航站楼,地铁,4,否\n","c,2016-07-01,08:15:00,10号线虹桥2号航站楼,地铁,0,否\n","c,2016-07-01,08:40:00,10号线新江湾城,地铁,5,否\n"]);self.assertEqual(len(d),1);self.assertEqual(d[0]["virtual_transfer_count"],"1");td.cleanup()
    def test_nonmetro_pair_barrier(self):
        td,d,p=self.run_case(["c,2016-07-01,08:00:00,2号线陆家嘴,地铁,0,否\n","c,2016-07-01,08:10:00,公交站,公交,2,否\n","c,2016-07-01,08:20:00,2号线南京西路,地铁,4,否\n"]);self.assertEqual(d,[]);self.assertEqual(p["barrier_non_metro"],1);td.cleanup()
    def test_nonmetro_blocks_stitch(self):
        td,d,p=self.run_case(["c,2016-07-01,07:50:00,2号线陆家嘴,地铁,0,否\n","c,2016-07-01,08:10:00,2号线虹桥2号航站楼,地铁,4,否\n","c,2016-07-01,08:12:00,公交站,公交,2,否\n","c,2016-07-01,08:15:00,10号线虹桥2号航站楼,地铁,0,否\n","c,2016-07-01,08:40:00,10号线新江湾城,地铁,5,否\n"]);self.assertEqual(len(d),2);self.assertEqual(p["virtual_transfer_candidate_blocked_by_intervening_event"],1);td.cleanup()
    def test_unmatched_metro_blocks_stitch(self):
        td,d,p=self.run_case(["c,2016-07-01,07:50:00,2号线陆家嘴,地铁,0,否\n","c,2016-07-01,08:10:00,2号线虹桥2号航站楼,地铁,4,否\n","c,2016-07-01,08:12:00,2号线虹桥2号航站楼,地铁,1,否\n","c,2016-07-01,08:15:00,10号线虹桥2号航站楼,地铁,0,否\n","c,2016-07-01,08:40:00,10号线新江湾城,地铁,5,否\n"]);self.assertEqual(len(d),2);self.assertEqual(p["virtual_transfer_candidate_blocked_by_intervening_event"],1);td.cleanup()
if __name__=="__main__":unittest.main()
