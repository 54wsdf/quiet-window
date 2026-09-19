from __future__ import annotations
import csv,json,tempfile,unittest
from pathlib import Path
from shanghai_sptcc_formal_ingest import ingest

HEADER="卡ID,刷卡日期,刷卡时间,刷卡站点,刷卡乘车类型,刷卡扣钱,是否优惠\n"
KEY=bytes.fromhex("11"*32)

class T(unittest.TestCase):
    def run_case(self,rows):
        td=tempfile.TemporaryDirectory();root=Path(td.name)
        raw=root/"x.csv";raw.write_text(HEADER+"".join(rows),encoding="utf-8")
        out=root/"j.csv.gz";audit=root/"a.json"
        payload=ingest(raw,root/"e.sqlite",out,audit,KEY)
        import gzip
        with gzip.open(out,"rt",encoding="utf-8",newline="") as f:data=list(csv.DictReader(f))
        return td,data,payload

    def test_basic_and_stable_split(self):
        td,data,p=self.run_case([
          "c,2016-07-01,08:00:00,2号线陆家嘴,地铁,0,否\n",
          "c,2016-07-01,08:20:00,2号线南京西路,地铁,4,否\n"])
        self.assertEqual(len(data),1);self.assertIn(data[0]["split"],{"train","validation","test"})
        self.assertNotEqual(data[0]["card_token"],"c");td.cleanup()

    def test_station_reject_barrier(self):
        td,data,p=self.run_case([
          "c,2016-07-01,08:00:00,2号线陆家嘴,地铁,0,否\n",
          "c,2016-07-01,08:10:00,坏token,地铁,4,否\n",
          "c,2016-07-01,08:20:00,2号线南京西路,地铁,4,否\n"])
        self.assertEqual(data,[]);self.assertEqual(p["station_barrier_events"],1);td.cleanup()

    def test_duplicate_timestamp_barrier(self):
        td,data,p=self.run_case([
          "c,2016-07-01,08:00:00,2号线陆家嘴,地铁,0,否\n",
          "c,2016-07-01,08:20:00,2号线南京西路,地铁,4,否\n",
          "c,2016-07-01,08:20:00,2号线陆家嘴,地铁,0,否\n",
          "c,2016-07-01,09:00:00,2号线南京西路,地铁,4,否\n"])
        self.assertEqual(data,[]);self.assertEqual(p["duplicate_timestamp_events"],2);td.cleanup()

    def test_over_six_hours_rejected(self):
        td,data,p=self.run_case([
          "c,2016-07-01,01:00:00,2号线陆家嘴,地铁,0,否\n",
          "c,2016-07-01,08:00:01,2号线南京西路,地铁,4,否\n"])
        self.assertEqual(data,[]);self.assertEqual(p["gate_segment_over_max_duration"],1);td.cleanup()

    def test_virtual_transfer_stitch(self):
        td,data,p=self.run_case([
          "c,2016-07-01,07:50:00,2号线陆家嘴,地铁,0,否\n",
          "c,2016-07-01,08:10:00,2号线虹桥2号航站楼,地铁,4,否\n",
          "c,2016-07-01,08:15:00,10号线虹桥2号航站楼,地铁,0,否\n",
          "c,2016-07-01,08:40:00,10号线新江湾城,地铁,5,否\n"])
        self.assertEqual(len(data),1);self.assertEqual(data[0]["virtual_transfer_count"],"1")
        self.assertIn("HONGQIAO_T2",data[0]["virtual_transfer_policy_ids"]);td.cleanup()

if __name__=="__main__":unittest.main()
