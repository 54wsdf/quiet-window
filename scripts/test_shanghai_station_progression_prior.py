from __future__ import annotations

import unittest

from shanghai_station_progression_prior import numeric, suffix4


class TestParsing(unittest.TestCase):
    def test_suffix4(self):
        self.assertEqual(suffix4("000111"), "0111")
        self.assertEqual(suffix4("0111"), "0111")
        self.assertIsNone(suffix4("—"))

    def test_numeric(self):
        self.assertEqual(numeric("19800"), 19800)
        self.assertEqual(numeric('"19800"'), 19800)
        self.assertIsNone(numeric("—"))


if __name__ == "__main__":
    unittest.main()
