from __future__ import annotations

import unittest

from shanghai_station_progression_prior import norm_code, parse_clock


class TestParsing(unittest.TestCase):
    def test_norm_code(self):
        by_code = {'0111': {}, '0112': {}}
        self.assertEqual(norm_code('000111', by_code), '0111')
        self.assertEqual(norm_code('0112', by_code), '0112')
        self.assertIsNone(norm_code('999999', by_code))

    def test_parse_clock(self):
        self.assertEqual(parse_clock('19800'), 19800)
        self.assertIsNone(parse_clock('—'))
        self.assertIsNone(parse_clock(''))


if __name__ == '__main__':
    unittest.main()
