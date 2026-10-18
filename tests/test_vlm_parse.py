"""Pins the tolerant reply parser (tools/dsh/vlm_mosaic.parse_tile_lines).

The strict form this originally accepted recovered 0 of 1920 real tile lines;
the model mostly echoes "#N: description" with no timestamp at all.  These tests
pin every observed shape so the regression cannot come back.
"""
import unittest

from tools.dsh import vlm_mosaic as M


class TestParseTileLines(unittest.TestCase):
    def test_all_observed_shapes(self):
        cases = {
            "tile 7 @ 00:05.4s :: hand on screwdriver": (7, "00:05.4", "hand on screwdriver"),
            "#7  00:05.4s: hand on screwdriver": (7, "00:05.4", "hand on screwdriver"),
            "#1: Both hands gripping a metal pin/rod at bottom center.":
                (1, "", "Both hands gripping a metal pin/rod at bottom center."),
            "#12  00:09.6s - both hands press small pack":
                (12, "00:09.6", "both hands press small pack"),
            "#3 00:02.1s left holds bracket, right inserts cylinder":
                (3, "00:02.1", "left holds bracket, right inserts cylinder"),
        }
        for line, (num, ts, desc) in cases.items():
            got = M.parse_tile_lines(line)
            self.assertIn(num, got, f"tile {num} not recovered from {line!r}")
            self.assertEqual(got[num][0], ts)
            self.assertEqual(got[num][1], desc)

    def test_trailing_json_is_not_a_tile(self):
        text = "#1: hands on part\n#2: hands on tool\n" \
               '{"usable": true, "reason_code": "OK", "explanation": "fine"}'
        got = M.parse_tile_lines(text)
        self.assertEqual(sorted(got), [1, 2])

    def test_twelve_tiles_of_a_full_sheet(self):
        text = "\n".join(f"#{i+1}: hands doing thing {i+1}" for i in range(12))
        self.assertEqual(len(M.parse_tile_lines(text)), 12)


if __name__ == "__main__":
    unittest.main()
