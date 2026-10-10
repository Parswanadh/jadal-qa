"""E2 mosaic builder tests: determinism, timestamp placement/format, sidecar.

< 30 s, no network, no GPU.

Wrapped in ``unittest.TestCase`` classes (plus a ``__main__`` convenience
runner) so the repo's only test runner,
``python -m unittest discover -s tests``, collects them.
"""
import os
import shutil
import sys
import tempfile
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from jadal.mosaic import build as B  # noqa: E402


def _frames(n, h=64, w=64, seed=0):
    rng = np.random.default_rng(seed)
    return [rng.integers(0, 256, (h, w, 3), dtype=np.uint8) for _ in range(n)]


def _white_frames(n, h=40, w=40):
    return [np.full((h, w, 3), 255, np.uint8) for _ in range(n)]


class MosaicBuildTest(unittest.TestCase):
    """Base: a per-test scratch directory."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="mosaic_test_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _path(self, name):
        return os.path.join(self.tmp, name)


class TestDeterminism(MosaicBuildTest):
    def test_byte_identical_png(self):
        n = 9
        grid = B.grid_for_n(n)
        times = B.tile_times(n)
        idx = list(range(n))
        frames = _frames(n)
        img1, sc1 = B.build_mosaic(frames, times, idx, grid, 112,
                                   window_id="w", source="s", fps_used=0.9)
        img2, sc2 = B.build_mosaic(frames, times, idx, grid, 112,
                                   window_id="w", source="s", fps_used=0.9)
        p1 = self._path("a.png")
        p2 = self._path("b.png")
        s1 = B.save_mosaic(img1, sc1, p1, self._path("a.json"))
        s2 = B.save_mosaic(img2, sc2, p2, self._path("b.json"))
        with open(p1, "rb") as f1, open(p2, "rb") as f2:
            assert f1.read() == f2.read()
        assert s1["mosaic_sha256"] == s2["mosaic_sha256"]


class TestLayout(MosaicBuildTest):
    def test_grids_are_spec_grids(self):
        assert B.grid_for_n(4) == (2, 2)
        assert B.grid_for_n(6) == (2, 3)
        assert B.grid_for_n(9) == (3, 3)
        assert B.grid_for_n(12) == (3, 4)
        assert B.grid_for_n(16) == (4, 4)

    def test_gutter_and_canvas_size(self):
        n = 6
        grid = (2, 3)
        tile_px, g = 160, B.GUTTER
        img, _ = B.build_mosaic(_frames(n), B.tile_times(n), list(range(n)),
                                grid, tile_px, window_id="w", source="s",
                                fps_used=0.6)
        assert img.size == (3 * tile_px + 4 * g, 2 * tile_px + 3 * g)

    def test_aspect_letterbox(self):
        # non-square frame must be letterboxed with mid-grey bars
        f = np.full((30, 60, 3), 255, np.uint8)
        tile = B.letterbox_tile(f, 160)
        a = np.asarray(tile)
        assert a.shape == (160, 160, 3)
        assert a[0, 80].tolist() == list(B.BG)   # top bar mid-grey
        assert a[80, 80].tolist() == [255, 255, 255]  # centre white


class TestTimestamps(MosaicBuildTest):
    def test_times_uniform_and_in_window(self):
        t = B.tile_times(9)
        assert len(t) == 9
        assert all(0.0 < x < 10.0 for x in t)
        assert all(t[i] < t[i + 1] for i in range(8))
        assert abs(t[0] - 0.5 * 10 / 9) < 1e-9

    def test_time_spacing_is_window_over_n(self):
        # equal spacing: t[i+1] - t[i] == window / N for every i
        for n in (4, 6, 9, 12, 16):
            t = B.tile_times(n)
            for i in range(n - 1):
                assert abs((t[i + 1] - t[i]) - B.WINDOW_S / n) < 1e-9

    def test_rendered_string_format(self):
        # the burned-in label format is part of the spec: t=X.Xs
        assert B.ts_text(3.5) == "t=3.5s"
        assert B.ts_text(0.5) == "t=0.5s"
        assert B.ts_text(3.456) == "t=3.5s"  # one decimal only

    def test_timestamp_top_right_of_every_tile(self):
        """Each tile's top-right must contain the dark timestamp box."""
        n = 4
        tile_px = 112
        grid = (2, 2)
        times = B.tile_times(n)
        # uniform white frames so the only dark pixels are the timestamp boxes
        frames = _white_frames(n)
        img, sc = B.build_mosaic(frames, times, list(range(n)), grid, tile_px,
                                 window_id="w", source="s", fps_used=0.4)
        a = np.asarray(img)
        for tile in sc["tiles"]:
            y0, x0 = tile["y"], tile["x"]
            # sample the top-right 24x18 patch of the tile, must have dark pixels
            patch = a[y0:y0 + 18, x0 + tile_px - 24:x0 + tile_px]
            assert patch.min() < 150, f"no dark timestamp box in tile {tile['index']}"
        # and the top-LEFT of a tile must stay white (box is not full-width)
        t0 = sc["tiles"][0]
        left = a[t0["y"]:t0["y"] + 18, t0["x"]:t0["x"] + 18]
        assert left.min() > 200

    def test_timestamp_text_is_white_on_dark(self):
        """White text pixels sit on a dark (semi-transparent) box."""
        n = 4
        tile_px = 112
        img, sc = B.build_mosaic(
            _white_frames(n), B.tile_times(n), list(range(n)), (2, 2), tile_px,
            window_id="w", source="s", fps_used=0.4)
        a = np.asarray(img)
        for tile in sc["tiles"]:
            y0, x0 = tile["y"], tile["x"]
            patch = a[y0:y0 + 18, x0 + tile_px - 26:x0 + tile_px]
            assert patch.min() < 150, "timestamp box not dark"
            assert patch.max() >= 250, "timestamp text not white"

    def test_top_left_tile_has_earliest_time(self):
        n = 9
        img, sc = B.build_mosaic(_frames(n), B.tile_times(n), list(range(n)),
                                 B.grid_for_n(n), 224, window_id="w",
                                 source="s", fps_used=0.9)
        tiles = sc["tiles"]
        # slot 0 is placed top-left and carries the smallest time
        tl = min(tiles, key=lambda t: (t["y"], t["x"]))
        assert tl["x"] == B.GUTTER and tl["y"] == B.GUTTER
        assert tl["t_s"] == min(t["t_s"] for t in tiles)
        # and the tile at 224 px was really built at 224 px
        assert sc["tile_px"] == 224
        assert img.size == (3 * 224 + 4 * B.GUTTER, 3 * 224 + 4 * B.GUTTER)

    def test_no_timestamps_variant_keeps_tiles_white(self):
        n = 4
        frames = _white_frames(n)
        img, _ = B.build_mosaic(frames, B.tile_times(n), list(range(n)), (2, 2),
                                112, window_id="w", source="s", fps_used=0.4,
                                timestamps=False)
        a = np.asarray(img)
        # no dark pixels inside the image area except gutters (mid-grey 128)
        assert a.min() >= 128


class TestSidecar(MosaicBuildTest):
    def test_sidecar_schema(self):
        n = 9
        img, sc = B.build_mosaic(_frames(n), B.tile_times(n), list(range(n)),
                                 (3, 3), 160, window_id="w7", source="indego",
                                 fps_used=0.9)
        assert set(sc.keys()) == {"window_id", "source", "grid", "tile_px",
                                  "fps_used", "tiles", "mosaic_sha256",
                                  "builder_version", "font"}
        assert sc["window_id"] == "w7"
        assert sc["grid"] == [3, 3]
        assert sc["tile_px"] == 160
        assert len(sc["tiles"]) == 9
        for t in sc["tiles"]:
            assert set(t.keys()) == {"index", "t_s", "src_frame_index", "x", "y",
                                     "w", "h"}
            assert t["w"] == 160 and t["h"] == 160
        # row-major placement: slot0 top-left, slot8 bottom-right
        assert sc["tiles"][0]["x"] == B.GUTTER and sc["tiles"][0]["y"] == B.GUTTER
        assert sc["tiles"][8]["x"] > sc["tiles"][0]["x"]
        assert sc["tiles"][8]["y"] > sc["tiles"][0]["y"]

    def test_sidecar_records_font_for_hash_explainability(self):
        _, sc = B.build_mosaic(_frames(4), B.tile_times(4), list(range(4)),
                               (2, 2), 112, window_id="w", source="s",
                               fps_used=0.4)
        assert sc["font"] == {"file": B.font_file(), "size": B.FONT_SIZE}
        assert isinstance(sc["font"]["file"], str)


class TestVariants(MosaicBuildTest):
    def test_shuffled_order_permutation_keeps_timestamps(self):
        n = 9
        frames = _frames(n, seed=3)
        timestamps = True
        img, sc = B.build_window("w", "s", frames, 10.0, n_tiles=n, tile_px=112,
                                 shuffle_seed=1, timestamps=timestamps)
        slots = [(t["y"], t["x"]) for t in sc["tiles"]]
        assert len(set(slots)) == n
        assert sorted(t["index"] for t in sc["tiles"]) == list(range(n))
        # timestamps still recorded per tile time
        assert sc["tiles"][0]["t_s"] != sc["tiles"][1]["t_s"]

    def test_motion_variant_zero_diff_for_static(self):
        n = 4
        frames = [np.full((32, 32, 3), 200, np.uint8) for _ in range(n)]
        img, sc = B.build_window("w", "s", frames, 10.0, n_tiles=n, tile_px=112,
                                 motion=True)
        a = np.asarray(img)
        # only gutters (128) and timestamps (dark) differ from the diff tiles (0)
        for t in sc["tiles"]:
            patch = a[t["y"]:t["y"] + 112, t["x"]:t["x"] + 112]
            assert patch.min() == 0


def load_tests(loader, tests, pattern):
    """Explicit suite so discovery always collects the classes above."""
    suite = unittest.TestSuite()
    for cls in (TestDeterminism, TestLayout, TestTimestamps, TestSidecar,
                TestVariants):
        suite.addTests(loader.loadTestsFromTestCase(cls))
    return suite


if __name__ == "__main__":
    unittest.main()
