"""E2 mosaic builder tests: determinism, timestamp placement, sidecar schema.

< 30 s, no network, no GPU.
"""
import json
import os
import sys

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from jadal.mosaic import build as B  # noqa: E402


def _frames(n, h=64, w=64, seed=0):
    rng = np.random.default_rng(seed)
    return [rng.integers(0, 256, (h, w, 3), dtype=np.uint8) for _ in range(n)]


def test_byte_identical_png(tmp_path):
    n = 9
    grid = B.grid_for_n(n)
    times = B.tile_times(n)
    idx = list(range(n))
    frames = _frames(n)
    img1, sc1 = B.build_mosaic(frames, times, idx, grid, 112,
                               window_id="w", source="s", fps_used=0.9)
    img2, sc2 = B.build_mosaic(frames, times, idx, grid, 112,
                               window_id="w", source="s", fps_used=0.9)
    p1 = str(tmp_path / "a.png")
    p2 = str(tmp_path / "b.png")
    s1 = B.save_mosaic(img1, sc1, p1, str(tmp_path / "a.json"))
    s2 = B.save_mosaic(img2, sc2, p2, str(tmp_path / "b.json"))
    assert open(p1, "rb").read() == open(p2, "rb").read()
    assert s1["mosaic_sha256"] == s2["mosaic_sha256"]


def test_grids_are_spec_grids():
    assert B.grid_for_n(4) == (2, 2)
    assert B.grid_for_n(6) == (2, 3)
    assert B.grid_for_n(9) == (3, 3)
    assert B.grid_for_n(12) == (3, 4)
    assert B.grid_for_n(16) == (4, 4)


def test_times_uniform_and_in_window():
    t = B.tile_times(9)
    assert len(t) == 9
    assert all(0.0 < x < 10.0 for x in t)
    assert all(t[i] < t[i + 1] for i in range(8))
    assert abs(t[0] - 0.5 * 10 / 9) < 1e-9


def test_timestamp_top_right_of_every_tile():
    """Each tile's top-right must contain the dark timestamp box."""
    n = 4
    tile_px = 112
    grid = (2, 2)
    times = B.tile_times(n)
    # uniform white frames so the only dark pixels are the timestamp boxes
    frames = [np.full((40, 40, 3), 255, np.uint8) for _ in range(n)]
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


def test_no_timestamps_variant_keeps_tiles_white():
    n = 4
    frames = [np.full((40, 40, 3), 255, np.uint8) for _ in range(n)]
    img, _ = B.build_mosaic(frames, B.tile_times(n), list(range(n)), (2, 2),
                            112, window_id="w", source="s", fps_used=0.4,
                            timestamps=False)
    a = np.asarray(img)
    # no dark pixels inside the image area except gutters (mid-grey 128)
    assert a.min() >= 128


def test_gutter_and_canvas_size():
    n = 6
    grid = (2, 3)
    tile_px, g = 160, B.GUTTER
    img, _ = B.build_mosaic(_frames(n), B.tile_times(n), list(range(n)),
                            grid, tile_px, window_id="w", source="s",
                            fps_used=0.6)
    assert img.size == (3 * tile_px + 4 * g, 2 * tile_px + 3 * g)


def test_sidecar_schema():
    n = 9
    img, sc = B.build_mosaic(_frames(n), B.tile_times(n), list(range(n)),
                             (3, 3), 160, window_id="w7", source="indego",
                             fps_used=0.9)
    assert set(sc.keys()) == {"window_id", "source", "grid", "tile_px",
                              "fps_used", "tiles", "mosaic_sha256",
                              "builder_version"}
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


def test_shuffled_order_permutation_keeps_timestamps():
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


def test_motion_variant_zero_diff_for_static():
    n = 4
    frames = [np.full((32, 32, 3), 200, np.uint8) for _ in range(n)]
    img, sc = B.build_window("w", "s", frames, 10.0, n_tiles=n, tile_px=112,
                             motion=True)
    a = np.asarray(img)
    # only gutters (128) and timestamps (dark) differ from the diff tiles (0)
    for t in sc["tiles"]:
        patch = a[t["y"]:t["y"] + 112, t["x"]:t["x"] + 112]
        assert patch.min() == 0


def test_aspect_letterbox():
    # non-square frame must be letterboxed with mid-grey bars
    f = np.full((30, 60, 3), 255, np.uint8)
    tile = B.letterbox_tile(f, 160)
    a = np.asarray(tile)
    assert a.shape == (160, 160, 3)
    assert a[0, 80].tolist() == list(B.BG)   # top bar mid-grey
    assert a[80, 80].tolist() == [255, 255, 255]  # centre white


if __name__ == "__main__":
    import pathlib
    import tempfile
    tmp = pathlib.Path(tempfile.mkdtemp())
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            try:
                fn(tmp)
            except TypeError:
                fn()
            print("ok", name)
    print("all tests passed")
