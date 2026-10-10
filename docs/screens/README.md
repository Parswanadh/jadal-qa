# Screenshots and demo clip (docs/screens)

**Licensed data, private repo only.** The `poc_*` files in this directory show
real frames from the Egocentric-10K dataset (factory head-camera video). That
data is licensed and may not be redistributed. Keep this repository private, and
never copy these files into a public snapshot.

- `poc_*.png`: dashboard screens. The gallery and drawer show real Egocentric-10K
  thumbnails. The Overview, Live and About screens show text and charts only.
- `poc_demo_backup.mp4`: pre-rendered demo clip, rendered from real stream events
  and real footage. The blackout at t=12 s is scripted, not from the footage.
- Regenerate the PNGs from the current build (`scripts/poc_server.py` on a port
  you own, then headless Chrome with `--window-size` per file, see
  `docs/reports/poc_dashboard.md`). Do not commit new frames without updating
  `docs/PUBLIC_SNAPSHOT_EXCLUDE.txt`.

The other screenshots in this directory (`batches-*`, `queue-*`, `segment-*`, and
so on) are not Egocentric-10K content and are not listed in the exclude file.
