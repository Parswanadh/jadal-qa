"""Default-path resolution for the PoC scripts (poc_run.py, poc_server.py, poc_up.sh).

Order, first match wins:
  data dir : $JADAL_POC_DATA, then <ROOT>/../poc_data, then <ROOT>/poc_data
  clip dir : $JADAL_CLIP_ROOT, then <ROOT>/../../awing/data/ego10k_scale/clips,
             then <ROOT>/../awing/data/ego10k_scale/clips

A candidate counts only if it EXISTS. If the env var is set but does not exist,
that is an error (an explicit override is never silently replaced). If nothing
resolves, PathError names the env var to set. There is no fallback to the demo
fixture here; that is only used when a caller passes --demo-fixture explicitly.

CLI (used by poc_up.sh):  python3 scripts/poc_paths.py data|clips
prints the resolved directory, or a message on stderr and exits 1.
"""
import os
import sys

DATA_ENV = "JADAL_POC_DATA"
CLIP_ENV = "JADAL_CLIP_ROOT"


class PathError(RuntimeError):
    pass


def data_candidates(root):
    return [os.path.normpath(os.path.join(root, "..", "poc_data")),
            os.path.normpath(os.path.join(root, "poc_data"))]


def clip_candidates(root):
    return [os.path.normpath(os.path.join(root, "..", "..", "awing", "data", "ego10k_scale", "clips")),
            os.path.normpath(os.path.join(root, "..", "awing", "data", "ego10k_scale", "clips"))]


def _resolve(kind, env_name, candidates, env):
    explicit = env.get(env_name)
    if explicit:
        if os.path.isdir(explicit):
            return os.path.normpath(explicit)
        raise PathError(f"{env_name}={explicit!r} does not exist (set it to an existing {kind} dir)")
    for c in candidates:
        if os.path.isdir(c):
            return c
    tried = ", ".join(candidates)
    raise PathError(f"no {kind} dir found (tried {tried}); set {env_name} to the {kind} dir")


def resolve_poc_data(root, env=None):
    return _resolve("PoC data", DATA_ENV, data_candidates(root),
                    os.environ if env is None else env)


def resolve_clip_root(root, env=None):
    return _resolve("Egocentric-10K clip", CLIP_ENV, clip_candidates(root),
                    os.environ if env is None else env)


def main(argv):
    root = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    if len(argv) != 2 or argv[1] not in ("data", "clips"):
        print("usage: poc_paths.py data|clips", file=sys.stderr)
        return 2
    try:
        path = resolve_poc_data(root) if argv[1] == "data" else resolve_clip_root(root)
    except PathError as e:
        print(f"poc_paths: {e}", file=sys.stderr)
        return 1
    print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
