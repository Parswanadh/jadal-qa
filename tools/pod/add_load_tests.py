#!/usr/bin/env python3
"""Append a self-contained `load_tests` hook so bare module-level `test_*` functions are collected by
`python -m unittest discover` (the repo's canonical runner), including a minimal pytest `monkeypatch` shim.

usage: add_load_tests.py <test_file.py> [...]
"""
import os
import re
import sys

HOOK = '''

# --- pod: make bare module-level `test_*` functions visible to `unittest discover` ---
# `unittest` collects only TestCase subclasses; the functions above were written pytest-style and were
# therefore silently skipped by `python -m unittest discover -s tests`. This hook adapts them, including a
# minimal `monkeypatch` shim (only `setattr` is used) so no test is silently dropped.
class _PodMonkeyPatch:
    _MISSING = object()

    def __init__(self):
        self._undo = []

    def setattr(self, target, name=None, value=_MISSING, raising=True):
        if isinstance(target, str):  # pytest-style "module.attr" form
            import importlib

            mod, _, attr = target.rpartition(".")
            target, name, value = importlib.import_module(mod), attr, name
        old = getattr(target, name, self._MISSING)
        self._undo.append((target, name, old))
        setattr(target, name, value)

    def undo(self):
        while self._undo:
            target, name, old = self._undo.pop()
            if old is self._MISSING:
                try:
                    delattr(target, name)
                except AttributeError:
                    pass
            else:
                setattr(target, name, old)


def _pod_wrap(fn):
    def _test(self):
        import inspect
        import tempfile

        kwargs = {}
        params = inspect.signature(fn).parameters
        mp = _PodMonkeyPatch()
        _tmp = None
        if "monkeypatch" in params:
            kwargs["monkeypatch"] = mp
        if "tmp_path" in params:
            _tmp = tempfile.TemporaryDirectory()
            import pathlib

            kwargs["tmp_path"] = pathlib.Path(_tmp.name)
        try:
            fn(**kwargs)
        finally:
            mp.undo()
            if _tmp is not None:
                _tmp.cleanup()

    return _test


def load_tests(loader, tests, pattern):  # noqa: D103
    import inspect
    import unittest as _u

    class _ModuleTests(_u.TestCase):
        pass

    for _name, _fn in sorted(list(globals().items())):
        if not (_name.startswith("test_") and callable(_fn)):
            continue
        if {"monkeypatch", "tmp_path"} & set(inspect.signature(_fn).parameters):
            setattr(_ModuleTests, _name, _pod_wrap(_fn))
        else:
            setattr(_ModuleTests, _name, staticmethod(_fn))
    return _u.TestSuite([loader.loadTestsFromTestCase(_ModuleTests)])
'''


def main(argv):
    for path in argv:
        if not os.path.exists(path):
            print("missing", path)
            continue
        src = open(path).read()
        if "def load_tests(" in src:
            print("already has load_tests:", path)
            continue
        if not re.search(r"(?m)^def test_", src):
            print("no bare test_ functions:", path)
            continue
        open(path, "a").write(HOOK)
        print("patched", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
