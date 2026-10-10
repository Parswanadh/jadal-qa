#!/usr/bin/env python3
"""Append a `load_tests` hook so bare module-level `test_*` functions are collected by
`python -m unittest discover` (the repo's canonical runner). Idempotent.

usage: add_load_tests.py <test_file.py> [...]
"""
import inspect  # noqa: F401  (used inside the generated hook)
import os
import re
import sys

HOOK = '''

# --- pod: make bare module-level `test_*` functions visible to `unittest discover` ---
# `unittest` collects only TestCase subclasses; the functions above were written pytest-style and were
# therefore silently skipped by `python -m unittest discover -s tests`. This hook adapts them.
def load_tests(loader, tests, pattern):  # noqa: D103
    import unittest as _u

    class _Missing:
        pass

    class _MonkeyPatch:
        """Minimal pytest `monkeypatch` shim: setattr() only, undone after each test."""

        def __init__(self):
            self._undo = []

        def setattr(self, target, name, value=None, raising=True):
            if isinstance(target, str):  # "module.attr" form
                mod, _, attr = target.rpartition(".")
                target, name, value = __import__(mod, fromlist=[attr]), attr, name
            old = getattr(target, name, _Missing)
            self._undo.append((target, name, old))
            setattr(target, name, value)

        def undo(self):
            while self._undo:
                target, name, old = self._undo.pop()
                if old is _Missing:
                    delattr(target, name)
                else:
                    setattr(target, name, old)

    class _ModuleTests(_u.TestCase):
        pass

    for _name, _fn in sorted(list(globals().items())):
        if not (_name.startswith("test_") and callable(_fn)):
            continue
        if "monkeypatch" in inspect.signature(_fn).parameters:

            def _wrap(fn):
                def _test(self):
                    mp = _MonkeyPatch()
                    try:
                        fn(monkeypatch=mp)
                    finally:
                        mp.undo()

                return _test

            setattr(_ModuleTests, _name, _wrap(_fn))
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
