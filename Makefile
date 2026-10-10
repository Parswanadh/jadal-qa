PY := /home/parshu/envs/jadal/bin/python

.PHONY: demo test service

# Full end-to-end demo on the IndEgo clips (GitHub issue #17, step 4):
# service -> dock -> batches -> verdict table -> reverify every receipt ->
# 10x determinism assert -> batch latency. See scripts/demo.sh.
demo:
	bash scripts/demo.sh

# All unit/integration tests (stubs; no hardware, no network).
# Each suite must finish well under the 120 s timeout.
test:
	timeout 120 env PYTHONNOUSERSITE=1 $(PY) -m unittest discover -s tests
	timeout 120 env PYTHONNOUSERSITE=1 $(PY) -m unittest discover -s web

# Run the service locally (stubs unless the real modules merged).
service:
	PYTHONNOUSERSITE=1 JADAL_USE_STUBS=1 $(PY) -m uvicorn jadal.service:app
