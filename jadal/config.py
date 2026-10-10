"""Shared configuration defaults for Jadal QA. READ-ONLY for module agents.

Everything here is a default; the service may override values from environment
variables or a JSON file. Nothing in this module touches the network.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNTIME = ROOT / "data_runtime"

LANGS = ["hi-IN", "kn-IN", "ta-IN", "te-IN", "en-IN"]

REASON_CODES = [
    "UNDECODABLE", "HANDS_NOT_VISIBLE", "BLUR", "TOO_DARK", "OVEREXPOSED",
    "SHAKY", "DUPLICATE", "WRONG_STEP", "NO_ACTIVE_WORK", "LOW_CONFIDENCE",
]


@dataclass
class Config:
    batch_size: int = 5            # N segments per batch
    batch_reject_k: int = 1        # reject the batch if rejected_count > k (fixed rule, no guarantee)
    segment_seconds: float = 20.0
    seed: int = 1234
    float_decimals: int = 4
    # L0 thresholds (placeholders until calibrated on our own labelled clips)
    blur_var_min: float = 60.0
    brightness_min: float = 40.0
    over_exposed_frac_max: float = 0.35
    motion_max: float = 12.0
    hands_visible_frac_min: float = 0.5
    # L2
    target_error: float = 0.05
    delta: float = 0.1
    min_accepted: int = 30
    # paths
    bucket_dir: Path = field(default_factory=lambda: RUNTIME / "bucket")
    segments_dir: Path = field(default_factory=lambda: RUNTIME / "segments")
    blurred_dir: Path = field(default_factory=lambda: RUNTIME / "blurred")
    db_path: Path = field(default_factory=lambda: RUNTIME / "jadal.db")
    voicebank_dir: Path = field(default_factory=lambda: ROOT / "voicebank")
    sop_dir: Path = field(default_factory=lambda: ROOT / "sop")
    web_dir: Path = field(default_factory=lambda: ROOT / "web")
    pico_port: str = field(default_factory=lambda: os.environ.get("JADAL_PICO_PORT", "/dev/ttyACM0"))
    pico_baud: int = 115200
    pipeline_version: str = "0.1.0"


DEFAULT = Config()
