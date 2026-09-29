"""
Central path configuration for the DFPE pipeline.

Paths inside this repo are resolved relative to this file. External
dependencies (layout-model repos, SLAM binary, ORB vocabulary, venv) live
outside the repo; each can be overridden with the environment variable
named next to it.
"""

import os
from pathlib import Path

HOME = Path.home()

# ── Inside the repo ───────────────────────────────────────────────────────────
REPO_ROOT       = Path(__file__).resolve().parents[1]
PIPELINE_DIR    = REPO_ROOT / "dpfe_pipeline"
DFPE_ROOT       = REPO_ROOT / "direct_360_FPE"
SLAM_CONFIG_DIR = REPO_ROOT / "configs" / "stella_vslam"
VIDEOS_DIR      = REPO_ROOT / "data" / "videos"            # per-video {id}/ dirs
RESULTS_DEFAULT = REPO_ROOT / "outputs" / "dfpe_results"

# ── External dependencies ─────────────────────────────────────────────────────
HORIZONNET_DIR = Path(os.environ.get("HORIZONNET_DIR", HOME / "HorizonNet"))
LGTNET_DIR     = Path(os.environ.get("LGTNET_DIR", HOME / "LGT-Net"))
SLAM_BIN       = Path(os.environ.get(
    "STELLA_SLAM_BIN", HOME / "stella_vslam_examples" / "build" / "run_image_slam"))
ORB_VOCAB      = Path(os.environ.get("ORB_VOCAB", HOME / "orb_vocab.fbow"))
VENV_PY        = os.environ.get("DFPE_VENV_PY", str(HOME / "venv" / "bin" / "python"))
