# Indoor Floor-Plan Pipeline (DFPE)

How to go from a **raw 360° video** to an **estimated floor plan**, using
either **HorizonNet** or **LGT-Net** as the layout estimator.

Two independent detectors find indoor/outdoor transitions in the video
(YOLO object cues and SegFormer segmentation). The pipeline merges them,
confirms a transition only when at least 2 sources agree, cuts the video
into indoor clips, and runs floor-plan estimation (direct_360_FPE) on each
clip.

---

### `direct_360_FPE` provenance

A copy of [EnriqueSolarte/direct_360_FPE](https://github.com/EnriqueSolarte/direct_360_FPE)
at commit `6a41911`, with local changes to 16 files (scale recovery, room-shape
solver, the `*_coords.json` export in `utils/eval_utils.py`, config). The full
diff against upstream is in [direct_360_FPE/LOCAL_CHANGES.patch](direct_360_FPE/LOCAL_CHANGES.patch).
Upstream's `mp3d_fpe_dataset/` helper folder was removed locally (the dataset
itself is on [HuggingFace](https://huggingface.co/datasets/EnriqueSolarte/mp3d_fpe)).

---

## 0. One-time setup

### External dependencies

These are not in the repo. Default locations are under `$HOME`; each can be
overridden with the environment variable shown (read by `dpfe_paths.py`).

| Dependency | Default location | Env var | Source |
|---|---|---|---|
| HorizonNet | `~/HorizonNet` | `HORIZONNET_DIR` | [sunset1995/HorizonNet](https://github.com/sunset1995/HorizonNet) @ `c9a7df9` |
| LGT-Net | `~/LGT-Net` | `LGTNET_DIR` | [zhigangjiang/LGT-Net](https://github.com/zhigangjiang/LGT-Net) @ `0045359` |
| stella_vslam (library) | installed to `/usr/local/lib` | – | [stella-cv/stella_vslam](https://github.com/stella-cv/stella_vslam) @ `8ac1be4` |
| stella_vslam_examples `run_image_slam` | `~/stella_vslam_examples/build/run_image_slam` | `STELLA_SLAM_BIN` | [stella-cv/stella_vslam_examples](https://github.com/stella-cv/stella_vslam_examples) @ `defc69e` |
| ORB vocabulary | `~/orb_vocab.fbow` | `ORB_VOCAB` | shipped with stella_vslam |
| venv interpreter | `~/venv/bin/python` | `DFPE_VENV_PY` | see below |

### Model checkpoints

| Model | Path |
|---|---|
| HorizonNet (mp3d) | `~/HorizonNet/ckpt/resnet50_rnn__mp3d.pth` |
| LGT-Net (mp3d) | `~/LGT-Net/checkpoints/SWG_Transformer_LGT_Net/mp3d/best.pkl` |
| YOLO OIV7 | `~/yolov8l-oiv7.pt` |

### Python environments

| Env | Used by | Key packages |
|---|---|---|
| `~/venv` (Python 3.12) | `transition_detection/`, all `dpfe_pipeline/` glue scripts | `ultralytics` 8.3.241, `opencv-python`, `pandas`, `numpy`, `scipy`, `matplotlib`, `tqdm` |
| conda `dfpe` | `run_horizonnet.py`, `run_dfpe.py` → `direct_360_FPE/main_eval_scene.py` | `torch` 2.4.1+cu118, HorizonNet deps, `direct_360_FPE/requirements.txt` |
| conda `lgtnet` | `run_lgtnet.py` (imports LGT-Net source) | `torch==1.7.1`, `LGT-Net/requirements.txt` |
| conda `sfuda` | `~/360SFUDA/pipeline_master.py` | Python 3.8, `torch` 1.12.1+cu113, `mmcv-full`, `timm`, `transformers` |

```bash
# dfpe (HorizonNet + direct_360_FPE)
conda create -n dfpe python=3.9 -y && conda activate dfpe
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install -r direct_360_FPE/requirements.txt
pip install -e direct_360_FPE

# lgtnet
conda create -n lgtnet python=3.8 -y && conda activate lgtnet
pip install -r ~/LGT-Net/requirements.txt
```

### SLAM binary

Build and install stella_vslam first (see its README), then:

```bash
cd ~/stella_vslam_examples
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j"$(nproc)"
```

**The SLAM camera config's `cols`/`rows` must equal the extracted frames' pixel
size exactly.** `extract_indoor_clips.py` writes frames at the video's native
resolution, so the config must match the source video. `run_slam.py` picks a
config from `configs/stella_vslam/` by resolution and fps.


For a new resolution/fps, copy `config_fps30_5760.yaml`, set `Camera.cols`,
`Camera.rows` and `Camera.fps` (check with `ffprobe`), and pass it with
`run_slam.py --config`. The video **must be equirectangular (360°)**.

---

## 1. Quick start

All commands run from the repo root.

```bash
VIDEO=~/lock_direction/new_lock_direction/VID_20260603_161558_00_097.mp4
ID=097

# Stage A: YOLO transitions -> ~/pipeline_output/{video_stem}/transitions.csv
~/venv/bin/python transition_detection/pipeline_lock_direction.py --video "$VIDEO"
mkdir -p data/videos/$ID
cp ~/pipeline_output/$(basename "$VIDEO" .mp4)/transitions.csv data/videos/$ID/transition_yolo.csv

# Stage B: SegFormer transitions (see step 3 below) -> data/videos/$ID/transition_sfuda.csv

# Stage C: floor plans
~/venv/bin/python dpfe_pipeline/pipeline.py --video "$VIDEO" --layout_model horizonnet
# LGT-Net instead:
~/venv/bin/python dpfe_pipeline/pipeline.py --video "$VIDEO" --layout_model lgtnet \
    --results_root outputs/dfpe_results_lgt
```

`--video_dir` defaults to `data/videos/{ID}`. Use a separate `--results_root`
per layout model: both variants share the same clips, so their DFPE outputs
land in the same `{id}_{n}` folders otherwise.

Results land in `outputs/dfpe_results/{ID}_{clip_index}/`, one folder per
indoor clip.

---

## 2. Step by step for a new video

### Step 1: Drop the video in

### Step 2: `transition_yolo.csv`

```bash
~/venv/bin/python transition_detection/pipeline_lock_direction.py --video path/to/video.mp4
# or every video in a folder (default ~/lock_direction/new_lock_direction):
~/venv/bin/python transition_detection/pipeline_lock_direction.py --videos_dir path/to/folder
```
For each video it writes to
`{--output_dir}/{video_stem}/` (default `~/pipeline_output/`):
- `oiv7_detections.csv`: raw per-frame detections
- `indoor_outdoor_classification.csv`: "building_area" method
- `indoor_outdoor_objects_classification.csv`: "objects_detection" method
- `transitions.csv`: columns `frame_id, direction, method`

Copy `transitions.csv` to `data/videos/{ID}/transition_yolo.csv`.

### Step 3: `transition_sfuda.csv`

```bash
conda run -n sfuda python ~/360SFUDA/pipeline_master.py \
    --input_video  ~/lock_direction/new_lock_direction/VID_..._097.mp4 \
    --output_base  ~/sfuda_out/097 \
    --weights      ~/360SFUDA/pre_trained_weights/city_b2_52.99.pth \
    --fps 30
cp ~/sfuda_out/097/transition_report.csv data/videos/097/transition_sfuda.csv
```

Output columns: `transition_frame, type, sky_pct, veg_pct, car_pct, range_start, range_end`.

### Step 4: Run the pipeline

```bash
~/venv/bin/python dpfe_pipeline/pipeline.py --video ~/lock_direction/new_lock_direction/VID_..._097.mp4 --layout_model horizonnet
```

| Stage | horizonnet | lgtnet |
|---|---|---|
| 1 | merge_transitions | merge_transitions |
| 2 | extract_indoor_clips | extract_indoor_clips |
| 3 | HorizonNet (all frames) | SLAM |
| 4 | SLAM | setup_scene |
| 5 | setup_scene | LGT-Net (keyframes only) |
| 6 | DFPE | DFPE |
| 7 | generate_maps | generate_maps |

`--from_stage N` / `--to_stage N` run a sub-range.

### Step 5: Visualisations

```bash
# Top-down GIF of the map being built
~/venv/bin/python dpfe_pipeline/generate_map_animation.py --results_dir outputs/dfpe_results/097_0

# Extra "unified" floor plan (all walls, no room separation)
~/venv/bin/python dpfe_pipeline/generate_maps.py --results_dir outputs/dfpe_results/097_0 --unified

# Side-by-side video: original + indoor/outdoor banner | SLAM GIF / final map
~/venv/bin/python dpfe_pipeline/visualize_video_map_animated.py \
    --video ~/lock_direction/new_lock_direction/VID_..._097.mp4 --video_dir data/videos/097 \
    --output outputs/097_indoor_map_viz.mp4
```

## 6. Experiments

Not part of the pipeline. They are kept for reference and depend on run data
that is not in the repo.

- `experiments/slam_scale_study/`: SLAM trajectory vs. MP3D-FPE ground truth
  on scene `1LXtFkjw3qL` (scale ratios, run-to-run comparisons). Reads
  `direct_360_FPE/slam_output/` and `direct_360_FPE/mp3d_fpe_dataset/`.
- `experiments/room_overlap_clipping/clip_room_overlaps.py`: removes overlaps
  between DFPE room polygons with Shapely (tried on video033/035 with LGT-Net).
