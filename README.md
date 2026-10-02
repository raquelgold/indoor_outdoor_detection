# Indoor/Outdoor Detection in 360° Video

Finds the frames where a **360° video** moves between **indoors and outdoors**.
Three independent methods each label every frame, and a transition is
confirmed only when at least 2 of them agree:

| Method | Signal | Source |
|---|---|---|
| `objects_detection` | indoor-like vs outdoor-like objects (chairs, doors, … vs trees, cars, …) | YOLO (Open Images V7) |
| `building_area` | a House/Building box filling > 98% of the frame | YOLO (Open Images V7) |
| `sfuda` | sky / vegetation / car pixel percentages | SegFormer panoramic segmentation (360SFUDA) |

The output (`transitions_confirmed.csv`, one per video) is the input of the
companion **floor-plan project**,
[raquelgold/dfpe](https://github.com/raquelgold/dfpe), which cuts the
indoor parts into clips and estimates a floor plan for each.

## Demo

![Indoor/outdoor classification on video 010](docs/demo_010.gif)

Video 010, about 13 s around entering and leaving a building. The top row shows
each method's per-frame decision, then `=` the **FINAL** decision. Boxes are
the YOLO detections the methods use: green = indoor-like objects, orange =
outdoor-like, blue = House/Building. Made with `visualize_classifications.py`
(see [Classification video](#classification-video)).

---

## Repository layout

```
pipeline_lock_direction.py     Step 1: YOLO detection + the two YOLO methods -> transitions.csv
merge_transitions.py           Step 3: combine all sources (>= 2 agree) -> transitions_confirmed.csv
visualize_classifications.py   video showing every method's per-frame label + FINAL
object_detection/              per-frame classifiers and transition debouncing
                               (+ standalone annotation tools)
extract_classes_yolo/          per-frame OIV7 class summaries
data/videos/{id}/              per video: transition_yolo.csv, transition_sfuda.csv,
                               transitions_confirmed.csv (23 Technion videos, 081–107)
docs/                          demo GIF
```

Step 2 (SegFormer) runs from the 360SFUDA project (see below).

---

## Setup

| What | Where | Notes |
|---|---|---|
| `~/venv` (Python 3.12) | everything in this repo | `ultralytics` 8.3.241, `opencv-python`, `pandas`, `numpy`, `tqdm`, `ffmpeg` on PATH (for the video) |
| YOLO OIV7 weights | `~/yolov8l-oiv7.pt` | from [Ultralytics](https://github.com/ultralytics/ultralytics); `--model` to change |
| 360SFUDA | `~/360SFUDA` (or `SFUDA_DIR`) | SegFormer step, and its decision rule is reused by the classification video |
| conda `sfuda` | runs 360SFUDA | Python 3.8, `torch` 1.12.1+cu113, `mmcv-full`, `timm`, `transformers` |
| SegFormer weights | `~/360SFUDA/pre_trained_weights/city_b2_52.99.pth` | SegFormer-B2 from [360SFUDA](https://github.com/zhengxuJosh/360SFUDA) |

---

## Running it on a video

```bash
VIDEO=~/lock_direction/new_lock_direction/VID_20260603_161558_00_097.mp4
ID=097
mkdir -p data/videos/$ID
```

### Step 1: YOLO → `transition_yolo.csv`

```bash
~/venv/bin/python pipeline_lock_direction.py --video "$VIDEO"
# or every video in a folder (default ~/lock_direction/new_lock_direction):
~/venv/bin/python pipeline_lock_direction.py --videos_dir path/to/folder
cp ~/pipeline_output/$(basename "$VIDEO" .mp4)/transitions.csv data/videos/$ID/transition_yolo.csv
```

Inference is skipped for any video whose `oiv7_detections.csv` already exists.
For each video it writes to `{--output_dir}/{video_stem}/` (default `~/pipeline_output/`):
- `oiv7_detections.csv`: raw per-frame detections
- `indoor_outdoor_classification.csv`: "building_area" labels
- `indoor_outdoor_objects_classification.csv`: "objects_detection" labels
- `transitions.csv`: `frame_id, direction, method`

### Step 2: SegFormer → `transition_sfuda.csv`

```bash
conda run -n sfuda python ~/360SFUDA/pipeline_master.py \
    --input_video "$VIDEO" \
    --output_base /mnt/sfuda_out/$ID \
    --weights     ~/360SFUDA/pre_trained_weights/city_b2_52.99.pth \
    --fps 30                      # the video's real frame rate (e.g. 24 for video 010)
cp /mnt/sfuda_out/$ID/transition_report.csv data/videos/$ID/transition_sfuda.csv
```

Output columns: `transition_frame, type, sky_pct, veg_pct, car_pct, range_start, range_end`.

### Step 3: Merge → `transitions_confirmed.csv`

```bash
~/venv/bin/python merge_transitions.py \
    --yolo_csv   data/videos/$ID/transition_yolo.csv \
    --sfuda_csv  data/videos/$ID/transition_sfuda.csv \
    --output_csv data/videos/$ID/transitions_confirmed.csv
```

Columns: `direction, frame_min, frame_max, sources, n_sources`. An empty file
(header only) means the sources never agreed, i.e. no confirmed transition.
This file is what the [floor-plan project](https://github.com/raquelgold/dfpe) needs.

### Classification video

Renders the video with a big INDOOR/OUTDOOR block per method, then `= FINAL`,
plus the YOLO boxes the methods use:

```bash
~/venv/bin/python visualize_classifications.py \
    --video          "$VIDEO" \
    --detections_dir ~/pipeline_output/$(basename "$VIDEO" .mp4) \
    --sfuda_csv      /mnt/sfuda_out/$ID/frame_object_stats.csv \
    --output         /mnt/sfuda_out/$ID/classifications_$ID.mp4
```

- `--only building`: just building_area and the House/Building boxes (no FINAL;
  `--sfuda_csv` not needed).
- `--only others`: objects_detection + SegFormer with the indoor/outdoor-like
  boxes. FINAL still uses all 3 sources.

---

## How each step decides

### `pipeline_lock_direction.py`

1. Loads `yolov8l-oiv7.pt` once and streams frames with `cv2.VideoCapture`.
2. Runs `model.predict(frame, conf=0.01, iou=0.7)`. The near-zero threshold is
   deliberate: the classifiers below do their own filtering.
3. Writes `oiv7_detections.csv`: `frame_id, class_id, class_name, conf, x1, y1, x2, y2, area_pct`
   (`area_pct` = bbox area / frame area × 100).
4. Calls `find_indoor_area` → `find_indoor_objects` → `detect_transitions`.

### `object_detection/find_indoor_area.py` ("building_area")

A frame is **indoor** iff some detection has `class_name` ∈ {House, Building},
`area_pct > 98.0` (`AREA_PCT_THRESHOLD`; the docstring's "90%" is stale) and
`conf >= 0.25`. It fires only when a wall fills almost the whole frame: high
precision, low recall. No temporal smoothing.

### `object_detection/find_indoor_objects.py` ("objects_detection")

Counts detections with `conf >= 0.1` against two sets:

- `INDOOR_LIKE`: Chair, Couch, Bed, Table, Toilet, Sink, Refrigerator, TV,
  Laptop, Microwave, Oven, Closet, Cabinetry, Bathtub, Shower, Mirror, Window,
  Door, Stairs, Furniture, …
- `OUTDOOR_LIKE`: Tree, Car, Street light, Stop sign, Traffic light/sign, Road,
  Bus, Truck, Bicycle.

More indoor objects → `indoor`; more outdoor → `outdoor`; a tie or no
qualifying objects → **the previous frame's label is carried forward** (frame 0
starts as `outdoor`).

### `object_detection/detect_transitions.py`

Runs over both label files. When the label changes at frame `i`, the next
`MIN_RUN = 10` frames must all hold the new label to confirm a transition
(`OUT_TO_IN` / `IN_TO_OUT`) at frame `i`.

### SegFormer (360SFUDA `suspicious_frames.py`)

Per frame: switches to indoor when sky, vegetation and cars are all low, and
back to outdoor when any is high (sky or vegetation > 15%, cars > 5%). Then
3-second smoothing removes short flickers.

### `merge_transitions.py`

- Chain-clusters same-direction events: an event joins a cluster if it is
  within `RADIUS = 100` frames of the cluster's last event.
- A cluster is confirmed if it has `>= MIN_SOURCES = 2` distinct sources.
  Keep this at 2: 3 silently drops real 2-source detections.
- `_resolve_overlaps()` collapses near-identical opposite-direction clusters
  (the same real crossing seen both ways): prefer the side sfuda voted for,
  then more sources, then the tighter frame span.

**FINAL** (in the classification video) comes from `compute_indoor_segments()`,
which turns the confirmed transitions into indoor frame ranges with
conservative bounds: indoors starts at the end of an `OUT_TO_IN` cluster,
ends just before the start of an `IN_TO_OUT` cluster, and a first `IN_TO_OUT`
means the video starts indoors. So FINAL is not a per-frame majority vote:
it switches only at confirmed transitions.

### Data flow

```
video ──YOLO(conf=0.01)──▶ oiv7_detections.csv
                              ├─▶ find_indoor_area    (>98% area, House/Building, conf≥0.25)
                              └─▶ find_indoor_objects (count-based, carry-forward on ties)
                                        both ─▶ detect_transitions (10-frame debounce) ─▶ transition_yolo.csv
video ──SegFormer (360SFUDA)─────────────────────────────────────────────────────────▶ transition_sfuda.csv

transition_yolo.csv + transition_sfuda.csv ─▶ merge_transitions (≥2 sources within 100 frames)
                                                    ─▶ transitions_confirmed.csv ─▶ floor-plan project
```

---

## Known limitations

- **SegFormer can read a covered ceiling as sky.** Under a canopy at a
  building entrance (video 010, frame 420) it reported 38% sky, so it stays
  OUTDOOR there.
- **YOLO results drift slightly between library/driver versions.** Re-running
  video 097 in September vs. July shifted its transitions by 3–16 frames.

---

## Original repositories and resources

| Repository | What it is used for here |
|---|---|
| [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics) | YOLO object detection (`yolov8l-oiv7.pt`, Open Images V7 classes) |
| [zhengxuJosh/360SFUDA](https://github.com/zhengxuJosh/360SFUDA) | Panoramic semantic segmentation (SegFormer-B2 weights `city_b2_52.99.pth`) |
| [NVlabs/SegFormer](https://github.com/NVlabs/SegFormer) | The segmentation architecture behind the 360SFUDA weights |
| [raquelgold/dfpe](https://github.com/raquelgold/dfpe) | Companion project: floor-plan estimation on the indoor clips |
