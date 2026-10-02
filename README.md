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
(see [Visualisation](#visualisation-classification-video)).

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

### Visualisation: classification video

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

## Original repositories and resources

| Repository | What it is used for here |
|---|---|
| [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics) | YOLO object detection (`yolov8l-oiv7.pt`, Open Images V7 classes) |
| [zhengxuJosh/360SFUDA](https://github.com/zhengxuJosh/360SFUDA) | Panoramic semantic segmentation (SegFormer-B2 weights `city_b2_52.99.pth`) |
| [NVlabs/SegFormer](https://github.com/NVlabs/SegFormer) | The segmentation architecture behind the 360SFUDA weights |
| [raquelgold/dfpe](https://github.com/raquelgold/dfpe) | Companion project: floor-plan estimation on the indoor clips |
