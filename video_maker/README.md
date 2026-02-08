# Contact Timeline Video Maker

This folder contains a project-specific renderer for generating contact-timeline videos from this repository's pipeline outputs.

## What it renders

For each frame, the output video stacks:
1. Raw frame
2. Header (`title`, frame number, position in sequence)
3. Timeline rows with a moving playhead:
   - `Pred` (required)
   - `GT` (optional, when `--gt-csv` is provided)
   - `Pred 2` (optional, when `--secondary-condensed-csv` is provided)

Binary mapping follows existing repo conventions:
- `Portable Object` -> `1` (`holding`)
- everything else -> `0` (`not holding`)

## CLI usage

Run from repository root:

```bash
python scripts/make_contact_timeline_video.py \
  --condensed-csv /path/to/detections_condensed.csv \
  --image-dir /path/to/frames \
  --gt-csv /path/to/gt.csv
```

Optional second prediction track:

```bash
python scripts/make_contact_timeline_video.py \
  --condensed-csv /path/to/new_model/detections_condensed.csv \
  --secondary-condensed-csv /path/to/baseline/detections_condensed.csv \
  --image-dir /path/to/frames
```

Default output path:

```text
<condensed_csv_parent>/visualizations/contact_timeline.mp4
```

## Programmatic usage

```python
from video_maker.contact_timeline_renderer import (
    load_pred_binary_from_condensed,
    load_gt_binary_aligned,
    render_contact_timeline_video,
)

frame_numbers, pred_binary = load_pred_binary_from_condensed("detections_condensed.csv")
gt_binary = load_gt_binary_aligned("gt.csv", frame_numbers)

render_contact_timeline_video(
    image_dir="frames",
    frame_numbers=frame_numbers,
    pred_binary=pred_binary,
    gt_binary=gt_binary,
    output_video_path="visualizations/contact_timeline.mp4",
)
```
