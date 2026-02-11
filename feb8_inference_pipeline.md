# Feb 8 Inference Pipeline: Exact Behavior, Condensing Rules, and Proposed Strict Spec

## 1) Why this document exists

This document is a code-grounded explanation of the current refactored inference pipeline in this repository, focused on:

- What the pipeline actually does today.
- How `detections_condensed.csv` is produced from `detections_full.csv`.
- Why you can see apparently inconsistent outputs (for example, `Portable Object` without visible object box).
- A proposed strict semantics spec if you want behavior to match the intuitive rule: "portable means hand touching a detected object (and not blue-glove experimenter hand)."

This is written as future-session context so a new Codex session can reason about runs without needing prior chat history.

## 2) High-level architecture (refactored pipeline)

### Main entrypoints

- CLI: `run_pipeline.py`
- Core orchestration: `pipeline/main.py`
- Detection inference: `pipeline/inference.py`
- Post-detection filters + condensing: `pipeline/filters.py`, `pipeline/postprocessing.py`
- Visual outputs: `pipeline/visualization.py`
- Timeline video composition: `scripts/make_contact_timeline_video.py`, `video_maker/contact_timeline_renderer.py`
- Batch wrappers:
  - `scripts/run_shrunk_inference_batch.py`
  - `scripts/run_shrunk_full_workflow.py`
  - `scripts/render_timeline_videos_from_manifest.py`

### Core data flow

1. Input frames/video are resolved and optionally preprocessed (`prepare_frames`).
2. Detector outputs per-frame hand rows and object rows (`run_on_directory`).
3. Filters mutate labels and/or mark rows filtered (`apply_detection_filters`).
4. Condensing creates one row per frame (`condense_dataframe`).
5. Optional outputs are written:
   - `detections_full.csv`
   - `detections_condensed.csv`
   - barcodes
   - `frames_det` annotated frames
   - downstream timeline videos (via separate scripts)

## 3) Exact inference and label assignment behavior (current implementation)

## 3.1 Detector output semantics

In `pipeline/inference.py`, hand detections carry a model-predicted contact state:

- `0 -> No Contact`
- `1 -> Self Contact`
- `2 -> Other Person Contact`
- `3 -> Portable Object`
- `4 -> Stationary Object Contact`

Important: this contact class is produced from the hand branch and does **not** require an object detection row to exist in the same frame.

Objects are emitted as separate rows (`detection_type == object`) with `contact_label = None`.

## 3.2 Blue glove logic

Runtime flow:

- `run_inference` calls detector with `blue_glove_filter=False` at raw inference stage.
- Blue-glove refinement is then applied in postprocessing if enabled (`apply_blue_glove_filter`).

What blue filter does:

- For each unfiltered hand row, compute blue-pixel proportion in hand bbox.
- If `blue_prop >= blue_threshold`, relabel hand row to:
  - `contact_label = No Contact`
  - `contact_state = 0`
  - `blue_glove_status = experimenter`
- Else mark `blue_glove_status = participant`.

It does not require object evidence and it does not remove rows; it relabels hand rows.

## 3.3 Object-related filters

### Object size filter

`apply_object_size_filter` marks oversized object rows as filtered (`is_filtered=True`) based on area/frame ratio.

- This affects object availability for subsequent matching filters.
- This does **not** directly relabel hand rows.

### Object/hand ratio filters

`apply_obj_bigger_than_hand_filter` and `apply_obj_smaller_than_hand_filter`:

- Only evaluate rows where hand label is already `Portable Object`.
- Attempt to match each portable hand to an object in the same frame (unfiltered object rows only).
- If a ratio rule triggers, relabel hand row to `No Contact`.

Critical behavior:

- If no object row is available/matched, these filters do nothing.
- Therefore portable hand labels can persist even when object detections are missing.

## 3.4 Optional tracking bridge behavior

If `--tracking-bridge` is enabled:

- `object_tracking/bridge.py` may promote a hand from `No Contact` to `Portable Object` based on temporal object track continuity and IoU confirmation logic.
- This is an additional source of `Portable Object` labels not strictly tied to same-frame visible object box.

## 4) Condensing rules (`detections_condensed.csv`)

`pipeline/postprocessing.py::condense_dataframe` makes one label per frame using only unfiltered hand rows.

### 4.1 Input subset used for condensing

For each frame:

- Take rows where `detection_type == hand` and `is_filtered == False`.
- Ignore object rows when choosing frame-level contact label.

### 4.2 Label normalization and priority

Hand labels are normalized to:

- `Portable Object`
- `Stationary Object`
- `No Contact`

Then one label is selected by priority:

1. `Portable Object` (highest)
2. `Stationary Object`
3. `No Contact` (lowest)

Implications:

- If any unfiltered hand in a frame remains `Portable Object`, condensed label is `Portable Object`.
- Condensing does not require object co-detection.

### 4.3 `source_hand` logic

`source_hand` is derived from surviving hand-side values:

- `Left`, `Right`, `Both`, or `NA`.

## 5) Why your observed behavior happens

## 5.1 "Portable in condensed even with no object box"

This is expected under current semantics because:

- Portable is a hand contact-state class.
- Condensing uses hand labels only.
- Object matching filters do not force demotion when no object exists.

## 5.2 "No visible bbox but green pred in timeline"

`frames_det` drawing uses `archive/nr_utils/bbox_draw.py::draw_presentation_bboxes`, which only executes drawing logic when both `obj_dets` and `hand_dets` are present.

So some frames can be predicted portable in condensed while showing no drawn boxes in `frames_det` imagery.

## 5.3 "Barcode PNG looks different from timeline row"

Timeline row is horizontally compressed to video width (`np.linspace` binning in `build_timeline_panel`), so short red segments can be visually dominated by neighboring green bins at certain widths.

The underlying per-frame sequence still comes from `detections_condensed.csv` (`Pred = contact_label == Portable Object`).

## 6) Ground-truth and metric semantics

## 6.1 GT parsing

GT loaders support two formats:

1. `frame_number, gt_binary` (0/1)
2. `frame_id, label` where label can include `holding`/`not_holding`

## 6.2 Prediction binarization

Across barcode/timeline/metrics wrappers, prediction binary is generally:

- `1` iff `contact_label == Portable Object`
- `0` otherwise

That means stationary, self-contact, other-person, and no-contact collapse into 0.

## 6.3 Intersection metrics in batch script

`scripts/run_shrunk_inference_batch.py` computes metrics on frame-number intersection between prediction and GT maps, then writes:

- per-dataset `metrics/\<dataset\>/metrics_intersection.json`
- aggregated `metrics/summary_intersection.csv`
- aggregated `metrics/summary_intersection.md`

## 7) Shrunk workflow output structure (important for future runs)

When using batch/full-workflow scripts, run root is typically:

- `.../thesis labels/pipeline_runs/<run_name_or_date_profile>/`

Typical contents:

- `run_settings.json`
- `run_manifest.csv`
- `predictions/<dataset_key>/`
  - `detections_full.csv`
  - `detections_condensed.csv`
  - `config.json`
  - `preprocessing_meta.json`
  - `visualizations/`
    - `barcode_pred.png`
    - `barcode_pred_vs_gt.png` (if GT provided)
    - `frames_det/*.png` (if generated)
    - `contact_timeline_frames_det.mp4` (if generated)
- `metrics/`
  - per-dataset metrics json
  - `summary_intersection.csv`
  - `summary_intersection.md`
- `logs/` (inference logs)
- `frames_det/frames_det_manifest.csv` and `frames_det/logs/` (from full workflow)
- `videos/contact_timeline_frames_det_manifest.csv` and `videos/logs/`

## 8) What is not "broken" vs what is a semantics mismatch

Not broken (code is doing this intentionally/currently):

- Portable can exist without object row.
- Condensed uses hand labels only.
- Blue filter only demotes when threshold condition is met.
- Rendering can omit boxes despite condensed portable segments.

Potentially mismatched with desired scientific semantics:

- If your intended definition is "portable requires detected object-hand association in same frame," current pipeline does not enforce that globally.

## 9) Proposed strict semantics spec (recommended target behavior)

If you want contact labels to match strict interpretation, use this target rule set:

## 9.1 Strict portable eligibility

A hand row may be `Portable Object` only if all are true:

1. Hand model predicts portable-contact candidate.
2. Hand is not blue-glove experimenter (`blue_glove_status != experimenter`).
3. There exists a matched unfiltered object in same frame passing matching + confidence threshold.

Else relabel hand row to `No Contact` (or optionally `Stationary Object` if explicit stationary criterion is met).

## 9.2 Condensed strict labeling

Per frame, choose `Portable Object` only if at least one strict-eligible hand exists.

Otherwise use remaining priority among non-portable classes.

## 9.3 Audit columns to add (for transparency)

Recommended full-csv fields:

- `portable_candidate_raw` (bool)
- `portable_strict_eligible` (bool)
- `strict_demote_reason` (enum)
- `matched_object_id` / `matched_object_conf`
- `matched_object_iou` / `matched_object_dist`

This makes post-hoc reasoning and error analysis much easier.

## 10) Suggested acceptance tests for a strict implementation

1. Hand portable + no object in frame -> condensed not portable.
2. Hand portable + only filtered object -> condensed not portable.
3. Hand portable + blue glove experimenter -> condensed not portable.
4. Hand portable + valid matched object + non-blue -> condensed portable.
5. Two hands: one strict portable, one no-contact -> condensed portable.
6. No hand rows in frame -> condensed no-contact and `source_hand=NA`.
7. Timeline/barcode/pred binary remain consistent with condensed file values.

## 11) Practical command references

### Run batch inference + metrics on shrunk datasets

```bash
python scripts/run_shrunk_inference_batch.py \
  --run-root "/home/nripstein/Documents/thesis data/thesis labels/pipeline_runs/<run_name>" \
  --profile baseline \
  --recompute-all
```

### Generate `frames_det` from existing full CSVs only (no re-inference)

```bash
python run_pipeline.py \
  --annotated-frames-only \
  --full-csv <pred_dir>/detections_full.csv \
  --image-dir <dataset_frames_dir> \
  --output-dir <pred_dir>
```

### Render timeline videos from manifest

```bash
python scripts/render_timeline_videos_from_manifest.py \
  --run-root "/home/nripstein/Documents/thesis data/thesis labels/pipeline_runs/<run_name>" \
  --skip-existing
```

### End-to-end workflow script

```bash
python scripts/run_shrunk_full_workflow.py \
  --run-root "/home/nripstein/Documents/thesis data/thesis labels/pipeline_runs/<run_name>" \
  --profile baseline \
  --recompute-all
```

## 12) Key takeaway

Your concern is valid: current outputs are internally consistent with the implemented rules, but the implemented rules are weaker than the strict object-contact semantics you expected.

So this is best described as a semantics/design mismatch, not necessarily a runtime failure.
