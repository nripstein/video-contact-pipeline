# Object Tracking Integration Plan

## Purpose
Add an optional object-tracking step to improve contact detection robustness when detector outputs are noisy (partial occlusion, rotation, short misses), while preserving baseline behavior by default.

## Context
The current pipeline uses per-frame detections plus filtering and condensing. The proposed tracking step adds short-horizon temporal continuity so brief detection failures can still contribute valid contact evidence.

## Primary Goal
Improve recall for true portable-object contact without causing large precision regressions.

## Non-Goals
- Replacing the detector with a full tracker-first pipeline.
- Building multi-object identity tracking.
- Adding trial segmentation models.
- Changing default behavior for existing runs.

## Locked Design Decisions
- Tracking logic lives in top-level `object_tracking/` for branch and module isolation.
- Integration point is after `apply_detection_filters(...)` and before `condense_dataframe(...)`.
- Feature ships off by default.
- Strategy is motion-only bridge (detector remains primary source of truth).
- Strict initialization is required.
- Default gap budget is 8 frames and configurable.
- Default promotion confirmation is 2 frames and configurable.

## Repository Placement
- `object_tracking/bridge.py`: orchestration and frame-by-frame tracking bridge.
- `object_tracking/state.py`: track dataclasses and state labels.
- `object_tracking/geometry.py`: IoU and geometry helpers.
- `object_tracking/motion.py`: linear bbox motion propagation.
- `object_tracking/__init__.py`: public package export.

## Public Interfaces

### PipelineConfig fields
Added to `pipeline/config.py`:
- `tracking_bridge_enabled: bool = False`
- `tracking_max_missed_frames: int = 8`
- `tracking_contact_iou_threshold: float = 0.15`
- `tracking_init_obj_confidence: float = 0.70`
- `tracking_promotion_confirm_frames: int = 2`
- `tracking_reassociate_iou_threshold: float = 0.10`

### CLI flags
Added to `run_pipeline.py`:
- `--tracking-bridge`
- `--tracking-max-missed-frames`
- `--tracking-iou-threshold`
- `--tracking-init-obj-confidence`
- `--tracking-promotion-confirm-frames`
- `--tracking-reassociate-iou-threshold`

### Full detections schema additions
Tracking metadata columns in `detections_full.csv`:
- `tracking_promoted`
- `tracking_track_id`
- `tracking_iou`
- `tracking_missed_count`
- `tracking_state`

No schema change for `detections_condensed.csv`.

## End-to-End Data Flow
1. Run preprocessing and inference as usual.
2. Apply existing filters (`blue glove`, object size, ratio filters).
3. If tracking enabled, run tracking bridge over filtered full detections.
4. Condense to one contact label per frame.
5. Save outputs and optional visualizations.

## Tracking State Machine
- `inactive`: no active track.
- `tracking`: active track exists and is being updated or propagated.
- `lost`: track exceeded miss budget and was dropped for that frame.

Track can be reinitialized later when strict initialization criteria are met again.

## Strict Initialization Rule
Initialize a track only if all conditions hold in a frame:
- At least one unfiltered hand row labeled `Portable Object`.
- At least one unfiltered object row with confidence above init threshold.
- Hand row is not marked as experimenter blue glove.
- Best object selected by deterministic ranking:
  - highest confidence
  - shortest hand-object center distance
  - highest IoU

## Motion-Only Bridge Rule
When active track exists:
- If a matching object detection is available and reassociation IoU is sufficient, update track with detection bbox.
- Otherwise predict next bbox using linear center velocity from previous bboxes.
- Increment miss counter on propagated frames.
- Drop track when miss counter exceeds `tracking_max_missed_frames`.

## Promotion Rule (Contact Recovery)
Promotion is allowed only on detector-miss frames and only for eligible hand rows:
- Candidate hand must be unfiltered.
- Candidate hand must not be experimenter blue glove.
- Candidate hand label must be `No Contact`.
- Track-to-hand IoU must be >= `tracking_contact_iou_threshold`.
- IoU condition must hold for `tracking_promotion_confirm_frames` consecutive eligible miss frames.

When promoted:
- Set `contact_label = Portable Object`
- Set `contact_state = 3`
- Mark tracking metadata columns for provenance.

## Safety Guardrails
- Never promote on experimenter-blue-glove frames.
- Never promote when no active track exists.
- Never promote on detector-hit frames.
- Drop uncertain tracks quickly via max-miss budget.
- Keep feature off by default for reproducibility and parity safety.

## Configuration Semantics
- Confidence values support both `[0,1]` and `[0,100]` style input thresholds.
- Thresholds are clamped into valid ranges where applicable.
- Confirmation frames are clamped to minimum `1`.
- Miss budget is clamped to minimum `0`.

## Testing Plan

### Unit coverage
`tests/test_object_tracking_bridge_unit.py`:
- strict init required
- no init from weak seeds
- promotion after confirmation streak
- no single-frame spike promotion
- miss budget expiry
- reinit behavior after lost state
- blue-glove promotion blocking

### Interface coverage
- `tests/test_cli_tracking_flags.py` validates CLI defaults and overrides.
- `tests/test_config_roundtrip.py` validates serialization roundtrip of tracking config fields.

### Regression coverage
- Keep parity behavior unchanged with tracking disabled.
- Existing baseline tests should remain valid.

## Evaluation Plan
Evaluate against labeled data using existing metrics:
- MoF
- Edit score
- F1@10/25/50/75
- Confusion counts (TP/FP/TN/FN)

Compare:
- baseline predictions (tracking off)
- tracking-enabled predictions

Run small sweeps on:
- `tracking_max_missed_frames`
- `tracking_contact_iou_threshold`
- `tracking_promotion_confirm_frames`
- optionally `tracking_init_obj_confidence` and reassociation IoU

Primary decision criterion:
- recover false negatives without unacceptable false positive growth.

## Rollout Plan
1. Keep tracking disabled by default.
2. Evaluate on held-out participant videos.
3. Select a recommended profile only if metrics improve materially.
4. Document tuned profile in README/Quickstart.

## Acceptance Criteria
- Code path is fully optional and does not alter default outputs.
- Tracking metadata is present when tracking runs.
- Tests covering tracking logic and interfaces pass.
- Evaluation artifacts show clear before/after comparison.

## Branch and Workflow Notes
- The implementation is intentionally isolated in `object_tracking/` for clean branch history and review.
- Pipeline changes are narrow: config, CLI wiring, one integration callsite, documentation, and tests.
