from __future__ import annotations

import random
from typing import Dict, List

from stimulus_detector.data_generation.types import PseudoLabel, SplitResult


def build_participant_split(
    participant_ids: List[str],
    val_fraction: float,
    seed: int,
) -> Dict[str, str]:
    participants = sorted({pid for pid in participant_ids if pid})
    if not participants:
        return {}
    if len(participants) == 1:
        return {participants[0]: "train"}

    rng = random.Random(seed)
    shuffled = participants[:]
    rng.shuffle(shuffled)

    val_count = int(round(len(shuffled) * val_fraction))
    val_count = max(1, min(len(shuffled) - 1, val_count))

    val_set = set(shuffled[:val_count])
    return {pid: ("val" if pid in val_set else "train") for pid in participants}


def split_labels_by_participant(
    labels: List[PseudoLabel],
    val_fraction: float,
    seed: int,
) -> SplitResult:
    participant_ids = [label.frame.participant_id for label in labels]
    participant_to_split = build_participant_split(participant_ids, val_fraction=val_fraction, seed=seed)

    train_labels: List[PseudoLabel] = []
    val_labels: List[PseudoLabel] = []

    for label in labels:
        split = participant_to_split.get(label.frame.participant_id, "train")
        if split == "val":
            val_labels.append(label)
        else:
            train_labels.append(label)

    return SplitResult(
        participant_to_split=participant_to_split,
        train_labels=train_labels,
        val_labels=val_labels,
    )
