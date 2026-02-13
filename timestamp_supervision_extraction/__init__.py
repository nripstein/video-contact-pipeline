"""Timestamp supervision extraction utilities."""

__all__ = [
    "evaluate_manifest",
    "evaluate_single_dataset",
]


def __getattr__(name: str):
    if name == "evaluate_manifest":
        from timestamp_supervision_extraction.evaluate_selected_timestamps import evaluate_manifest

        return evaluate_manifest
    if name == "evaluate_single_dataset":
        from timestamp_supervision_extraction.evaluate_selected_timestamps import evaluate_single_dataset

        return evaluate_single_dataset
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
