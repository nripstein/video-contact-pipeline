from __future__ import annotations

from pathlib import Path

from pipeline.preprocessing import get_sorted_image_list


def test_frame_sort_order_numeric_first(tmp_path: Path):
    filenames = [
        "10.png",
        "2.png",
        "001.png",
        "frame_3.png",
        "abc.png",
        ".hidden.png",
    ]

    for name in filenames:
        (tmp_path / name).write_text("x", encoding="utf-8")

    sorted_list = get_sorted_image_list(str(tmp_path))
    sorted_names = [Path(p).name for p in sorted_list]

    assert sorted_names == ["001.png", "2.png", "frame_3.png", "10.png", "abc.png"]
