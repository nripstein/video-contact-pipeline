from __future__ import annotations

from pathlib import Path
import sys

TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parents[0]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.main import list_videos


def test_list_videos_sorted_case_insensitive(tmp_path: Path):
    (tmp_path / "a.mp4").write_text("x")
    (tmp_path / "b.MP4").write_text("x")
    (tmp_path / "c.txt").write_text("x")
    (tmp_path / "d.mov").write_text("x")

    videos = list_videos(str(tmp_path))
    names = [Path(p).name for p in videos]
    assert names == ["a.mp4", "b.MP4", "d.mov"]
