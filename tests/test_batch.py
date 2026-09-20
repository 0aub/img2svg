import csv

import numpy as np
import pytest
from PIL import Image

from img2svg import batch
from img2svg.config import Config


@pytest.fixture
def folder(tmp_path, flat_art):
    d = tmp_path / "in"
    d.mkdir()
    Image.fromarray(flat_art).save(d / "one.png")
    Image.fromarray(flat_art[::-1]).save(d / "two.png")
    return d


def test_each_input_gets_its_own_folder(folder, tmp_path):
    out = tmp_path / "out"
    rows = batch.run([str(p) for p in sorted(folder.iterdir())], str(out),
                     Config(), compare=False, log=lambda *a: None)
    assert [r["name"] for r in rows] == ["one", "two"]
    for name in ("one", "two"):
        assert (out / name / (name + ".svg")).exists()
    assert (out / "index.html").exists()
    assert (out / "results.csv").exists()


def test_index_is_sorted_worst_first(tmp_path):
    rows = [{"name": "good", "status": "ok", "mean": 0.2, "p95": 0.4, "curves": 5, "kb": 1.0},
            {"name": "bad", "status": "ok", "mean": 3.1, "p95": 6.0, "curves": 9, "kb": 2.0}]
    html = batch._index(rows, Config())
    assert html.index("bad") < html.index("good")


def test_csv_carries_every_field(folder, tmp_path):
    out = tmp_path / "out"
    batch.run([str(p) for p in sorted(folder.iterdir())], str(out), Config(),
              compare=False, log=lambda *a: None)
    with open(out / "results.csv", encoding="utf-8") as fh:
        got = list(csv.DictReader(fh))
    assert {r["name"] for r in got} == {"one", "two"}
    assert all(r["status"] == "ok" for r in got)
    assert all(int(r["curves"]) > 0 for r in got)


def test_one_unreadable_file_does_not_end_the_batch(folder, tmp_path):
    (folder / "broken.png").write_bytes(b"not a png")
    out = tmp_path / "out"
    rows = batch.run([str(p) for p in sorted(folder.iterdir())], str(out), Config(),
                     compare=False, log=lambda *a: None)
    by = {r["name"]: r for r in rows}
    assert by["broken"]["status"] == "error"
    assert by["one"]["status"] == "ok" and by["two"]["status"] == "ok"


def test_colliding_basenames_keep_both_results(tmp_path, flat_art):
    a, b = tmp_path / "a", tmp_path / "b"
    for d in (a, b):
        d.mkdir()
        Image.fromarray(flat_art).save(d / "logo.png")
    out = tmp_path / "out"
    rows = batch.run([str(a / "logo.png"), str(b / "logo.png")], str(out), Config(),
                     compare=False, log=lambda *a: None)
    assert [r["name"] for r in rows] == ["logo", "logo-2"]
    assert (out / "logo" / "logo.svg").exists()
    assert (out / "logo-2" / "logo-2.svg").exists()


def test_difference_map_is_black_where_it_matches():
    heat = batch._heat(np.zeros((4, 4), dtype=np.float32))
    assert heat.shape == (4, 4, 3)
    assert heat.max() == 0
    assert batch._heat(np.full((4, 4), 6.0, dtype=np.float32))[..., 0].min() == 255
