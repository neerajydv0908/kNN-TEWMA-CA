import csv

from knn_tewma_ca.cli import main


def test_reproduce_command_prints_presentation_numbers(tmp_path, capsys):
    assert main(["reproduce", "--out", str(tmp_path), "--no-plots", "--quiet"]) == 0
    out = capsys.readouterr().out
    for number in ("0.5400", "0.8433", "0.7600", "0.8054", "0.7067", "0.8933", "+56.2%"):
        assert number in out
    assert (tmp_path / "reproduce.md").exists()


def test_full_run_writes_reports_and_clean_charts(tmp_path):
    assert main(["all", "--seeds", "2", "--workers", "2", "--out", str(tmp_path), "--quiet"]) == 0
    for name in ("reproduce.md", "corrected.md", "comparison.md", "runs.csv"):
        assert (tmp_path / name).stat().st_size > 0

    with (tmp_path / "runs.csv").open() as fh:
        modes = {row["mode"] for row in csv.DictReader(fh)}
    assert modes == {"reproduce", "corrected", "round2", "allocated_end", "all_heads", "censored"}

    pngs = sorted((tmp_path / "figures").glob("*.png"))
    assert len(pngs) == 12
    for png in pngs:
        data = png.read_bytes()
        assert data.startswith(b"\x89PNG")
        assert b"tEXt" not in data and b"iTXt" not in data, f"{png.name} carries text metadata"
