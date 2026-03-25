"""
Tests for scripts/deploy_models.py.

Only the pure-Python functions are tested here (_pick_latest, build_sklearn_files,
upload). No GCS connection is made.
"""
import pytest
from pathlib import Path

from scripts.deploy_models import _pick_latest, build_sklearn_files, upload


# ---------------------------------------------------------------------------
# _pick_latest
# ---------------------------------------------------------------------------
class TestPickLatest:
    def test_returns_most_recent_by_name(self, tmp_path):
        (tmp_path / "config_sklearn__20260101.json").touch()
        (tmp_path / "config_sklearn__20260322.json").touch()
        (tmp_path / "config_sklearn__20260101_old.json").touch()
        result = _pick_latest(tmp_path, "config_sklearn__*.json")
        assert result.name == "config_sklearn__20260322.json"

    def test_returns_single_match(self, tmp_path):
        (tmp_path / "config_sklearn__20260101.json").touch()
        result = _pick_latest(tmp_path, "config_sklearn__*.json")
        assert result.name == "config_sklearn__20260101.json"

    def test_raises_when_no_match(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="No file matching"):
            _pick_latest(tmp_path, "config_sklearn__*.json")

    def test_raises_mentions_directory(self, tmp_path):
        with pytest.raises(FileNotFoundError, match=str(tmp_path)):
            _pick_latest(tmp_path, "config_sklearn__*.json")


# ---------------------------------------------------------------------------
# build_sklearn_files
# ---------------------------------------------------------------------------
class TestBuildSklearnFiles:
    def _make_sklearn_dir(self, tmp_path, timestamp="20260322"):
        (tmp_path / f"config_sklearn__{timestamp}.json").touch()
        (tmp_path / f"label_encoder_sklearn__{timestamp}.pkl").touch()
        (tmp_path / f"efficientnet_b3__logistic_regression__{timestamp}.pkl").touch()
        return tmp_path

    def test_returns_three_entries(self, tmp_path):
        result = build_sklearn_files(self._make_sklearn_dir(tmp_path))
        assert len(result) == 3

    def test_blob_names_are_under_models_sklearn_prefix(self, tmp_path):
        result = build_sklearn_files(self._make_sklearn_dir(tmp_path))
        for blob_name in result.values():
            assert blob_name.startswith("models_sklearn/")

    def test_blob_names_contain_expected_patterns(self, tmp_path):
        result = build_sklearn_files(self._make_sklearn_dir(tmp_path))
        blob_names = list(result.values())
        assert any("config_sklearn" in b for b in blob_names)
        assert any("label_encoder" in b for b in blob_names)
        assert any("logistic_regression" in b for b in blob_names)

    def test_local_paths_are_path_objects(self, tmp_path):
        result = build_sklearn_files(self._make_sklearn_dir(tmp_path))
        for local_path in result.keys():
            assert isinstance(local_path, Path)

    def test_picks_latest_when_multiple_timestamps(self, tmp_path):
        self._make_sklearn_dir(tmp_path, "20260101")
        self._make_sklearn_dir(tmp_path, "20260322")
        result = build_sklearn_files(tmp_path)
        for local_path in result.keys():
            assert "20260322" in local_path.name

    def test_raises_when_files_missing(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            build_sklearn_files(tmp_path)


# ---------------------------------------------------------------------------
# upload
# ---------------------------------------------------------------------------
class TestUpload:
    def _make_file(self, tmp_path, size_bytes=1_048_576):
        f = tmp_path / "model.pt"
        f.write_bytes(b"x" * size_bytes)
        return f

    def test_dry_run_prints_dry_run_marker(self, tmp_path, capsys):
        bucket = type("Bucket", (), {"name": "test-bucket"})()
        upload(bucket, self._make_file(tmp_path), "models/model.pt", dry_run=True)
        assert "[DRY-RUN]" in capsys.readouterr().out

    def test_dry_run_prints_filename(self, tmp_path, capsys):
        bucket = type("Bucket", (), {"name": "test-bucket"})()
        upload(bucket, self._make_file(tmp_path), "models/model.pt", dry_run=True)
        assert "model.pt" in capsys.readouterr().out

    def test_dry_run_prints_file_size_mb(self, tmp_path, capsys):
        bucket = type("Bucket", (), {"name": "test-bucket"})()
        upload(bucket, self._make_file(tmp_path, 1_048_576), "models/model.pt", dry_run=True)
        assert "1.0 MB" in capsys.readouterr().out

    def test_dry_run_does_not_call_gcs(self, tmp_path):
        called = []

        class FakeBlob:
            def upload_from_filename(self, path):
                called.append(path)

        class FakeBucket:
            name = "test-bucket"
            def blob(self, name):
                return FakeBlob()

        upload(FakeBucket(), self._make_file(tmp_path), "models/model.pt", dry_run=True)
        assert called == []

    def test_real_upload_calls_gcs_with_correct_blob_name(self, tmp_path):
        uploaded = {}

        class FakeBlob:
            def __init__(self, name):
                self.name = name
            def upload_from_filename(self, path):
                uploaded[self.name] = path

        class FakeBucket:
            name = "test-bucket"
            def blob(self, name):
                return FakeBlob(name)

        f = self._make_file(tmp_path)
        upload(FakeBucket(), f, "models/model.pt", dry_run=False)
        assert "models/model.pt" in uploaded

    def test_real_upload_passes_correct_local_path(self, tmp_path):
        uploaded = {}

        class FakeBlob:
            def __init__(self, name):
                self.name = name
            def upload_from_filename(self, path):
                uploaded[self.name] = path

        class FakeBucket:
            name = "test-bucket"
            def blob(self, name):
                return FakeBlob(name)

        f = self._make_file(tmp_path)
        upload(FakeBucket(), f, "models/model.pt", dry_run=False)
        assert uploaded["models/model.pt"] == str(f)
