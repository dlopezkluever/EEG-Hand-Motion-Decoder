"""Basic tests for data loading and configuration."""

import sys
from pathlib import Path

import pytest

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TestConfig:
    """Verify config module imports and returns expected values."""

    def test_config_imports(self):
        from src.config import SUBJECTS, RUNS, DATA_PATH, get_config

        assert isinstance(SUBJECTS, list)
        assert RUNS == [3, 7]
        assert DATA_PATH.name == "eegbci"

    def test_get_config_returns_dict(self):
        from src.config import get_config

        cfg = get_config()
        assert isinstance(cfg, dict)
        assert "SUBJECTS" in cfg
        assert "RANDOM_SEED" in cfg


class TestDataLoader:
    """Verify data download and Raw loading for Subject 1."""

    @pytest.fixture(scope="class")
    def raw(self):
        from src.data_loader import download_data, load_raw

        download_data(subjects=[1], runs=[3, 7])
        return load_raw(subject=1, runs=[3, 7])

    def test_channel_count(self, raw):
        assert len(raw.ch_names) == 64

    def test_sampling_rate(self, raw):
        assert raw.info["sfreq"] == 160.0

    def test_annotations_exist(self, raw):
        assert len(raw.annotations) > 0

    def test_data_shape(self, raw):
        data = raw.get_data()
        assert data.shape[0] == 64     # channels
        assert data.shape[1] > 0       # time points
