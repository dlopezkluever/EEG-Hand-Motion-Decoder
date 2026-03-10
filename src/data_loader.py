"""PhysioNet EEGBCI data download and MNE Raw loading."""

import logging
import time
from pathlib import Path

import mne
from mne.datasets import eegbci

from src.config import DATA_PATH, RUNS, SUBJECTS

logger = logging.getLogger(__name__)


def download_data(
    subjects: list[int] | None = None,
    runs: list[int] | None = None,
    data_path: str | Path = DATA_PATH,
    max_retries: int = 3,
) -> dict[int, list[Path]]:
    """Download EDF files from PhysioNet for the given subjects and runs.

    Returns a dict mapping subject ID to a list of local EDF file paths.
    Files are cached locally; re-runs do not re-download existing files.
    """
    subjects = subjects or SUBJECTS
    runs = runs or RUNS
    data_path = Path(data_path)
    data_path.mkdir(parents=True, exist_ok=True)

    downloaded: dict[int, list[Path]] = {}

    for subj in subjects:
        for attempt in range(1, max_retries + 1):
            try:
                file_paths = eegbci.load_data(
                    subj, runs, path=str(data_path), update_path=False
                )
                file_paths = [Path(p) for p in file_paths]
                downloaded[subj] = file_paths

                for fp in file_paths:
                    logger.info(
                        "Downloaded  subject=%03d  file=%s", subj, fp.name
                    )
                break

            except Exception as exc:
                wait = 2 ** attempt
                logger.warning(
                    "Download failed for subject %d (attempt %d/%d): %s  "
                    "— retrying in %ds",
                    subj, attempt, max_retries, exc, wait,
                )
                if attempt == max_retries:
                    logger.error(
                        "Giving up on subject %d after %d attempts.",
                        subj, max_retries,
                    )
                    raise
                time.sleep(wait)

    logger.info(
        "Download complete: %d subjects, %d total files.",
        len(downloaded),
        sum(len(v) for v in downloaded.values()),
    )
    return downloaded


def load_raw(
    subject: int,
    runs: list[int] | None = None,
    data_path: str | Path = DATA_PATH,
) -> mne.io.Raw:
    """Load EDF files for one subject into a single concatenated MNE Raw.

    Applies the standard_1020 montage and strips the 'S1 obj-' prefix that
    PhysioNet channel names sometimes carry.
    """
    runs = runs or RUNS
    data_path = Path(data_path)

    file_paths = eegbci.load_data(
        subject, runs, path=str(data_path), update_path=False
    )

    raws = []
    for fp in file_paths:
        raw = mne.io.read_raw_edf(fp, preload=True, verbose=False)
        raws.append(raw)

    raw = mne.concatenate_raws(raws)

    # Standardise channel names so the montage lookup works
    eegbci.standardize(raw)

    montage = mne.channels.make_standard_montage("standard_1005")
    raw.set_montage(montage, on_missing="warn")

    logger.info(
        "Loaded  subject=%03d  runs=%s  channels=%d  sfreq=%.0f Hz  "
        "n_times=%d  duration=%.1f s",
        subject,
        runs,
        len(raw.ch_names),
        raw.info["sfreq"],
        raw.n_times,
        raw.times[-1],
    )
    return raw


# ------------------------------------------------------------------
# Quick standalone check
# ------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
    )

    logger.info("=== EEG BCI Data Loader — Smoke Test ===")

    # Download & load Subject 1, Runs 3 & 7
    download_data(subjects=[1], runs=[3, 7])
    raw = load_raw(subject=1, runs=[3, 7])

    # Summary
    print(f"\nRaw data shape : {raw.get_data().shape}")
    print(f"Channels ({len(raw.ch_names)}): {raw.ch_names[:10]} ...")
    print(f"Sampling rate  : {raw.info['sfreq']} Hz")
    print(f"Duration       : {raw.times[-1]:.1f} s")

    # First 5 annotations
    annotations = raw.annotations
    print(f"\nAnnotations ({len(annotations)} total):")
    for i, ann in enumerate(annotations[:5]):
        print(f"  [{i}] onset={ann['onset']:.2f}s  duration={ann['duration']:.2f}s  desc='{ann['description']}'")

    logger.info("=== Smoke test complete ===")
