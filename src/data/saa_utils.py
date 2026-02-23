from __future__ import annotations

import csv
import io
import zipfile
from typing import List, Optional

from src.models.saa_minimal import SAASample


def _parse_int(value: str | None) -> Optional[int]:
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _parse_bool(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"true", "1", "yes"}


def _list_saa_recordings(outer_zip_path: str) -> set[str]:
    with zipfile.ZipFile(outer_zip_path) as outer:
        return {
            name
            for name in outer.namelist()
            if name.startswith("recordings/recordings/")
            and name.endswith(".mp3")
        }


def load_saa_samples(
    outer_zip_path: str,
    include_missing: bool = False,
    validate_files: bool = True,
) -> List[SAASample]:
    """
    Load Speech Accent Archive metadata from speakers_all.csv and return samples.
    """
    available = _list_saa_recordings(outer_zip_path) if validate_files else None

    with zipfile.ZipFile(outer_zip_path) as outer:
        csv_bytes = outer.read("speakers_all.csv")

    reader = csv.DictReader(io.TextIOWrapper(io.BytesIO(csv_bytes), encoding="utf-8"))
    samples: List[SAASample] = []
    for raw_row in reader:
        row = {k: v for k, v in raw_row.items() if k}
        filename = (row.get("filename") or "").strip()
        if not filename:
            continue

        file_missing = _parse_bool(row.get("file_missing?"))
        if file_missing and not include_missing:
            continue

        if available is not None:
            mp3_path = f"recordings/recordings/{filename}.mp3"
            if mp3_path not in available:
                continue

        speaker_id = str((row.get("speakerid") or "").strip())
        if not speaker_id:
            continue

        samples.append(
            SAASample(
                speaker_id=speaker_id,
                filename=filename,
                native_language=(row.get("native_language") or "").strip(),
                sex=(row.get("sex") or "").strip() or None,
                country=(row.get("country") or "").strip() or None,
                age=_parse_int(row.get("age")),
                age_onset=_parse_int(row.get("age_onset")),
                birthplace=(row.get("birthplace") or "").strip() or None,
            )
        )

    return samples
