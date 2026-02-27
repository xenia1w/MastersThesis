from __future__ import annotations

import io
import zipfile
from pathlib import Path
from typing import List

from src.models.prosody import L2ArcticSample


def list_l2arctic_samples(outer_zip_path: str) -> List[L2ArcticSample]:
    samples: List[L2ArcticSample] = []
    with zipfile.ZipFile(outer_zip_path) as outer:
        inner_zips = [name for name in outer.namelist() if name.endswith(".zip")]
        for inner_name in inner_zips:
            speaker_id = Path(inner_name).stem
            inner_bytes = outer.read(inner_name)
            with zipfile.ZipFile(io.BytesIO(inner_bytes)) as inner:
                for member in inner.namelist():
                    if not member.endswith(".wav"):
                        continue
                    prefix = f"{speaker_id}/wav/"
                    if not member.startswith(prefix):
                        continue
                    wav_name = Path(member).name
                    samples.append(
                        L2ArcticSample(speaker_id=speaker_id, wav_name=wav_name)
                    )
    return samples
