from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

from src.data.audio_utils import load_saa_mp3
from src.features.utterance_embedding import mean_std_pool
from src.models.saa_minimal import SAAEmbedding, SAASample
from src.models.wavlm_encoder import WavLMEncoder


class SAAMinimalController:
    def __init__(
        self,
        outer_zip: str,
        model_name: str,
        save_root: str,
    ) -> None:
        self.outer_zip = outer_zip
        self.encoder = WavLMEncoder(model_name=model_name)
        self.save_root_path = Path(save_root)
        self.save_root_path.mkdir(parents=True, exist_ok=True)

    def encode_sample(self, sample: SAASample) -> SAAEmbedding:
        waveform, sr = load_saa_mp3(self.outer_zip, sample.filename)
        frames = self.encoder.encode_frames(waveform, sr)
        utt_emb = self.encoder.encode_utterance(waveform, sr)
        utt_emb_meanstd = mean_std_pool(frames)

        return SAAEmbedding(
            speaker_id=sample.speaker_id,
            filename=sample.filename,
            native_language=sample.native_language,
            sex=sample.sex,
            country=sample.country,
            age=sample.age,
            age_onset=sample.age_onset,
            birthplace=sample.birthplace,
            sampling_rate=sr,
            frames=frames,
            utt_emb=utt_emb,
            utt_emb_meanstd=utt_emb_meanstd,
        )

    def save_embedding(self, embedding: SAAEmbedding) -> Path:
        speaker_dir = self.save_root_path / embedding.speaker_id
        speaker_dir.mkdir(parents=True, exist_ok=True)
        out_path = speaker_dir / f"{embedding.filename}.pt"
        self._save_embedding_payload(out_path, embedding)
        return out_path

    def run(self, samples: Iterable[SAASample]) -> List[SAAEmbedding]:
        results = []
        for sample in samples:
            embedding = self.encode_sample(sample)
            self.save_embedding(embedding)
            results.append(embedding)
        return results

    def _save_embedding_payload(
        self,
        out_path: Path,
        embedding: SAAEmbedding,
    ) -> None:
        payload = {
            "speaker_id": embedding.speaker_id,
            "filename": embedding.filename,
            "native_language": embedding.native_language,
            "sex": embedding.sex,
            "country": embedding.country,
            "age": embedding.age,
            "age_onset": embedding.age_onset,
            "birthplace": embedding.birthplace,
            "sampling_rate": embedding.sampling_rate,
            "frame_representations": embedding.frames,
            "utterance_embedding": embedding.utt_emb,
            "utterance_embedding_meanstd": embedding.utt_emb_meanstd,
            "model_name": self.encoder.model_name,
        }
        import torch

        torch.save(payload, out_path)
