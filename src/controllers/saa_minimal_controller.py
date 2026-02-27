from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

from src.data.audio_utils import load_saa_mp3
from src.features.utterance_embedding import mean_std_pool
from src.models.prosody import ProsodyEmbedding, SAAMetadata, SAASample
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

    def encode_sample(self, sample: SAASample) -> ProsodyEmbedding:
        waveform, sr = load_saa_mp3(self.outer_zip, sample.filename)
        frames = self.encoder.encode_frames(waveform, sr)
        utt_emb = self.encoder.encode_utterance(waveform, sr)
        utt_emb_meanstd = mean_std_pool(frames)

        return ProsodyEmbedding(
            dataset="saa",
            speaker_id=sample.speaker_id,
            file_name=sample.filename,
            sampling_rate=sr,
            frames=frames,
            utt_emb=utt_emb,
            utt_emb_meanstd=utt_emb_meanstd,
            saa_metadata=SAAMetadata(
                native_language=sample.native_language,
                sex=sample.sex,
                country=sample.country,
                age=sample.age,
                age_onset=sample.age_onset,
                birthplace=sample.birthplace,
            ),
        )

    def save_embedding(self, embedding: ProsodyEmbedding) -> Path:
        speaker_dir = self.save_root_path / embedding.speaker_id
        speaker_dir.mkdir(parents=True, exist_ok=True)
        out_path = speaker_dir / f"{embedding.file_name}.pt"
        self._save_embedding_payload(out_path, embedding)
        return out_path

    def run(self, samples: Iterable[SAASample]) -> List[ProsodyEmbedding]:
        results = []
        for sample in samples:
            embedding = self.encode_sample(sample)
            self.save_embedding(embedding)
            results.append(embedding)
        return results

    def _save_embedding_payload(
        self,
        out_path: Path,
        embedding: ProsodyEmbedding,
    ) -> None:
        saa_meta = embedding.saa_metadata
        payload = {
            "dataset": embedding.dataset,
            "speaker_id": embedding.speaker_id,
            "file_name": embedding.file_name,
            "native_language": saa_meta.native_language if saa_meta else None,
            "sex": saa_meta.sex if saa_meta else None,
            "country": saa_meta.country if saa_meta else None,
            "age": saa_meta.age if saa_meta else None,
            "age_onset": saa_meta.age_onset if saa_meta else None,
            "birthplace": saa_meta.birthplace if saa_meta else None,
            "sampling_rate": embedding.sampling_rate,
            "frame_representations": embedding.frames,
            "utterance_embedding": embedding.utt_emb,
            "utterance_embedding_meanstd": embedding.utt_emb_meanstd,
            "model_name": self.encoder.model_name,
        }
        import torch

        torch.save(payload, out_path)
