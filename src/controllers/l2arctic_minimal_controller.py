from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

from src.data.audio_utils import load_l2arctic_wav
from src.features.utterance_embedding import mean_std_pool
from src.models.l2arctic_minimal import L2ArcticEmbedding, L2ArcticSample
from src.models.wavlm_encoder import WavLMEncoder


class L2ArcticMinimalController:
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

    def encode_sample(self, sample: L2ArcticSample) -> L2ArcticEmbedding:
        waveform, sr = load_l2arctic_wav(
            self.outer_zip, sample.speaker_id, sample.wav_name
        )
        frames = self.encoder.encode_frames(waveform, sr)
        utt_emb = self.encoder.encode_utterance(waveform, sr)
        utt_emb_meanstd = mean_std_pool(frames)

        return L2ArcticEmbedding(
            speaker_id=sample.speaker_id,
            wav_name=sample.wav_name,
            sampling_rate=sr,
            frames=frames,
            utt_emb=utt_emb,
            utt_emb_meanstd=utt_emb_meanstd,
        )

    def save_embedding(self, embedding: L2ArcticEmbedding) -> Path:
        speaker_dir = self.save_root_path / embedding.speaker_id
        speaker_dir.mkdir(parents=True, exist_ok=True)
        out_path = speaker_dir / f"{embedding.wav_name.replace('.wav', '')}.pt"
        self._save_embedding_payload(out_path, embedding)
        return out_path

    def run(self, samples: Iterable[L2ArcticSample]) -> List[L2ArcticEmbedding]:
        results = []
        for sample in samples:
            embedding = self.encode_sample(sample)
            self.save_embedding(embedding)
            results.append(embedding)
        return results

    def _save_embedding_payload(
        self,
        out_path: Path,
        embedding: L2ArcticEmbedding,
    ) -> None:
        payload = {
            "speaker_id": embedding.speaker_id,
            "wav_name": embedding.wav_name,
            "sampling_rate": embedding.sampling_rate,
            "frame_representations": embedding.frames,
            "utterance_embedding": embedding.utt_emb,
            "utterance_embedding_meanstd": embedding.utt_emb_meanstd,
            "model_name": self.encoder.model_name,
        }
        import torch

        torch.save(payload, out_path)
