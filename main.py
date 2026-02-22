from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

from src.data.audio_utils import load_l2arctic_wav
from src.features.utterance_embedding import mean_pool, mean_std_pool
from src.models.wavlm_encoder import WavLMEncoder


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return F.cosine_similarity(a, b, dim=0).item()


def frame_level_similarity_naive(a: torch.Tensor, b: torch.Tensor) -> float:
    min_len = min(a.shape[0], b.shape[0])
    a_trim = a[:min_len]
    b_trim = b[:min_len]
    return F.cosine_similarity(a_trim, b_trim, dim=1).mean().item()


def frame_level_similarity_topk(a: torch.Tensor, b: torch.Tensor, k: int = 200) -> float:
    """
    Compare frame sets without alignment by sampling frames and averaging
    max cosine similarity per frame (symmetric).
    """
    a = F.normalize(a, dim=1)
    b = F.normalize(b, dim=1)

    def sample(x: torch.Tensor) -> torch.Tensor:
        if x.shape[0] <= k:
            return x
        idx = torch.linspace(0, x.shape[0] - 1, steps=k).long()
        return x[idx]

    a_s = sample(a)
    b_s = sample(b)

    sim = a_s @ b_s.T  # [Ta, Tb]
    score_ab = sim.max(dim=1).values.mean()
    score_ba = sim.max(dim=0).values.mean()
    return 0.5 * (score_ab + score_ba).item()


def run_l2arctic_minimal() -> None:
    outer_zip = "data/raw/l2arctic_release_v5.0.zip"
    samples = [
        ("ABA", "arctic_a0001.wav"),
        ("ABA", "arctic_a0002.wav"),
        ("ABA", "arctic_a0003.wav"),
        ("ASI", "arctic_a0001.wav"),
        ("ASI", "arctic_a0002.wav"),
        ("ASI", "arctic_a0003.wav"),
        ("BWC", "arctic_a0001.wav"),
        ("BWC", "arctic_a0002.wav"),
        ("BWC", "arctic_a0003.wav"),
    ]

    encoder = WavLMEncoder(model_name="microsoft/wavlm-base-plus-sv")
    save_root = Path("data/processed/l2arctic_minimal_embeddings")
    save_root.mkdir(parents=True, exist_ok=True)

    results = []
    for speaker_id, wav_name in samples:
        waveform, sr = load_l2arctic_wav(outer_zip, speaker_id, wav_name)
        frames = encoder.encode_frames(waveform, sr)
        utt_emb = encoder.encode_utterance(waveform, sr)
        utt_emb_meanstd = mean_std_pool(frames)

        speaker_dir = save_root / speaker_id
        speaker_dir.mkdir(parents=True, exist_ok=True)
        out_path = speaker_dir / f"{wav_name.replace('.wav', '')}.pt"
        torch.save(
            {
                "speaker_id": speaker_id,
                "wav_name": wav_name,
                "sampling_rate": sr,
                "frame_representations": frames,
                "utterance_embedding": utt_emb,
                "utterance_embedding_meanstd": utt_emb_meanstd,
                "model_name": encoder.model_name,
            },
            out_path,
        )

        results.append(
            {
                "speaker_id": speaker_id,
                "wav_name": wav_name,
                "frames": frames,
                "utt_emb": utt_emb,
                "utt_emb_meanstd": utt_emb_meanstd,
            }
        )

    print("Saved embeddings to", save_root)
    print()

    # Speaker centroids (xvector)
    by_speaker = {}
    for r in results:
        by_speaker.setdefault(r["speaker_id"], []).append(r)

    centroids = {}
    for speaker_id, items in by_speaker.items():
        embs = torch.stack([it["utt_emb"] for it in items], dim=0)
        centroids[speaker_id] = F.normalize(embs.mean(dim=0), dim=0)

    print("Speaker centroid cosine similarities (xvector):")
    speakers = sorted(centroids.keys())
    for i in range(len(speakers)):
        for j in range(i + 1, len(speakers)):
            s1, s2 = speakers[i], speakers[j]
            sim = cosine(centroids[s1], centroids[s2])
            print(f"  {s1} vs {s2}: {sim:.4f}")
    print()

    print("Within-speaker avg cosine to centroid (xvector):")
    for speaker_id, items in by_speaker.items():
        sims = [cosine(it["utt_emb"], centroids[speaker_id]) for it in items]
        avg_sim = sum(sims) / len(sims)
        print(f"  {speaker_id}: {avg_sim:.4f}")


if __name__ == "__main__":
    run_l2arctic_minimal()
