from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Tuple

import torch
import torch.nn.functional as F

from src.data.audio_utils import load_l2arctic_wav
from src.features.utterance_embedding import mean_std_pool
from src.metrics.similarity import (
    cosine,
    frame_level_similarity_naive,
    frame_level_similarity_topk,
)
from src.models.wavlm_encoder import WavLMEncoder


def default_samples() -> List[Tuple[str, str]]:
    return [
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


def run_l2arctic_minimal(
    outer_zip: str = "data/raw/l2arctic_release_v5.0.zip",
    samples: Iterable[Tuple[str, str]] | None = None,
    model_name: str = "microsoft/wavlm-base-plus-sv",
    save_root: str = "data/processed/l2arctic_minimal_embeddings",
) -> None:
    sample_list = list(samples) if samples is not None else default_samples()

    encoder = WavLMEncoder(model_name=model_name)
    save_root_path = Path(save_root)
    save_root_path.mkdir(parents=True, exist_ok=True)

    results = []
    for speaker_id, wav_name in sample_list:
        waveform, sr = load_l2arctic_wav(outer_zip, speaker_id, wav_name)
        frames = encoder.encode_frames(waveform, sr)
        utt_emb = encoder.encode_utterance(waveform, sr)
        utt_emb_meanstd = mean_std_pool(frames)

        speaker_dir = save_root_path / speaker_id
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

    print("Saved embeddings to", save_root_path)
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

    print()
    print("Pairwise comparisons (xvector + frame-level):")
    for i in range(len(results)):
        for j in range(i + 1, len(results)):
            a = results[i]
            b = results[j]
            utt_sim = cosine(a["utt_emb"], b["utt_emb"])
            utt_sim_meanstd = cosine(a["utt_emb_meanstd"], b["utt_emb_meanstd"])
            frame_sim_naive = frame_level_similarity_naive(a["frames"], b["frames"])
            frame_sim_topk = frame_level_similarity_topk(a["frames"], b["frames"])
            label = f'{a["speaker_id"]}:{a["wav_name"]} vs {b["speaker_id"]}:{b["wav_name"]}'
            print(label)
            print(f"  utterance-level cosine (xvector): {utt_sim:.4f}")
            print(f"  utterance-level cosine (mean+std): {utt_sim_meanstd:.4f}")
            print(f"  frame-level cosine (naive): {frame_sim_naive:.4f}")
            print(f"  frame-level cosine (topk):  {frame_sim_topk:.4f}")
            print()
