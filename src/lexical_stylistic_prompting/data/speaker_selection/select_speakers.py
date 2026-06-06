"""
Rank TED-LIUM technical speakers by lexical rarity and select the top N.

Metric: mean Zipf frequency of content words across all speaker segments.
Zipf scale: 0 (unknown/very rare) – 7 (extremely common, e.g. "the").
A lower score means more unusual vocabulary.
"""

import re
from pathlib import Path
from collections import Counter

from loguru import logger
from tqdm import tqdm
from wordfreq import zipf_frequency
from datasets import load_dataset

# ── Paths ────────────────────────────────────────────────────────────────────
HERE = Path(__file__).parent
TECHNICAL_SPEAKERS_FILE = HERE / "speakers_technical.txt"
OUTPUT_FILE = HERE / "speakers_selected.txt"

# ── Parameters ───────────────────────────────────────────────────────────────
TOP_N = 30
MIN_SEGMENTS = 30          # drop speakers with too little data
MIN_WORD_LEN = 3           # ignore very short tokens
ZIPF_COMMON_THRESHOLD = 5  # words above this are too common to be informative

# Basic English stopwords to exclude from scoring
STOPWORDS = {
    "the", "and", "that", "this", "with", "for", "are", "was", "have", "not",
    "but", "they", "from", "you", "all", "had", "her", "his", "its", "our",
    "out", "who", "been", "one", "has", "their", "what", "were", "when", "we",
    "can", "more", "also", "about", "which", "into", "than", "then", "there",
    "will", "would", "could", "should", "just", "some", "very", "like", "even",
    "know", "think", "going", "get", "got", "did", "does", "said",
}


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z]+", text.lower())


def speaker_rarity_score(texts: list[str]) -> tuple[float, int, Counter]:
    """Return (mean_zipf, n_content_tokens, word_counter)."""
    word_counts: Counter = Counter()
    for text in texts:
        for tok in tokenize(text):
            if len(tok) >= MIN_WORD_LEN and tok not in STOPWORDS:
                word_counts[tok] += 1

    if not word_counts:
        return 7.0, 0, word_counts

    total_zipf = 0.0
    n_tokens = 0
    for word, count in word_counts.items():
        z = zipf_frequency(word, "en")
        if z > ZIPF_COMMON_THRESHOLD:
            continue  # skip very common content words too
        total_zipf += z * count
        n_tokens += count

    if n_tokens == 0:
        return 7.0, 0, word_counts

    return total_zipf / n_tokens, n_tokens, word_counts


def main() -> None:
    technical_speakers = set(
        line.strip()
        for line in TECHNICAL_SPEAKERS_FILE.read_text().splitlines()
        if line.strip()
    )
    logger.info(f"Candidate technical speakers: {len(technical_speakers)}")

    logger.info("Loading TED-LIUM dataset (train split) from cache...")
    dataset = load_dataset(
        "distil-whisper/tedlium", "release3",
        split="train",
        trust_remote_code=True,
    )

    # Dataset speaker_id includes a year suffix (e.g. "AlGore_2006").
    # Score each talk independently — keeps vocabulary profiles focused per topic.
    speaker_texts: dict[str, list[str]] = {}
    for example in tqdm(dataset.select_columns(["speaker_id", "text"]), desc="Scanning dataset"):
        sid = example["speaker_id"]
        if sid.rsplit("_", 1)[0] in technical_speakers:
            speaker_texts.setdefault(sid, []).append(example["text"])

    logger.info(f"Found {len(speaker_texts)} technical speakers in dataset")

    # Score each speaker
    results = []
    for sid, texts in tqdm(speaker_texts.items(), desc="Scoring speakers"):
        if len(texts) < MIN_SEGMENTS:
            continue
        mean_zipf, n_tokens, word_counts = speaker_rarity_score(texts)
        n_unique = len(word_counts)
        results.append({
            "speaker_id": sid,
            "mean_zipf": mean_zipf,
            "n_segments": len(texts),
            "n_tokens": n_tokens,
            "n_unique_words": n_unique,
            "top_rare_words": [
                w for w, _ in word_counts.most_common()
                if zipf_frequency(w, "en") < 3.0
            ][:8],
        })

    # Sort by mean Zipf ascending (lower = rarer = more unusual)
    results.sort(key=lambda x: x["mean_zipf"])
    logger.info(f"Scored {len(results)} speakers with ≥ {MIN_SEGMENTS} segments\n")

    # Print ranked table
    col = f"{'Rank':<5} {'Speaker':<30} {'Mean Zipf':>9} {'Segments':>9} {'Unique words':>13}"
    print(col)
    print("-" * len(col))
    for rank, r in enumerate(results[:TOP_N * 2], 1):
        print(
            f"{rank:<5} {r['speaker_id']:<30} {r['mean_zipf']:>9.3f} "
            f"{r['n_segments']:>9} {r['n_unique_words']:>13}  "
            f"rare: {', '.join(r['top_rare_words'])}"
        )

    # Save top N (full talk IDs, e.g. "AlGore_2006")
    top = results[:TOP_N]
    OUTPUT_FILE.write_text("\n".join(r["speaker_id"] for r in top) + "\n")
    logger.info(f"\nSaved top {TOP_N} speakers → {OUTPUT_FILE}")

    # Summary stats
    print(f"\nTop {TOP_N} mean Zipf range: "
          f"{top[0]['mean_zipf']:.3f} – {top[-1]['mean_zipf']:.3f}")
    print(f"All candidates mean Zipf range: "
          f"{results[0]['mean_zipf']:.3f} – {results[-1]['mean_zipf']:.3f}")


if __name__ == "__main__":
    main()
