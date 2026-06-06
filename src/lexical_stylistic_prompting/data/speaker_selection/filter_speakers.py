"""
Match TED-LIUM speakers against TED-plus metadata and filter by topic.
Run: python filter_speakers.py
"""
import re
import urllib.request
import csv
import io

META_URL = "https://raw.githubusercontent.com/cynco/TED-plus-and-GenderListener/master/data/meta_plus.csv"
SPEAKERS_FILE = "/Users/xmena/Desktop/speakers_100.txt"

TOPICS = ["technology", "science", "health", "innovation", "design", "business"]


def camel_to_words(name: str) -> str:
    """PaulaScher -> paula scher"""
    return re.sub(r"([a-z])([A-Z])", r"\1 \2", name).lower()


def normalize(name: str) -> str:
    return re.sub(r"\s+", "", name).lower()


def main():
    # Load TED-LIUM speakers
    with open(SPEAKERS_FILE) as f:
        tedlium_names = [line.strip() for line in f if line.strip()]

    tedlium_normalized = {normalize(n): n for n in tedlium_names}

    # Download meta_plus.csv
    print("Downloading meta_plus.csv ...")
    with urllib.request.urlopen(META_URL) as r:
        content = r.read().decode("utf-8")

    reader = csv.DictReader(io.StringIO(content))
    rows = list(reader)
    print(f"TED-plus: {len(rows)} talks, {len(tedlium_names)} TED-LIUM speakers\n")

    # Match
    matched = []
    for row in rows:
        key = normalize(row["main_speaker"])
        if key in tedlium_normalized:
            matched.append({
                "tedlium_id": tedlium_normalized[key],
                "speaker": row["main_speaker"],
                "occupation": row.get("speaker_occupation", ""),
                "tags": row.get("tags", ""),
                **{t: row.get(t, "0") for t in TOPICS},
            })

    print(f"Matched {len(matched)} / {len(tedlium_names)} TED-LIUM speakers in TED-plus\n")

    # Filter by topic
    technical = [m for m in matched if any(m[t] == "1" for t in ["technology", "science", "health", "innovation"])]
    print(f"Speakers in technology/science/health/innovation: {len(technical)}\n")

    print(f"{'Speaker':<25} {'Occupation':<35} {'Topics'}")
    print("-" * 100)
    for m in sorted(technical, key=lambda x: x["speaker"]):
        active = [t for t in TOPICS if m[t] == "1"]
        print(f"{m['speaker']:<25} {m['occupation'][:34]:<35} {', '.join(active)}")

    # Save filtered list
    out = "/Users/xmena/Desktop/speakers_technical.txt"
    with open(out, "w") as f:
        for m in technical:
            f.write(m["tedlium_id"] + "\n")
    print(f"\nSaved {len(technical)} speaker IDs → {out}")


if __name__ == "__main__":
    main()
