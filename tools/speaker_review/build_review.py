"""Build a speaker-labelling bundle from disagreements between diarization systems.

For each recording it reads our pipeline's verbose_json segments (with word
timestamps) and the turns of every diarization system found next to the audio
(``<audio>_turns.json`` = pyannote, the label space we score in, plus
``<audio>_turns_<system>.json``). Other systems' labels are mapped onto pyannote's
by maximum overlap. Clips are cut where systems disagree with our word labels, plus
random speaker changes and random speech windows for an unbiased error estimate.
Random windows may overlap the targeted ones (so a word can be labelled twice), because
excluding the targeted regions would bias the random sample towards easy speech.

Output: ``<out>/data.js`` + ``<out>/clips/*.mp3`` + ``index.html``; open index.html
in a browser and export ``<audio>_speaker_annotations.json`` from there.

    uv run python tools/speaker_review/build_review.py \
        --segments-dir <dir with *_m4a_segments.json> --out tests/assets/internal/speaker_review \
        "FM PO Meister.m4a" "FM PO Wi & Cl.m4a"
"""

import argparse
import itertools
import json
import random
import shutil
import subprocess
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

INTERNAL = Path(__file__).resolve().parents[2] / "tests" / "assets" / "internal"
CONTEXT_S = 2.0
MAX_CLIP_S = 16.0
MERGE_GAP_S = 1.5
REF_CLIP_S = 6.0


def stem(audio: str) -> str:
    return audio.replace(".", "_")


def overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def load_turns(path: Path) -> list[tuple[float, float, str]]:
    return [(float(a), float(b), str(s)) for a, b, s in json.loads(path.read_text()) if b > a]


def label_mapping(system: list, reference: list) -> dict[str, str]:
    """One-to-one by maximum total overlap per group (vibevoice labels are chunk-local
    'c<k>_...'), leftover labels to their best-overlapping reference speaker."""
    ref_labels = sorted({s for *_, s in reference})
    groups: dict[str, list[str]] = defaultdict(list)
    for label in sorted({s for *_, s in system}):
        groups[label.split("_S")[0] if label.startswith("c") and "_S" in label else ""].append(label)

    mapping: dict[str, str] = {}
    for labels in groups.values():
        matrix = np.zeros((len(labels), len(ref_labels)))
        for a0, a1, s in system:
            if s not in labels:
                continue
            for b0, b1, r in reference:
                if b0 < a1 and b1 > a0:
                    matrix[labels.index(s), ref_labels.index(r)] += overlap(a0, a1, b0, b1)
        rows, cols = linear_sum_assignment(-matrix)
        for row, col in zip(rows, cols):
            if matrix[row, col] > 0:
                mapping[labels[row]] = ref_labels[col]
        for row, label in enumerate(labels):
            if label not in mapping:
                mapping[label] = ref_labels[int(matrix[row].argmax())] if matrix[row].max() > 0 else f"extra:{label}"
    return mapping


def label_at(turns: list, start: float, end: float) -> str | None:
    best, best_overlap = None, 0.0
    for a0, a1, s in turns:
        if a0 >= end:
            break
        o = overlap(a0, a1, start, end)
        if o > best_overlap:
            best, best_overlap = s, o
    return best


def cut(src: Path, start: float, end: float, dst: Path) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-ss",
            f"{start:.3f}",
            "-t",
            f"{end - start:.3f}",
            "-i",
            str(src),
            "-ac",
            "1",
            "-b:a",
            "96k",
            str(dst),
        ],
        check=True,
    )


def build_file(audio: str, segments_dir: Path, out: Path, counts: dict[str, int], rng: random.Random) -> dict | None:
    src = INTERNAL / audio
    reference = load_turns(INTERNAL / f"{stem(audio)}_turns.json")
    systems = {
        path.name.removeprefix(f"{stem(audio)}_turns_").removesuffix(".json"): load_turns(path)
        for path in sorted(INTERNAL.glob(f"{stem(audio)}_turns_*.json"))
    }
    mapped = {}
    for name, turns in systems.items():
        mapping = label_mapping(turns, reference)
        mapped[name] = sorted((a, b, mapping[s]) for a, b, s in turns)

    segments = json.loads((segments_dir / f"{stem(audio)}_segments.json").read_text())
    words = [
        {"s": w["start"], "e": w["end"], "w": w["word"].strip(), "spk": seg["speaker"], "seg": seg["id"]}
        for seg in segments
        for w in seg["words"] or []
        if w["end"] > w["start"]
    ]
    if not words:
        print(f"{audio}: no timestamped words, skipped")
        return None

    for word in words:
        word["hints"] = {name: label_at(turns, word["s"], word["e"]) for name, turns in mapped.items()}
        word["score"] = sum(1 for h in word["hints"].values() if h is not None and h != word["spk"])

    regions: list[list[dict]] = []
    for word in words:
        if word["score"] == 0:
            continue
        if regions and word["s"] - regions[-1][-1]["e"] < MERGE_GAP_S:
            regions[-1].append(word)
        else:
            regions.append([word])
    ranked = sorted(regions, key=lambda r: -sum(w["score"] * (w["e"] - w["s"]) for w in r))

    windows: list[tuple[float, float, str]] = []

    def free(start: float, end: float, origin: str) -> bool:
        is_random = origin.startswith("random")
        return all(end <= a or start >= b for a, b, o in windows if not is_random or o.startswith("random"))

    def add(start: float, end: float, origin: str) -> bool:
        start, end = max(0.0, start), min(end, start + MAX_CLIP_S)
        if not free(start, end, origin):
            return False
        windows.append((start, end, origin))
        return True

    taken = 0
    for region in ranked:
        if taken >= counts["disagreement"]:
            break
        mid = (region[0]["s"] + region[-1]["e"]) / 2
        start = max(region[0]["s"] - CONTEXT_S, mid - MAX_CLIP_S / 2)
        taken += add(start, min(region[-1]["e"] + CONTEXT_S, start + MAX_CLIP_S), "disagreement")

    changes = [b["s"] for a, b in itertools.pairwise(words) if a["spk"] != b["spk"]]
    rng.shuffle(changes)
    taken = 0
    for t in changes:
        if taken >= counts["random_change"]:
            break
        taken += add(t - 5.0, t + 5.0, "random_change")

    speech = [w["s"] for w in words]
    taken, attempts = 0, 0
    while taken < counts["random_any"] and attempts < 1000:
        attempts += 1
        t = rng.choice(speech)
        taken += add(t - 1.0, t + 9.0, "random_any")

    order = {"disagreement": [], "random_change": [], "random_any": []}
    for window in sorted(windows):
        order[window[2]].append(window)
    rng.shuffle(order["disagreement"])
    interleaved = []
    while any(order.values()):
        for origin, quota in (("disagreement", 3), ("random_change", 1), ("random_any", 1)):
            for _ in range(quota):
                if order[origin]:
                    interleaved.append(order[origin].pop(0))

    clip_dir = out / "clips"
    clips = []
    for index, (start, end, origin) in enumerate(interleaved):
        in_clip = [w for w in words if start <= (w["s"] + w["e"]) / 2 < end]
        if not in_clip:
            continue
        name = f"{stem(audio)}_{index:03d}.mp3"
        cut(src, start, end, clip_dir / name)
        clips.append(
            {
                "id": f"{stem(audio)}_{index:03d}",
                "origin": origin,
                "start": round(start, 3),
                "end": round(end, 3),
                "src": f"clips/{name}",
                "words": [
                    {"s": w["s"], "e": w["e"], "w": w["w"], "spk": w["spk"], "hints": w["hints"]} for w in in_clip
                ],
            }
        )

    speakers = []
    for label in sorted({s for *_, s in reference}):
        candidates = [
            (a, b)
            for a, b, s in reference
            if s == label
            and b - a >= REF_CLIP_S
            and all(label_at(turns, a, a + REF_CLIP_S) == label for turns in mapped.values())
        ]
        candidates.sort(key=lambda t: t[0] - t[1])
        refs = []
        for k, (a, _) in enumerate(candidates[:2]):
            name = f"{stem(audio)}_ref_{label}_{k}.mp3"
            cut(src, a + 0.5, a + 0.5 + REF_CLIP_S, clip_dir / name)
            refs.append(f"clips/{name}")
        talk = sum(b - a for a, b, s in reference if s == label)
        speakers.append({"label": label, "minutes": round(talk / 60, 1), "refs": refs})

    print(audio, {o: sum(c["origin"] == o for c in clips) for o in order}, "speakers", [s["label"] for s in speakers])
    return {"audio": audio, "systems": sorted(systems), "speakers": speakers, "clips": clips}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("audio", nargs="+")
    parser.add_argument("--segments-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=INTERNAL / "speaker_review")
    parser.add_argument("--disagreement", type=int, default=40)
    parser.add_argument("--random-change", type=int, default=15)
    parser.add_argument("--random-any", type=int, default=8)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    (args.out / "clips").mkdir(parents=True, exist_ok=True)
    counts = {"disagreement": args.disagreement, "random_change": args.random_change, "random_any": args.random_any}
    rng = random.Random(args.seed)
    files = [f for audio in args.audio if (f := build_file(audio, args.segments_dir, args.out, counts, rng))]
    if not files:
        raise SystemExit("no recording has timestamped words; nothing to review")
    (args.out / "data.js").write_text(
        "window.REVIEW = "
        + json.dumps({"build": datetime.now().strftime("%Y%m%dT%H%M%S"), "files": files}, ensure_ascii=False)
        + ";\n"
    )
    shutil.copy(Path(__file__).with_name("index.html"), args.out / "index.html")
    print("open", (args.out / "index.html").resolve())


if __name__ == "__main__":
    main()
