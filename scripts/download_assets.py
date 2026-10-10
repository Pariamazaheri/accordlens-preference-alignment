"""Fetch immutable public assets; large base weights remain outside Git."""

import argparse
import gzip
import hashlib
import json
import urllib.request
from pathlib import Path

from huggingface_hub import snapshot_download

HH_REV = "09be8c5bbc57cb3887f3a9732ad6aa7ec602a1fa"
MT_REV = "f7d2896d2cc5d80f8b55c2bbc722613555233c25"
FILES = {
    "human.parquet": (
        "human-00000-of-00001-25f4910818759289.parquet",
        "4877bc46a40929f4082c3c79593700fb897b1d6c7f4c473032694a01322f5769",
    ),
    "gpt4.parquet": (
        "gpt4_pair-00000-of-00001-c0b431264a82ddc0.parquet",
        "57068327ffb5f8fddb15fedbbfe40b26fe862da2326331e5cf64a42320a00e71",
    ),
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", action="store_true")
    args = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    raw = root / "data/raw"
    raw.mkdir(parents=True, exist_ok=True)
    for name, (filename, checksum) in FILES.items():
        path = raw / name
        urllib.request.urlretrieve(
            f"https://huggingface.co/datasets/lmsys/mt_bench_human_judgments/resolve/{MT_REV}/data/{filename}",
            path,
        )
        if hashlib.sha256(path.read_bytes()).hexdigest() != checksum:
            raise ValueError("Dataset checksum mismatch")
    for split in ["train", "test"]:
        with (raw / f"{split}.jsonl").open("wb") as dest:
            for subset in [
                "harmless-base",
                "helpful-base",
                "helpful-online",
                "helpful-rejection-sampled",
            ]:
                path = raw / f"{subset}-{split}.jsonl.gz"
                urllib.request.urlretrieve(
                    f"https://huggingface.co/datasets/Anthropic/hh-rlhf/resolve/{HH_REV}/{subset}/{split}.jsonl.gz",
                    path,
                )
                with gzip.open(path, "rb") as stream:
                    for line in stream:
                        dest.write(line)
    if args.model:
        config = json.loads((root / "configs/experiment.json").read_text())
        snapshot_download(
            config["model_id"],
            revision=config["model_revision"],
            local_dir=root / "models/qwen",
        )


if __name__ == "__main__":
    main()
