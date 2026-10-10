"""Measure bilingual support-message token costs with tokenizer-only assets."""

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--llama2", required=True)
    p.add_argument("--llama3", required=True)
    args = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    rows = []
    for model, path in [("Llama 2", args.llama2), ("Llama 3", args.llama3)]:
        tokenizer = AutoTokenizer.from_pretrained(
            path, local_files_only=True, use_fast=True
        )
        for case in json.loads((root / "data/bilingual_cases.json").read_text()):
            for language in ["English", "Persian"]:
                text = case[language]
                tokens = tokenizer.encode(text, add_special_tokens=False)
                rows.append(
                    {
                        "model": model,
                        "topic": case["topic"],
                        "language": language,
                        "characters": len(text),
                        "tokens": len(tokens),
                        "characters_per_token": len(text) / len(tokens),
                        "vocabulary_size": len(tokenizer),
                    }
                )
    (root / "reports/tokenizer_audit.json").write_text(
        json.dumps(rows, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
