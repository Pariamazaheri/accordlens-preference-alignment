"""Prepare supplied exports without modifying them; also accepts official JSONL."""

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
from transformers import AutoTokenizer

from accordlens.data import encode_response, preference_pair


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_rows(path):
    if path.suffix == ".arrow":
        with pa.memory_map(str(path)) as stream:
            return pa.ipc.open_stream(stream).read_all().to_pylist()
    return [json.loads(line) for line in path.read_text().splitlines()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--human", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    out = root / "data/processed"
    out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    pools = {}
    diagnostics = {}
    sources = {}
    for split, path in [("train", args.train), ("test", args.test)]:
        rows = load_rows(path)
        pool = {}
        bad = 0
        for row in rows:
            try:
                p = preference_pair(row["chosen"], row["rejected"])
                pool.setdefault(p["id"], p)
            except ValueError:
                bad += 1
        pools[split] = pool
        diagnostics[split] = {
            "source_rows": len(rows),
            "unique_valid_pairs": len(pool),
            "invalid_pairs": bad,
        }
        sources[split] = {"sha256": sha(path), "format": path.suffix}
    cross = {p["prompt_id"] for p in pools["train"].values()} & {
        p["prompt_id"] for p in pools["test"].values()
    }
    selected = {}
    seen = set()
    rejected = {}
    for split, need in [("train", 64), ("test", 32)]:
        selected[split] = []
        skipped = 0
        for key, p in sorted(pools[split].items()):
            if p["prompt_id"] in cross or p["prompt_id"] in seen:
                continue
            try:
                for name in ["chosen", "rejected"]:
                    encode_response(tok, p["messages"], p[name], 256)
            except ValueError:
                skipped += 1
                continue
            selected[split].append(p)
            seen.add(p["prompt_id"])
            if len(selected[split]) == need:
                break
        if len(selected[split]) != need:
            raise ValueError("Insufficient complete short pairs")
        rejected[split] = skipped
    splits = {
        "train": selected["train"][:48],
        "validation": selected["train"][48:],
        "test": selected["test"],
    }
    manifest = []
    for split, rows in splits.items():
        (out / f"{split}.json").write_text(json.dumps(rows, ensure_ascii=False))
        manifest.extend(
            {"split": split, "id": p["id"], "prompt_id": p["prompt_id"]} for p in rows
        )
    (root / "data/split_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    judge = []
    human = pd.read_parquet(args.human)
    reference = (
        pd.read_parquet(args.reference).iloc[:1000]
        if args.reference.suffix == ".parquet"
        else pd.read_excel(args.reference)
    )
    for source, frame in [("human", human), ("gpt4_reference", reference)]:
        frame = frame[frame.turn == 1].sample(frac=1, random_state=42)
        ids = set()
        count = 0
        for index, row in frame.iterrows():
            # Unique question per source limits pseudo-replication.
            if int(row.question_id) in ids:
                continue
            a = row.conversation_a
            b = row.conversation_b
            if isinstance(a, str):
                a = json.loads(a)
            if isinstance(b, str):
                b = json.loads(b)
            a = list(a)
            b = list(b)
            q = a[0]["content"]
            aa = a[1]["content"]
            bb = b[1]["content"]
            if q != b[0]["content"]:
                raise ValueError("Judge question mismatch")
            from accordlens.judging import judge_prompt, normalize_label

            text = judge_prompt(q, aa, bb)
            token_ids = tok.apply_chat_template(
                [{"role": "user", "content": text}],
                tokenize=True,
                add_generation_prompt=True,
            )
            if len(token_ids) > 1536:
                continue
            judge.append(
                {
                    "id": f"{source}:{index}",
                    "question_id": int(row.question_id),
                    "source": source,
                    "annotator": str(row.judge),
                    "label": normalize_label(row.winner),
                    "original_label": str(row.winner),
                    "question": q,
                    "a": aa,
                    "b": bb,
                    "input_tokens": len(token_ids),
                }
            )
            ids.add(int(row.question_id))
            count += 1
            if count == 12:
                break
        if count != 12:
            raise ValueError("Insufficient judge cases")
    (out / "judge.json").write_text(json.dumps(judge, ensure_ascii=False))
    profile = {
        "sources": sources,
        "diagnostics": diagnostics,
        "cross_split_prompt_count": len(cross),
        "length_rejections_before_selection": rejected,
        "selection": "Ascending SHA256 pair identifier; first eligible unique prompts after cross-source prompt exclusion, complete native chat sequences <=256 tokens",
        "human_parquet_sha256": sha(args.human),
        "reference_xlsx_sha256": sha(args.reference),
        "human_dataset_revision": "f7d2896d2cc5d80f8b55c2bbc722613555233c25",
        "judge_reference_annotators": reference.judge.value_counts().to_dict(),
    }
    (root / "data/profile.json").write_text(json.dumps(profile, indent=2) + "\n")
    print(json.dumps(profile, indent=2))


if __name__ == "__main__":
    main()
