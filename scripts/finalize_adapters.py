"""Sanitize adapter metadata and verify independent reload on held-out pairs."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from accordlens.data import encode_response
from accordlens.experiment import logps, write


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    args = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "configs/experiment.json").read_text())
    for stage in ["sft", "dpo", "orpo", "ppo"]:
        path = root / f"artifacts/{stage}/adapter_config.json"
        a = json.loads(path.read_text())
        a["base_model_name_or_path"] = config["model_id"]
        a["revision"] = config["model_revision"]
        write(path, a)
        (root / f"artifacts/{stage}/README.md").write_text(
            f"# {stage.upper()} adapter\n\nExperimental rank-4 Qwen adapter. See the [model card](../MODEL_CARD.md) for protocol, measured results, and limitations. Base weights are not included. Derived weights follow the Apache 2.0 base-model license.\n"
        )
    torch.set_num_threads(2)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=True,
        dtype=torch.float32,
        attn_implementation="eager",
    )
    model = PeftModel.from_pretrained(model, root / "artifacts/dpo").eval()
    rows = json.loads((root / "data/processed/test.json").read_text())[:2]
    predictions = json.loads((root / "reports/preference_predictions.json").read_text())
    differences = []
    with torch.no_grad():
        for row in rows:
            recorded = next(
                r
                for r in predictions
                if r["method"] == "dpo"
                and r["split"] == "test"
                and r["id"] == row["id"]
            )
            for key in ["chosen", "rejected"]:
                encoded = encode_response(tokenizer, row["messages"], row[key], 256)
                actual, _ = logps(model, encoded)
                differences.append(abs(actual.item() - recorded[f"{key}_logp"]))
    assert max(differences) < 1e-4
    write(
        root / "reports/reload_verification.json",
        {
            "adapter": "dpo",
            "heldout_pairs": len(rows),
            "max_absolute_logp_difference": max(differences),
            "passed": True,
        },
    )
    checksums = {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted((root / "artifacts").rglob("*"))
        if path.is_file()
    }
    write(root / "reports/artifact_checksums.json", checksums)


if __name__ == "__main__":
    main()
