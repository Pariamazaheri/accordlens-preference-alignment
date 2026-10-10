"""Recompute published metrics from row-level evidence without model downloads."""

import hashlib
import json
from pathlib import Path

import numpy as np

from .judging import summarize, wilson


def validate(root):
    root = Path(root)

    def read(name):
        return json.loads((root / name).read_text())

    manifest = read("data/split_manifest.json")
    assert len(manifest) == 96
    assert len({r["id"] for r in manifest}) == 96
    assert len({r["prompt_id"] for r in manifest}) == 96
    expected = {
        s: {r["id"] for r in manifest if r["split"] == s}
        for s in ["train", "validation", "test"]
    }
    rows = read("reports/preference_predictions.json")
    summary = read("reports/experiment_summary.json")
    assert len(rows) == 5 * 48
    for method in ["base", "sft", "dpo", "orpo", "ppo"]:
        for split in ["validation", "test"]:
            group = [r for r in rows if r["method"] == method and r["split"] == split]
            assert {r["id"] for r in group} == expected[split]
            for r in group:
                assert r["chosen_tokens"] > 0 and r["rejected_tokens"] > 0
                margin = (
                    r["chosen_logp"] / r["chosen_tokens"]
                    - r["rejected_logp"] / r["rejected_tokens"]
                )
                assert abs(margin - r["mean_logp_margin"]) < 1e-5
        test = [r for r in rows if r["method"] == method and r["split"] == "test"]
        wins = sum(r["mean_logp_margin"] > 0 for r in test)
        assert summary["test"][method]["mean_logp_accuracy"] == wins / 32
        assert np.allclose(summary["test"][method]["ci95"], wilson(wins, 32))
    rewards = read("reports/reward_predictions.json")
    test = [r for r in rewards if r["split"] == "test"]
    assert len(test) == 32
    assert (
        summary["reward_test"]["pairwise_accuracy"]
        == sum(r["chosen"] > r["rejected"] for r in test) / 32
    )
    judge = read("reports/judge_predictions.json")
    js = read("reports/judge_summary.json")
    for source in ["human", "gpt4_reference"]:
        calculated = summarize([r for r in judge if r["source"] == source])
        for key, value in calculated.items():
            if isinstance(value, float):
                assert np.isclose(value, js[source][key], rtol=1e-10, atol=1e-10)
            elif isinstance(value, list):
                assert np.allclose(value, js[source][key], rtol=1e-10, atol=1e-10)
            else:
                assert value == js[source][key]
    assert summary["reference_adapter_unchanged"]
    for name, checksum in read("reports/artifact_checksums.json").items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == checksum
    for stage in ["sft", "dpo", "orpo", "ppo"]:
        config = read(f"artifacts/{stage}/adapter_config.json")
        assert config["base_model_name_or_path"] == summary["config"]["model_id"]
        assert config["revision"] == summary["config"]["model_revision"]
    notebook = read("notebooks/decision_audit.ipynb")
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is not None
            assert not any(o["output_type"] == "error" for o in cell["outputs"])
    print(
        "Evidence, split isolation, portable adapters, and executed notebook validated."
    )
