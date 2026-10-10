"""Build an evidence walkthrough; heavy experiments remain explicit CLI jobs."""

import subprocess
import sys
from pathlib import Path

import nbformat as nbf

root = Path(__file__).resolve().parents[1]
nb = nbf.v4.new_notebook()
md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell
nb.cells = [
    md(
        "# AccordLens: Preference Alignment & Judge Reliability\n\nAn executed walkthrough of measured Qwen adaptation, human/GPT-4 evaluator reliability, and bilingual support token costs. This notebook reads released evidence and verifies core mathematics without model downloads or retraining."
    ),
    code("""import json
from pathlib import Path

import pandas as pd
import torch
from IPython.display import Image, display

from accordlens.objectives import dpo_loss, gae, orpo_loss

ROOT = Path.cwd()
if not (ROOT / "configs").exists():
    ROOT = ROOT.parent

def read(path):
    return json.loads((ROOT / path).read_text())

summary = read("reports/experiment_summary.json")
summary["config"]"""),
    md(
        "## Data isolation\n\nFull dialogue history is preserved. Eligibility rejects overlength sequences rather than truncating away the response. Train/validation come from the official training source; test comes from the official test source. This favors short conversations and limits generalization."
    ),
    code("""manifest = pd.DataFrame(read("data/split_manifest.json"))
assert manifest["id"].nunique() == len(manifest)
assert manifest["prompt_id"].nunique() == len(manifest)
display(manifest.groupby("split").size().rename("pairs"))
read("reports/source_verification.json")"""),
    md(
        "## Held-out policy diagnostics\n\nMean-token likelihood ranking and response NLL are descriptive metrics. They do not prove generated-answer helpfulness. DPO optimizes reference-relative summed log probabilities; ORPO uses mean-token odds. The small run establishes no preference-ranking gain."
    ),
    code("""display(pd.DataFrame(summary["test"]).T)
display(Image(filename=str(ROOT / "reports/figures/preference_quality.png")))
read("reports/paired_comparisons.json")"""),
    md(
        "## Objective sanity checks\n\nAt identical policy/reference likelihoods, DPO loss equals log(2). ORPO retains gradients even for unlikely responses. GAE gives the terminal reward to earlier tokens when gamma=lambda=1 and the critic is zero."
    ),
    code("""chosen = torch.tensor([-4.0], requires_grad=True)
rejected = torch.tensor([-5.0], requires_grad=True)
loss, margin = dpo_loss(chosen, rejected, chosen.detach(), rejected.detach())
loss.backward()
print("DPO initial loss:", loss.item(), "gradient signs:", chosen.grad.item(), rejected.grad.item())
assert chosen.grad.item() < 0 < rejected.grad.item()
low_chosen = torch.tensor([-30.0], requires_grad=True)
low_rejected = torch.tensor([-35.0], requires_grad=True)
orpo, _, _ = orpo_loss(low_chosen, low_rejected)
orpo.backward()
assert low_rejected.grad.item() != 0
advantages, returns = gae(torch.tensor([0.0, 0.0, 2.0]), torch.zeros(3), lam=1)
assert torch.equal(returns, torch.tensor([2.0, 2.0, 2.0]))
returns"""),
    md(
        "## Reward generalization and PPO\n\nThe reward head uses frozen SFT representations and validation checkpoint selection. Its test interval spans chance. PPO uses exact rollout density, fixed old probabilities, multiple clipped updates, a critic and GAE. Its small rollout budget cannot establish robust alignment."
    ),
    code("""display(Image(filename=str(ROOT / "reports/figures/reward_model.png")))
display(pd.DataFrame(read("reports/ppo_rollouts.json")))
summary["reward_test"]"""),
    code("""display(Image(filename=str(ROOT / "reports/figures/ppo_diagnostics.png")))
history = pd.DataFrame(read("reports/training_history.json"))
display(history[history["method"] == "ppo"])
assert summary["reference_adapter_unchanged"]"""),
    md(
        "## Human labels are different from model labels\n\nThe source spreadsheet used GPT-4 pair labels. Genuine human annotations are evaluated separately. Invalid outputs count as failures, never as ties. Answer-order swaps expose weak evaluator reliability."
    ),
    code("""display(pd.DataFrame(read("reports/judge_summary.json")).T)
display(Image(filename=str(ROOT / "reports/figures/judge_reliability.png")))
judge_rows = pd.DataFrame(read("reports/judge_predictions.json"))
display(judge_rows[["source", "label", "raw_output", "verdict", "swapped_raw_output", "swapped_verdict"]])"""),
    md(
        "## Bilingual retail-support token costs\n\nSix authored parallel messages provide a compact tokenizer comparison. Fewer tokens may reduce context costs; they do not prove better answers. These cases are not a retail adaptation benchmark."
    ),
    code("""tokens = pd.DataFrame(read("reports/tokenizer_audit.json"))
display(tokens.groupby(["model", "language"])[["tokens", "characters_per_token"]].mean())
display(Image(filename=str(ROOT / "reports/figures/bilingual_token_cost.png")))"""),
    md(
        "## Artifact portability and decision\n\nThe exported DPO adapter is independently reloaded and compared against recorded held-out likelihoods. The engineering pipeline is reproducible, but these results do not authorize a production model or automated judge release."
    ),
    code("""verification = read("reports/reload_verification.json")
assert verification["passed"]
verification"""),
]
nb.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.12"},
}
path = root / "notebooks/decision_audit.ipynb"
nbf.write(nb, path)
subprocess.run([sys.executable, "-m", "ruff", "format", str(path)], check=True)
