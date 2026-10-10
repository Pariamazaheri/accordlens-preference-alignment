"""Generate figures, paired uncertainty, and the recruiter-facing evidence report."""

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def read(name):
    return json.loads((ROOT / name).read_text())


def main():
    summary = read("reports/experiment_summary.json")
    judge = read("reports/judge_summary.json")
    rows = pd.DataFrame(read("reports/preference_predictions.json"))
    hist = pd.DataFrame(read("reports/training_history.json"))
    names = ["base", "sft", "dpo", "orpo", "ppo"]
    colors = ["#64748b", "#0284c7", "#7c3aed", "#0d9488", "#e87924"]
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.titlesize": 12,
        }
    )
    out = ROOT / "reports/figures"
    out.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    acc = np.array([summary["test"][n]["mean_logp_accuracy"] for n in names])
    lo = np.array([summary["test"][n]["ci95"][0] for n in names])
    hi = np.array([summary["test"][n]["ci95"][1] for n in names])
    axes[0].bar(names, acc, color=colors)
    axes[0].errorbar(
        names, acc, yerr=[acc - lo, hi - acc], fmt="none", ecolor="#172554", capsize=4
    )
    axes[0].axhline(0.5, color="#94a3b8", ls="--")
    axes[0].set(
        ylim=(0, 1),
        ylabel="Preferred response ranking accuracy",
        title="Held-out pairs · n=32 · Wilson 95% intervals",
    )
    axes[1].bar(names, [summary["test"][n]["chosen_nll"] for n in names], color=colors)
    axes[1].set(
        ylabel="Chosen response mean token NLL",
        title="Better fit does not establish preference gains",
    )
    fig.savefig(out / "preference_quality.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), layout="constrained")
    for ax, name, c in zip(axes, ["sft", "dpo", "orpo"], colors[1:4]):
        group = hist[hist.method == name]
        ax.plot(group.step, group.loss, color=c)
        ax.set(title=name.upper(), xlabel="Optimizer update", ylabel="Objective loss")
    fig.suptitle(
        "Different objectives have different scales; compare within each panel"
    )
    fig.savefig(out / "training_diagnostics.png", dpi=180)
    plt.close(fig)
    rewards = pd.DataFrame(read("reports/reward_predictions.json"))
    rmhist = pd.DataFrame(read("reports/reward_history.json"))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    axes[0].plot(rmhist.step, rmhist.loss, label="Train + regularization")
    axes[0].plot(rmhist.step, rmhist.validation_loss, label="Validation BT")
    axes[0].axvline(
        read("artifacts/reward_metadata.json")["best_validation_epoch"],
        ls="--",
        color="#64748b",
    )
    axes[0].legend()
    axes[0].set(
        xlabel="Reward-head update",
        ylabel="Loss",
        title="Validation selects an early reward checkpoint",
    )
    test = rewards[rewards.split == "test"]
    axes[1].scatter(test.rejected, test.chosen, color="#0d9488")
    limits = [
        min(test.chosen.min(), test.rejected.min()),
        max(test.chosen.max(), test.rejected.max()),
    ]
    axes[1].plot(limits, limits, color="#94a3b8", ls="--")
    axes[1].set(
        xlabel="Rejected response raw reward",
        ylabel="Chosen response raw reward",
        title=f"Held-out reward ranking: {summary['reward_test']['pairwise_accuracy']:.1%}",
    )
    fig.savefig(out / "reward_model.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), layout="constrained")
    for i, source in enumerate(["human", "gpt4_reference"]):
        s = judge[source]
        axes[0].bar(i - 0.17, s["agreement"], width=0.34, color="#0284c7")
        axes[0].bar(i + 0.17, s["majority_baseline"], width=0.34, color="#94a3b8")
        axes[0].errorbar(
            i - 0.17,
            s["agreement"],
            yerr=[
                [s["agreement"] - s["agreement_ci95"][0]],
                [s["agreement_ci95"][1] - s["agreement"]],
            ],
            fmt="none",
            ecolor="#172554",
            capsize=4,
        )
    axes[0].set(
        xticks=[0, 1],
        xticklabels=["Human labels", "GPT-4 reference"],
        ylim=(0, 1),
        ylabel="Agreement rate",
        title="Qwen judge vs majority-label baseline",
    )
    from matplotlib.patches import Patch

    axes[0].legend(
        handles=[
            Patch(color="#0284c7", label="Judge"),
            Patch(color="#94a3b8", label="Majority label"),
        ]
    )
    for i, source in enumerate(["human", "gpt4_reference"]):
        s = judge[source]
        axes[1].bar(i - 0.17, s["valid_rate"], width=0.34, color="#0d9488")
        axes[1].bar(i + 0.17, s["swap_consistency"], width=0.34, color="#e87924")
    axes[1].set(
        xticks=[0, 1],
        xticklabels=["Human labels", "GPT-4 reference"],
        ylim=(0, 1),
        ylabel="Rate",
        title="Valid format is insufficient for reliable judgment",
    )
    axes[1].legend(
        handles=[
            Patch(color="#0d9488", label="Valid verdict"),
            Patch(color="#e87924", label="Swap consistent · valid pairs"),
        ]
    )
    fig.savefig(out / "judge_reliability.png", dpi=180)
    plt.close(fig)
    tokens = pd.DataFrame(read("reports/tokenizer_audit.json"))
    fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
    for i, model in enumerate(["Llama 2", "Llama 3"]):
        means = [
            tokens[
                (tokens.model == model) & (tokens.language == language)
            ].tokens.mean()
            for language in ["English", "Persian"]
        ]
        ax.bar(
            np.arange(2) + (i - 0.5) * 0.34,
            means,
            width=0.34,
            label=model,
            color=["#64748b", "#0284c7"][i],
        )
    ax.set(
        xticks=[0, 1],
        xticklabels=["English", "Persian"],
        ylabel="Mean tokens per support message",
        title="Six authored bilingual retail support messages",
    )
    ax.legend()
    fig.savefig(out / "bilingual_token_cost.png", dpi=180)
    plt.close(fig)
    ppo = hist[hist.method == "ppo"]
    rollout = pd.DataFrame(read("reports/ppo_rollouts.json"))
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), layout="constrained")
    axes[0].plot(
        rollout.rollout, rollout.normalized_reward, marker="o", color="#0d9488"
    )
    axes[0].set(
        xlabel="Training rollout",
        ylabel="Normalized terminal reward",
        title="Six short rollouts · no quality claim",
    )
    axes[1].plot(ppo.step, ppo.clip_fraction, color="#e87924", label="Clipped tokens")
    axes[1].set(
        xlabel="PPO epoch update",
        ylabel="Clip fraction",
        title="Fixed rollout density across epochs",
        ylim=(-0.05, 1.05),
    )
    axes[2].plot(ppo.step, ppo.value_loss, color="#7c3aed")
    axes[2].set(
        xlabel="PPO epoch update",
        ylabel="Critic loss",
        title="Learned token-value baseline",
    )
    fig.savefig(out / "ppo_diagnostics.png", dpi=180)
    plt.close(fig)
    rng = np.random.default_rng(42)
    paired = {}
    testrows = rows[rows.split == "test"]
    sft = testrows[testrows.method == "sft"].set_index("id")
    for name in ["dpo", "orpo", "ppo"]:
        stage = testrows[testrows.method == name].set_index("id").loc[sft.index]
        diff = (stage.mean_logp_margin.to_numpy() > 0).astype(float) - (
            sft.mean_logp_margin.to_numpy() > 0
        ).astype(float)
        boot = diff[rng.integers(0, len(diff), size=(10000, len(diff)))].mean(1)
        paired[name] = {
            "accuracy_difference_vs_sft": float(diff.mean()),
            "paired_bootstrap_ci95": np.quantile(boot, [0.025, 0.975]).tolist(),
            "resamples": 10000,
            "seed": 42,
        }
    (ROOT / "reports/paired_comparisons.json").write_text(
        json.dumps(paired, indent=2) + "\n"
    )
    table = "\n".join(
        f"| {n.upper()} | {summary['test'][n]['mean_logp_accuracy']:.2%} | {summary['test'][n]['ci95'][0]:.1%}–{summary['test'][n]['ci95'][1]:.1%} | {summary['test'][n]['chosen_nll']:.3f} | {summary['test'][n]['sum_logp_accuracy']:.2%} |"
        for n in names
    )
    (
        ROOT / "reports/benchmark.md"
    ).write_text(f"""# Preference learning and evaluator reliability

This is a measured engineering benchmark with a deliberately small CPU budget. It does not establish production model quality. The base is already instruction-tuned Qwen, so these stages perform additional adaptation rather than alignment from a raw pretrained model.

## Held-out preference evidence

| Policy | Mean-token ranking | 95% Wilson interval | Chosen token NLL ↓ | Sum-token ranking |
|---|---:|---:|---:|---:|
{table}

All methods use the same 32 held-out pairs. Mean-token ranking compares normalized response likelihoods; sum-token ranking is supplied as a length-sensitive sensitivity check. Neither is an evaluation of generated-answer helpfulness or safety. DPO optimizes a **reference-relative sum-log-probability margin**, which differs from this descriptive likelihood ranking. Do not treat raw rewards, DPO implicit margins, ORPO odds, or these likelihood scores as interchangeable scales.

SFT improves chosen-response fit, from 2.853 to 2.324 NLL, while ranking stays at 53.13%. DPO reaches 50%; ORPO and PPO reach 53.13%. No measured preference gain is established. Paired bootstrap intervals are recorded in `paired_comparisons.json`; all include zero. One seed, short complete conversations, and 32 test pairs limit conclusions. No method was selected using the test set.

## Reward model and PPO

A frozen SFT backbone supplies final-token representations to an 896-parameter linear Bradley–Terry head. The validation loss selects update 17 from 200 updates; the test ranking is {summary["reward_test"]["pairwise_accuracy"]:.2%}, interval {summary["reward_test"]["ci95"][0]:.1%}–{summary["reward_test"]["ci95"][1]:.1%}. This interval includes chance. It is an economical reward-model architecture and may miss nuanced preference distinctions. Raw reward logits have no universal 0.5 threshold.

PPO uses six sampled training prompts, at most 24 response tokens, a learned token-value critic, GAE (gamma=1, lambda=.95), sampled reference-KL reward shaping (coefficient .02), fixed rollout log probabilities, and up to three clipped optimization epochs per rollout. Generation samples the unmodified policy distribution (temperature=1, top_p=1, top_k=0). Reward normalization uses train-only statistics and clips to ±5. The reference adapter is verified unchanged. No sampled token crossed the 0.2 clipping boundary during this recorded run, so clipping implementation is demonstrated by tests rather than an observed clipped update. Reward inputs are decoded and reformatted with the same native chat terminator as reward-training completions. Single-trajectory batches and short rollouts make this a mechanics benchmark, not a robust RLHF quality study. Sampled KL may be negative; it is an estimator, not the exact full-vocabulary KL.

## Judge evidence

The supplied export matches the first 1,000 rows of the official **GPT-4 pair judgments**, not human annotations. Genuine human labels come from the separate human split. Each source contributes 12 distinct questions, complete answers, two answer orders, greedy decoding, and an eight-token output budget. No input is silently truncated; cases exceeding 1,536 tokens are excluded before sampling. Invalid outputs count as agreement failures and are never converted to ties. The short output budget may itself cause invalid answers.

| Reference | Agreement | Majority baseline | Valid verdicts | Swap consistency among valid pairs |
|---|---:|---:|---:|---:|
| Human | {judge["human"]["agreement"]:.1%} | {judge["human"]["majority_baseline"]:.1%} | {judge["human"]["valid_rate"]:.1%} | {judge["human"]["swap_consistency"]:.1%} ({judge["human"]["swap_valid_pairs"]} pairs) |
| GPT-4 | {judge["gpt4_reference"]["agreement"]:.1%} | {judge["gpt4_reference"]["majority_baseline"]:.1%} | {judge["gpt4_reference"]["valid_rate"]:.1%} | {judge["gpt4_reference"]["swap_consistency"]:.1%} ({judge["gpt4_reference"]["swap_valid_pairs"]} pairs) |

The majority baseline uses the evaluation label distribution as a descriptive comparator, not a fitted deployable classifier. Human annotations can be subjective; agreement is not factual correctness. Swap inconsistency is evidence of order sensitivity, but does not isolate positional bias from all other model limitations. This judge should not gate customer-facing changes autonomously.

## Bilingual cost diagnostics

Six authored parallel retail support messages compare Llama 2 and Llama 3 token counts without loading their model weights. Characters per token measures Python Unicode code points, not bytes or grapheme clusters. Results describe these messages only; vocabulary size alone does not explain all differences, and fewer tokens do not prove better answer quality.

## Reproduction budget

Rank-4 query/value adapters train {summary["trainable_policy_parameters"]:,} parameters ({summary["trainable_policy_parameters"] / summary["total_policy_parameters"]:.4%} including the added adapter in the denominator). SFT, DPO, and ORPO each perform 20 AdamW updates with two examples per update, using the same shuffled order. Thus 40 of 48 training pairs contribute to each short run; all 48 contribute to reward training. DPO/PPO begin at SFT; ORPO begins at the instruction base and jointly learns chosen responses and preferences. Budgets differ by objective and are not compute-matched. Checkpoint policies are fixed final updates for policy stages and validation-selected for the reward head. Full stage details are in the configuration and row-level JSON evidence. Total experiment wall time was {summary["total_seconds"]:.1f} seconds in the recorded local runtime; hardware and library versions affect reproducibility.
""")


if __name__ == "__main__":
    main()
