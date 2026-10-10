"""CPU-budgeted real-model experiment with independent preference objectives."""

import gc
import json
import time

import numpy as np
import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from .data import encode_response
from .judging import judge_prompt, parse_verdict, summarize, wilson
from .objectives import clipped_policy_loss, dpo_loss, gae, orpo_loss, response_logps


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def load_policy(path, adapter=None):
    base = AutoModelForCausalLM.from_pretrained(
        path, local_files_only=True, dtype=torch.float32, attn_implementation="eager"
    )
    base.config.use_cache = False
    if adapter:
        return PeftModel.from_pretrained(base, adapter, is_trainable=True)
    return get_peft_model(
        base,
        LoraConfig(
            r=4,
            lora_alpha=8,
            lora_dropout=0,
            target_modules=["q_proj", "v_proj"],
            task_type="CAUSAL_LM",
        ),
    )


def logps(model, encoded):
    ids, mask = encoded
    return response_logps(
        model(ids, attention_mask=torch.ones_like(ids)).logits, ids, mask
    )


def feature(model, encoded):
    ids, _ = encoded
    output = model(ids, attention_mask=torch.ones_like(ids), output_hidden_states=True)
    return output.hidden_states[-1][:, -1, :]


def train(root, model_path):
    config = json.loads((root / "configs/experiment.json").read_text())
    set_seed(config["seed"])
    torch.set_num_threads(2)
    tok = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    data = {
        s: json.loads((root / f"data/processed/{s}.json").read_text())
        for s in ["train", "validation", "test"]
    }
    encoded = {
        s: [
            {
                k: encode_response(tok, p["messages"], p[k], config["max_length"])
                for k in ["chosen", "rejected"]
            }
            for p in rows
        ]
        for s, rows in data.items()
    }
    all_history = []
    evaluations = []
    timings = {}
    start = time.time()

    def progress(stage, step):
        write(
            root / "reports/progress.json",
            {"stage": stage, "step": step, "elapsed_seconds": time.time() - start},
        )
        print(f"{stage}: {step}", flush=True)

    def evaluate(model, name):
        model.eval()
        with torch.no_grad():
            for split in ["validation", "test"]:
                for pair, e in zip(data[split], encoded[split]):
                    c, cn = logps(model, e["chosen"])
                    r, rn = logps(model, e["rejected"])
                    evaluations.append(
                        {
                            "method": name,
                            "split": split,
                            "id": pair["id"],
                            "chosen_logp": c.item(),
                            "rejected_logp": r.item(),
                            "chosen_tokens": int(cn.item()),
                            "rejected_tokens": int(rn.item()),
                            "mean_logp_margin": (c / cn - r / rn).item(),
                            "sum_logp_margin": (c - r).item(),
                        }
                    )

    model = load_policy(model_path)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    evaluate(model, "base")
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=config["learning_rate"]
    )
    rng = np.random.default_rng(42)
    order = rng.permutation(len(data["train"]))
    begin = time.time()
    for step in range(config["steps"]):
        model.eval()
        opt.zero_grad()
        loss = 0
        for j in range(2):
            e = encoded["train"][order[(2 * step + j) % len(order)]]
            c, n = logps(model, e["chosen"])
            part = -(c / n).mean() / 2
            part.backward()
            loss += part.item()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        opt.step()
        all_history.append({"method": "sft", "step": step + 1, "loss": loss})
        progress("sft", step + 1)
    timings["sft"] = time.time() - begin
    model.save_pretrained(root / "artifacts/sft")
    evaluate(model, "sft")
    # Cached SFT features: reward head training cannot alter the language model.
    features = {}
    model.eval()
    with torch.no_grad():
        for split, rows in encoded.items():
            features[split] = torch.cat(
                [
                    torch.cat(
                        [feature(model, e["chosen"]), feature(model, e["rejected"])]
                    )
                    for e in rows
                ]
            ).view(len(rows), 2, -1)
    dim = features["train"].shape[-1]
    reward_head = torch.nn.Linear(dim, 1, bias=False)
    torch.nn.init.zeros_(reward_head.weight)
    rm_opt = torch.optim.AdamW(reward_head.parameters(), lr=0.003, weight_decay=0.1)
    best = float("inf")
    state = None
    begin = time.time()
    rm_history = []
    for epoch in range(200):
        rm_opt.zero_grad()
        scores = reward_head(features["train"]).squeeze(-1)
        loss = (
            -torch.nn.functional.logsigmoid(scores[:, 0] - scores[:, 1]).mean()
            + 0.001 * scores.square().mean()
        )
        loss.backward()
        rm_opt.step()
        with torch.no_grad():
            val = reward_head(features["validation"]).squeeze(-1)
            vl = -torch.nn.functional.logsigmoid(val[:, 0] - val[:, 1]).mean().item()
        rm_history.append(
            {"step": epoch + 1, "loss": loss.item(), "validation_loss": vl}
        )
        if vl < best:
            best = vl
            state = {k: v.detach().clone() for k, v in reward_head.state_dict().items()}
            best_epoch = epoch + 1
    reward_head.load_state_dict(state)
    reward_head.eval()
    with torch.no_grad():
        train_scores = reward_head(features["train"]).squeeze(-1).flatten()
        center = train_scores.mean().item()
        scale = max(train_scores.std().item(), 1e-6)
        reward_records = []
        for split in ["validation", "test"]:
            scores = reward_head(features[split]).squeeze(-1)
            for p, s in zip(data[split], scores):
                reward_records.append(
                    {
                        "split": split,
                        "id": p["id"],
                        "chosen": s[0].item(),
                        "rejected": s[1].item(),
                    }
                )
    from safetensors.torch import save_file

    save_file(reward_head.state_dict(), str(root / "artifacts/reward_head.safetensors"))
    write(
        root / "artifacts/reward_metadata.json",
        {
            "hidden_size": dim,
            "center": center,
            "scale": scale,
            "best_validation_epoch": best_epoch,
            "backbone": "sft",
            "trainable_parameters": dim,
            "architecture": "Frozen SFT final-token hidden state plus linear Bradley-Terry head",
        },
    )
    timings["reward_head"] = time.time() - begin
    write(root / "reports/reward_history.json", rm_history)
    write(root / "reports/reward_predictions.json", reward_records)
    del model, features, opt
    gc.collect()
    for method in ["dpo", "orpo"]:
        # DPO starts at SFT; ORPO uses the same instruction base with no SFT stage.
        model = load_policy(
            model_path, root / "artifacts/sft" if method == "dpo" else None
        )
        ref = []
        if method == "dpo":
            model.eval()
            with torch.no_grad():
                for e in encoded["train"]:
                    c, _ = logps(model, e["chosen"])
                    r, _ = logps(model, e["rejected"])
                    ref.append((c.detach(), r.detach()))
        opt = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=config["learning_rate"],
        )
        begin = time.time()
        for step in range(config["steps"]):
            model.eval()
            opt.zero_grad()
            loss_value = 0
            for j in range(2):
                i = int(order[(2 * step + j) % len(order)])
                e = encoded["train"][i]
                c, cn = logps(model, e["chosen"])
                r, rn = logps(model, e["rejected"])
                if method == "dpo":
                    loss, margin = dpo_loss(c, r, *ref[i], beta=config["beta"])
                else:
                    loss, _, _ = orpo_loss(c / cn, r / rn, config["orpo_weight"])
                (loss / 2).backward()
                loss_value += loss.item() / 2
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            opt.step()
            all_history.append({"method": method, "step": step + 1, "loss": loss_value})
            progress(method, step + 1)
        timings[method] = time.time() - begin
        model.save_pretrained(root / f"artifacts/{method}")
        evaluate(model, method)
        del model, opt
        gc.collect()
    # Token-level PPO: exact rollout density, multiple epochs, critic and GAE.
    model = load_policy(model_path, root / "artifacts/sft")
    model.load_adapter(
        root / "artifacts/sft", adapter_name="reference", is_trainable=False
    )
    model.set_adapter("default")
    model.eval()
    ref_snapshot = {
        k: v.detach().clone()
        for k, v in model.state_dict().items()
        if ".reference." in k
    }
    value_head = torch.nn.Linear(dim, 1)
    torch.nn.init.zeros_(value_head.weight)
    torch.nn.init.zeros_(value_head.bias)
    policy_params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(
        policy_params + list(value_head.parameters()), lr=config["ppo_learning_rate"]
    )
    begin = time.time()
    rollouts = []
    for rollout in range(config["ppo_rollouts"]):
        p = data["train"][int(order[rollout])]
        prompt = tok.apply_chat_template(
            p["messages"],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        model.set_adapter("default")
        model.eval()
        with torch.no_grad():
            ids = model.generate(
                prompt,
                attention_mask=torch.ones_like(prompt),
                max_new_tokens=config["ppo_max_new_tokens"],
                do_sample=True,
                top_k=0,
                top_p=1.0,
                temperature=1.0,
                use_cache=True,
                pad_token_id=tok.eos_token_id,
            )
            mask = torch.zeros_like(ids)
            mask[:, prompt.shape[1] :] = 1
            indices = mask[:, 1:].bool()
            old_out = model(ids, output_hidden_states=True)
            old = (
                torch.nn.functional.log_softmax(old_out.logits[:, :-1].float(), -1)
                .gather(-1, ids[:, 1:, None])
                .squeeze(-1)[indices]
                .detach()
            )
            hidden = old_out.hidden_states[-1][:, :-1][indices]
            old_values = value_head(hidden).squeeze(-1).detach()
            model.set_adapter("reference")
            ref_out = model(ids, output_hidden_states=True)
            reference = (
                torch.nn.functional.log_softmax(ref_out.logits[:, :-1].float(), -1)
                .gather(-1, ids[:, 1:, None])
                .squeeze(-1)[indices]
                .detach()
            )
            response_text = tok.decode(
                ids[0, prompt.shape[1] :], skip_special_tokens=True
            )
            reward_encoded = encode_response(tok, p["messages"], response_text, 512)
            raw_reward = reward_head(feature(model, reward_encoded)).item()
            normalized_reward = float(np.clip((raw_reward - center) / scale, -5, 5))
            rewards = -config["ppo_kl_coefficient"] * (old - reference)
            rewards[-1] += normalized_reward
            advantages, returns = gae(rewards, old_values)
            if len(advantages) > 1:
                advantages = (advantages - advantages.mean()) / (
                    advantages.std(unbiased=False) + 1e-8
                )
        model.set_adapter("default")
        model.eval()
        for epoch in range(config["ppo_epochs"]):
            opt.zero_grad()
            out = model(ids, output_hidden_states=True)
            new = (
                torch.nn.functional.log_softmax(out.logits[:, :-1].float(), -1)
                .gather(-1, ids[:, 1:, None])
                .squeeze(-1)[indices]
            )
            values = value_head(out.hidden_states[-1][:, :-1][indices]).squeeze(-1)
            pg, clipfrac = clipped_policy_loss(new, old, advantages)
            clipped_values = old_values + (values - old_values).clamp(-0.2, 0.2)
            vl = (
                0.5
                * torch.maximum(
                    (values - returns).square(), (clipped_values - returns).square()
                ).mean()
            )
            loss = pg + 0.5 * vl
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                policy_params + list(value_head.parameters()), 1
            )
            opt.step()
            approximate_kl = ((new - old).exp() - 1 - (new - old)).mean().item()
            all_history.append(
                {
                    "method": "ppo",
                    "step": len([h for h in all_history if h["method"] == "ppo"]) + 1,
                    "rollout": rollout + 1,
                    "epoch": epoch + 1,
                    "loss": loss.item(),
                    "policy_loss": pg.item(),
                    "value_loss": vl.item(),
                    "clip_fraction": clipfrac.item(),
                    "old_policy_approx_kl": approximate_kl,
                }
            )
            if approximate_kl > 0.1:
                break
        rollouts.append(
            {
                "rollout": rollout + 1,
                "id": p["id"],
                "raw_reward": raw_reward,
                "normalized_reward": normalized_reward,
                "sampled_reference_kl": (old - reference).mean().item(),
                "response": tok.decode(
                    ids[0, prompt.shape[1] :], skip_special_tokens=True
                ),
                "response_tokens": int(indices.sum()),
            }
        )
        progress("ppo", rollout + 1)
    if not all(torch.equal(model.state_dict()[k], v) for k, v in ref_snapshot.items()):
        raise AssertionError("Reference adapter changed")
    timings["ppo"] = time.time() - begin
    model.save_pretrained(root / "artifacts/ppo", selected_adapters=["default"])
    save_file(value_head.state_dict(), str(root / "artifacts/value_head.safetensors"))
    evaluate(model, "ppo")
    write(root / "reports/ppo_rollouts.json", rollouts)
    write(root / "reports/training_history.json", all_history)
    write(root / "reports/preference_predictions.json", evaluations)
    summary = {}
    for name in ["base", "sft", "dpo", "orpo", "ppo"]:
        rows = [r for r in evaluations if r["method"] == name and r["split"] == "test"]
        wins = sum(r["mean_logp_margin"] > 0 for r in rows)
        summary[name] = {
            "n": len(rows),
            "mean_logp_accuracy": wins / len(rows),
            "ci95": wilson(wins, len(rows)),
            "mean_margin": float(np.mean([r["mean_logp_margin"] for r in rows])),
            "chosen_nll": float(
                np.mean([-r["chosen_logp"] / r["chosen_tokens"] for r in rows])
            ),
            "sum_logp_accuracy": sum(r["sum_logp_margin"] > 0 for r in rows)
            / len(rows),
        }
    rm = [r for r in reward_records if r["split"] == "test"]
    wins = sum(r["chosen"] > r["rejected"] for r in rm)
    write(
        root / "reports/experiment_summary.json",
        {
            "config": config,
            "trainable_policy_parameters": trainable,
            "total_policy_parameters": total,
            "test": summary,
            "reward_test": {
                "n": len(rm),
                "pairwise_accuracy": wins / len(rm),
                "ci95": wilson(wins, len(rm)),
            },
            "timings_seconds": timings,
            "total_seconds": time.time() - start,
            "reference_adapter_unchanged": True,
            "torch_version": torch.__version__,
        },
    )
    print(json.dumps(summary, indent=2), flush=True)


def judge(root, model_path):
    torch.set_num_threads(2)
    set_seed(42)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        dtype=torch.float32,
        attn_implementation="eager",
    ).eval()
    tok = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    rows = json.loads((root / "data/processed/judge.json").read_text())
    records = []
    for i, row in enumerate(rows):
        result = dict(row)
        for swapped in [False, True]:
            a, b = (row["b"], row["a"]) if swapped else (row["a"], row["b"])
            prompt = judge_prompt(row["question"], a, b)
            ids = tok.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            )
            if ids.shape[1] > 1536:
                raise ValueError("Judge input exceeds budget; no silent truncation")
            with torch.no_grad():
                out = model.generate(
                    ids,
                    attention_mask=torch.ones_like(ids),
                    do_sample=False,
                    max_new_tokens=8,
                    use_cache=True,
                    pad_token_id=tok.eos_token_id,
                )
            text = tok.decode(out[0, ids.shape[1] :], skip_special_tokens=True)
            key = "swapped_" if swapped else ""
            result[key + "raw_output"] = text
            result[key + "verdict"] = parse_verdict(text)
        # Avoid redistributing entire third-party conversations.
        for key in ["question", "a", "b"]:
            result.pop(key)
        records.append(result)
        print(f"judge {i + 1}/{len(rows)}", flush=True)
        write(root / "reports/judge_predictions.json", records)
    write(
        root / "reports/judge_summary.json",
        {
            s: summarize([r for r in records if r["source"] == s])
            for s in ["human", "gpt4_reference"]
        },
    )
