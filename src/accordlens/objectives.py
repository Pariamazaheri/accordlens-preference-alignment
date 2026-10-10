"""Numerically stable, response-only alignment objectives."""

import torch
import torch.nn.functional as F


def response_logps(logits, ids, mask):
    """Return response log-probability sums and counts with causal shifting."""
    shifted = mask[:, 1:].to(logits.dtype)
    counts = shifted.sum(-1)
    if torch.any(counts <= 0):
        raise ValueError("Every sequence must contain scored response tokens")
    lp = F.log_softmax(logits[:, :-1].float(), -1)
    token_lp = lp.gather(-1, ids[:, 1:, None]).squeeze(-1)
    return (token_lp * shifted).sum(-1), counts


def dpo_loss(chosen, rejected, ref_chosen, ref_rejected, beta=0.1):
    if beta <= 0:
        raise ValueError("beta must be positive")
    margin = beta * ((chosen - rejected) - (ref_chosen - ref_rejected))
    return -F.logsigmoid(margin).mean(), margin


def log_odds(log_probability):
    """Stable log(p/(1-p)); do not clamp low probabilities and erase gradients."""
    if torch.any(log_probability >= 0):
        raise ValueError("log probability must be strictly negative")
    return log_probability - torch.log(-torch.expm1(log_probability))


def orpo_loss(chosen_mean, rejected_mean, weight=0.1):
    if weight < 0:
        raise ValueError("weight must be nonnegative")
    margin = log_odds(chosen_mean) - log_odds(rejected_mean)
    supervised = -chosen_mean.mean()
    preference = -F.logsigmoid(margin).mean()
    return supervised + weight * preference, supervised, preference


def clipped_policy_loss(new, old, advantages, epsilon=0.2):
    """Token-level PPO surrogate; old rollout log probabilities stay fixed."""
    if not 0 < epsilon < 1:
        raise ValueError("epsilon must be between zero and one")
    ratio = (new - old).exp()
    objective = torch.minimum(
        ratio * advantages, ratio.clamp(1 - epsilon, 1 + epsilon) * advantages
    )
    return -objective.mean(), ((ratio - 1).abs() > epsilon).float().mean()


def gae(rewards, values, gamma=1.0, lam=0.95):
    """Undiscounted finite response trajectories; terminal next value is zero."""
    if rewards.shape != values.shape or rewards.ndim != 1 or len(rewards) == 0:
        raise ValueError("rewards and values must be equal nonempty vectors")
    advantage = torch.zeros_like(rewards)
    carry = torch.zeros((), device=rewards.device)
    for t in reversed(range(len(rewards))):
        next_value = values[t + 1] if t + 1 < len(values) else 0.0
        delta = rewards[t] + gamma * next_value - values[t]
        carry = delta + gamma * lam * carry
        advantage[t] = carry
    return advantage, advantage + values
