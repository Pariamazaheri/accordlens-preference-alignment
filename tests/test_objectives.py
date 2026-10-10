import math

import pytest
import torch

from accordlens.objectives import (
    clipped_policy_loss,
    dpo_loss,
    gae,
    log_odds,
    orpo_loss,
    response_logps,
)


def test_response_shift_excludes_prompt_and_padding():
    logits = torch.zeros(1, 5, 4, requires_grad=True)
    ids = torch.tensor([[0, 1, 2, 3, 0]])
    mask = torch.tensor([[0, 0, 1, 1, 0]])
    sums, counts = response_logps(logits, ids, mask)
    assert counts.item() == 2
    assert sums.item() == pytest.approx(-2 * math.log(4))
    sums.backward()
    assert logits.grad[0, 0].abs().sum() == 0
    assert logits.grad[0, 1:3].abs().sum() > 0
    assert logits.grad[0, 3:].abs().sum() == 0


def test_no_response_raises():
    with pytest.raises(ValueError):
        response_logps(
            torch.zeros(1, 3, 4), torch.zeros(1, 3, dtype=torch.long), torch.zeros(1, 3)
        )


def test_dpo_initial_reference_loss_and_gradient_direction():
    c = torch.tensor([-4.0], requires_grad=True)
    r = torch.tensor([-5.0], requires_grad=True)
    loss, margin = dpo_loss(c, r, c.detach(), r.detach())
    assert loss.item() == pytest.approx(math.log(2))
    assert margin.item() == 0
    loss.backward()
    assert c.grad.item() < 0 < r.grad.item()


def test_dpo_reference_shift_invariance():
    c = torch.tensor([-3.0])
    r = torch.tensor([-5.0])
    rc = torch.tensor([-4.0])
    rr = torch.tensor([-4.0])
    a, _ = dpo_loss(c, r, rc, rr)
    b, _ = dpo_loss(c, r, rc + 7, rr + 7)
    assert torch.equal(a, b)


@pytest.mark.parametrize("p", [0.00001, 0.1, 0.5, 0.99999])
def test_stable_log_odds(p):
    got = log_odds(torch.tensor(math.log(p), dtype=torch.float64))
    assert got.item() == pytest.approx(math.log(p / (1 - p)))


def test_low_probability_orpo_retains_gradient():
    c = torch.tensor([-30.0], requires_grad=True)
    r = torch.tensor([-35.0], requires_grad=True)
    loss, _, _ = orpo_loss(c, r)
    loss.backward()
    assert c.grad.item() != 0 and r.grad.item() != 0
    assert torch.isfinite(loss)


def test_orpo_zero_weight_is_sft():
    c = torch.tensor([-3.0])
    r = torch.tensor([-4.0])
    loss, sft, _ = orpo_loss(c, r, 0)
    assert loss == sft


def test_ppo_positive_advantage_clipping_blocks_extra_gain():
    new = torch.tensor([math.log(1.5)], requires_grad=True)
    loss, frac = clipped_policy_loss(new, torch.zeros(1), torch.ones(1))
    assert loss.item() == pytest.approx(-1.2)
    assert frac.item() == 1
    loss.backward()
    assert new.grad.item() == 0


def test_ppo_negative_advantage_clipping():
    new = torch.tensor([math.log(0.5)], requires_grad=True)
    loss, _ = clipped_policy_loss(new, torch.zeros(1), -torch.ones(1))
    assert loss.item() == pytest.approx(0.8)
    loss.backward()
    assert new.grad.item() == 0


def test_gae_terminal_returns():
    adv, returns = gae(torch.tensor([0.0, 0.0, 2.0]), torch.zeros(3), lam=1)
    assert torch.equal(returns, torch.tensor([2.0, 2.0, 2.0]))
    assert torch.equal(adv, returns)


def test_gae_value_baseline():
    adv, returns = gae(torch.tensor([0.0, 2.0]), torch.tensor([1.0, 1.0]), lam=1)
    assert torch.equal(adv, torch.ones(2))
    assert torch.equal(returns, torch.ones(2) * 2)
