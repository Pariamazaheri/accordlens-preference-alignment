# Design decisions

## Alignment objectives

General next-token training does not directly optimize the preferences collected for an assistant. SFT fits demonstration responses; it is useful for teaching behavior but is neither universally required by every algorithm nor sufficient to guarantee safety. This project begins with an already instruction-tuned model.

DPO uses `-logsigmoid(beta * [(policy_chosen-policy_rejected) - (reference_chosen-reference_rejected)])` with response log-probability sums. The frozen reference contributes an anchor to the preference objective. Beta controls the reference-relative scaling and relates to regularization in the derivation; it should not be interpreted solely as a learning-rate multiplier.

ORPO uses chosen-response mean-token NLL plus a weighted negative log-sigmoid of the difference in response log odds. Mean-token probabilities follow the objective convention. Stable `log(p/(1-p))` computation preserves gradients even for very unlikely responses. ORPO has no additional reference model and begins at the instruction base here.

PPO optimizes generated trajectories using learned rewards, a critic, GAE, and clipped importance ratios. Clipping constrains the surrogate rather than guaranteeing a hard bound on every policy update. A sampled KL penalty adds a separate reference anchor. A very large coefficient can suppress useful change; a zero coefficient removes that anchor and can encourage reward-model exploitation. Neither setting alone determines instability. The recorded protocol uses .02 and does not claim a coefficient-sweep result.

## Evaluation and release decisions

Judge format validity, agreement, and order consistency answer different questions. Both orders are evaluated on complete candidate answers. Unknown outputs count as failures rather than ties. The short generation budget is deliberate and reported. One human annotation per sampled question avoids treating repeated annotators as independent observations but does not constitute consensus. Tie-inconsistent GPT-4 labels are grouped as ties, with original labels preserved.

Likelihood ranking is a practical reproducible diagnostic, not generated-answer quality. Normalized and summed log probabilities are supplied because length changes the comparison. Preference models must ultimately be evaluated on representative tasks with qualified human review.

## Operational boundaries

This repository supplies tested reusable experiment code, portable adapters, frozen source revisions, evidence validation, and CI. It does not include a hosted service or production quality approval. The original source files remain untouched. Published data contains identifiers and authored cases; full third-party conversations are downloaded locally, not committed.

The released configuration describes a fixed small protocol. Dataset selection, rank-4 adapters, and model identity are intentionally fixed in preparation and loading functions; increasing dataset size/rank requires changing those functions together with the configuration and revalidating results. Do not assume editing an arbitrary configuration key alone changes every implementation constant.
