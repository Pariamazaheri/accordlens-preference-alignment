# Implementation correctness review

The reviewed prototype provided real training and exploratory outputs, but several issues limited the interpretation of its results. This rebuild preserves the technical subject while replacing notebook-global execution with reusable modules and independently checkable evidence.

| Finding | Correction | Verification |
|---|---|---|
| PPO used one update per rollout; clipping had little opportunity to affect optimization | Fixed rollout probabilities, token-level clipping, multiple epochs, critic, GAE, and clipped value updates | Analytical clipping tests, actual Qwen adapter isolation test, rollout diagnostics |
| Nucleus sampling differed from the raw-policy likelihood used in the surrogate | Sample the unmodified distribution with temperature 1, top-p 1, and top-k 0 | Explicit generation parameters |
| Raw reward logits were compared with an arbitrary 0.5 baseline | Evaluate paired rankings; normalize rewards using training statistics | Row-level reward evidence and independent metric recomputation |
| Completion masks could become empty after truncation, but execution continued | Require complete native-chat responses and fail on missing scored tokens | Response mask and truncation rejection tests |
| Conversation preprocessing shortened history and text | Preserve full dialogue and select complete short sequences | Public data reproduces exact split manifests |
| ORPO code appeared in repeated training cells; low-log-probability clipping erased gradients | One explicit stage and stable log-odds using `expm1` | Gradient tests below log probability -10 |
| GPT-4 comparison labels were described as human judgments | Confirm official export identity and evaluate genuine human labels separately | Public-source verification and annotator metadata |
| Unrecognized judge outputs silently became ties | Strict A/B/TIE contract with a separate invalid class | Parser tests and raw model outputs |
| Judge visualization counted A/B human labels using model_a/model_b names | Canonical labels throughout all metrics and figures | Label tests and evidence audit |
| Training-stage scores had incompatible scales | Compare policies using matched held-out criteria; keep objective plots separate | Shared test IDs and sensitivity analysis |

The rebuild tests actual PEFT serialization and reference-adapter immutability on a tiny offline Qwen model, then independently reloads the real exported DPO adapter against held-out recorded likelihoods. Public-source preparation reproduces both preference manifests and judge cases from the local exports.

All reported gains and failures are measured. Lower chosen-response NLL is not evidence of broad helpfulness, and preference rankings from this short run do not demonstrate production alignment.
