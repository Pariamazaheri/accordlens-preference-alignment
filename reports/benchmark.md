# Preference learning and evaluator reliability

This is a measured engineering benchmark with a deliberately small CPU budget. It does not establish production model quality. The base is already instruction-tuned Qwen, so these stages perform additional adaptation rather than alignment from a raw pretrained model.

## Held-out preference evidence

| Policy | Mean-token ranking | 95% Wilson interval | Chosen token NLL ↓ | Sum-token ranking |
|---|---:|---:|---:|---:|
| BASE | 53.12% | 36.4%–69.1% | 2.853 | 53.12% |
| SFT | 53.12% | 36.4%–69.1% | 2.324 | 56.25% |
| DPO | 50.00% | 33.6%–66.4% | 2.373 | 56.25% |
| ORPO | 53.12% | 36.4%–69.1% | 2.319 | 56.25% |
| PPO | 53.12% | 36.4%–69.1% | 2.321 | 56.25% |

All methods use the same 32 held-out pairs. Mean-token ranking compares normalized response likelihoods; sum-token ranking is supplied as a length-sensitive sensitivity check. Neither is an evaluation of generated-answer helpfulness or safety. DPO optimizes a **reference-relative sum-log-probability margin**, which differs from this descriptive likelihood ranking. Do not treat raw rewards, DPO implicit margins, ORPO odds, or these likelihood scores as interchangeable scales.

SFT improves chosen-response fit, from 2.853 to 2.324 NLL, while ranking stays at 53.13%. DPO reaches 50%; ORPO and PPO reach 53.13%. No measured preference gain is established. Paired bootstrap intervals are recorded in `paired_comparisons.json`; all include zero. One seed, short complete conversations, and 32 test pairs limit conclusions. No method was selected using the test set.

## Reward model and PPO

A frozen SFT backbone supplies final-token representations to an 896-parameter linear Bradley–Terry head. The validation loss selects update 17 from 200 updates; the test ranking is 65.62%, interval 48.3%–79.6%. This interval includes chance. It is an economical reward-model architecture and may miss nuanced preference distinctions. Raw reward logits have no universal 0.5 threshold.

PPO uses six sampled training prompts, at most 24 response tokens, a learned token-value critic, GAE (gamma=1, lambda=.95), sampled reference-KL reward shaping (coefficient .02), fixed rollout log probabilities, and up to three clipped optimization epochs per rollout. Generation samples the unmodified policy distribution (temperature=1, top_p=1, top_k=0). Reward normalization uses train-only statistics and clips to ±5. The reference adapter is verified unchanged. No sampled token crossed the 0.2 clipping boundary during this recorded run, so clipping implementation is demonstrated by tests rather than an observed clipped update. Reward inputs are decoded and reformatted with the same native chat terminator as reward-training completions. Single-trajectory batches and short rollouts make this a mechanics benchmark, not a robust RLHF quality study. Sampled KL may be negative; it is an estimator, not the exact full-vocabulary KL.

## Judge evidence

The supplied export matches the first 1,000 rows of the official **GPT-4 pair judgments**, not human annotations. Genuine human labels come from the separate human split. Each source contributes 12 distinct questions, complete answers, two answer orders, greedy decoding, and an eight-token output budget. No input is silently truncated; cases exceeding 1,536 tokens are excluded before sampling. Invalid outputs count as agreement failures and are never converted to ties. The short output budget may itself cause invalid answers.

| Reference | Agreement | Majority baseline | Valid verdicts | Swap consistency among valid pairs |
|---|---:|---:|---:|---:|
| Human | 16.7% | 41.7% | 91.7% | 18.2% (11 pairs) |
| GPT-4 | 25.0% | 58.3% | 75.0% | 55.6% (9 pairs) |

The majority baseline uses the evaluation label distribution as a descriptive comparator, not a fitted deployable classifier. Human annotations can be subjective; agreement is not factual correctness. Swap inconsistency is evidence of order sensitivity, but does not isolate positional bias from all other model limitations. This judge should not gate customer-facing changes autonomously.

## Bilingual cost diagnostics

Six authored parallel retail support messages compare Llama 2 and Llama 3 token counts without loading their model weights. Characters per token measures Python Unicode code points, not bytes or grapheme clusters. Results describe these messages only; vocabulary size alone does not explain all differences, and fewer tokens do not prove better answer quality.

## Reproduction budget

Rank-4 query/value adapters train 270,336 parameters (0.0547% including the added adapter in the denominator). SFT, DPO, and ORPO each perform 20 AdamW updates with two examples per update, using the same shuffled order. Thus 40 of 48 training pairs contribute to each short run; all 48 contribute to reward training. DPO/PPO begin at SFT; ORPO begins at the instruction base and jointly learns chosen responses and preferences. Budgets differ by objective and are not compute-matched. Checkpoint policies are fixed final updates for policy stages and validation-selected for the reward head. Full stage details are in the configuration and row-level JSON evidence. Total experiment wall time was 203.2 seconds in the recorded local runtime; hardware and library versions affect reproducibility.
