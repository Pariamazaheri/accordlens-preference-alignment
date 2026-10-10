# AccordLens experimental adapters

Base: `Qwen/Qwen2.5-0.5B-Instruct`, revision `7ae557604adf67be50417f59c2c2f167def9a775`, Apache 2.0. Four rank-4 adapters target q_proj and v_proj with alpha 8 and zero dropout. Each policy adapter has 270,336 trainable parameters. PPO also exports a value head; the reward head has 896 parameters and uses frozen SFT hidden states.

SFT/DPO/ORPO policy stages each use 20 updates with two examples per update; PPO uses six sampled prompts with up to three update epochs. Data has 48 train, 16 validation, and 32 held-out short complete HH-RLHF pairs. DPO/PPO initialize at SFT; ORPO initializes at the instruction base. The reward head is selected by validation BT loss, not the test set. Test evidence and limitations are in `reports/benchmark.md`.

Intended use: reproducible experiments, objective verification, and portfolio demonstration. These adapters do not establish improved general helpfulness, retail readiness, or safe deployment. Their narrow training data and tiny budgets can introduce regressions. The small judge fails the reliability checks in this project and should not make autonomous approval decisions.

Use `PeftModel.from_pretrained(base_model, "artifacts/dpo")` with the exact pinned base model, or the CLI example in the README. No base weights or full tokenizer assets are included. `reward_head.safetensors` is a bias-free linear head on SFT final-token hidden states; inputs must use the same full native-chat completion format. `value_head.safetensors` is a linear token-value head used with the PPO policy. All model-derived artifacts follow Apache 2.0; `LICENSE-QWEN` provides its terms.
