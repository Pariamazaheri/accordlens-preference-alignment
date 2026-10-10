# Third-party notices and development provenance

Code: MIT, copyright Paria Mazaheri. AI assistance supported the portfolio review, rebuild, validation, and writing. Real model training and judging were executed locally; no benchmark results were invented. Existing source notebooks and checkpoints were reviewed as input, and new experiments were run with isolated artifacts.

Qwen2.5-0.5B-Instruct is by the Qwen team and licensed Apache 2.0. Adapter and head files are derived training artifacts; the full base license is included at `artifacts/LICENSE-QWEN`. Full base weights are not redistributed. Model card: https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct

Anthropic HH-RLHF is from Anthropic, MIT: https://github.com/anthropics/hh-rlhf . Full text is excluded from Git. Conversations can contain offensive or unsafe material; they are preference-learning data and do not represent endorsed content.

MT-Bench human and GPT-4 judgments are from LMSYS, CC BY 4.0: https://huggingface.co/datasets/lmsys/mt_bench_human_judgments . Reference: Zheng et al., Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena, https://arxiv.org/abs/2306.05685 . Released results contain transformed verdicts and identifiers, not full conversations. CC BY 4.0 terms: https://creativecommons.org/licenses/by/4.0/ . Human and GPT-4 records are distinguished explicitly.

Llama 2 and Llama 3 tokenizer assets were used locally for count diagnostics only. They are not redistributed. Their model-specific license/access terms remain applicable. Six bilingual support messages in `data/bilingual_cases.json` were authored for this project and do not contain customer records.
