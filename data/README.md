# Data provenance and selection

`profile.json` records pinned public asset checksums, invalid pair counts, and source diagnostics. `split_manifest.json` contains 48 train, 16 validation, and 32 test identifiers with globally unique exact parsed prompt hashes. These identify a length-filtered sample from Anthropic HH-RLHF. Dataset revision: `09be8c5bbc57cb3887f3a9732ad6aa7ec602a1fa`.

`bilingual_cases.json` contains six authored retail-support messages in English and Persian, used only for tokenizer cost diagnostics.

MT-Bench human/GPT-4 data revision: `f7d2896d2cc5d80f8b55c2bbc722613555233c25`. The provided GPT-4 export matches the first 1,000 official rows. The judge protocol samples 12 unique eligible questions from each source with seed 42; eligibility requires full native-chat input <=1,536 tokens. Candidate answers are never silently truncated. Only first-turn cases are used.

Run `scripts/download_assets.py` and `scripts/prepare_local.py` as described in the README. Raw and prepared full-text datasets live in ignored local folders. `reports/source_verification.json` records exact public-source reproduction. Hash ordering is stable; tokenizer/model revision and chat formatting determine eligibility. Parsed prompts retain all turns and normalize surrounding turn whitespace; this checks exact processed equality, not semantic duplicates.
