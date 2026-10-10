"""Load a published adapter for a local, unvalidated response demonstration."""

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def respond(model_path, adapter, prompt):
    if not prompt.strip():
        raise ValueError("Prompt must be nonempty")
    torch.set_num_threads(2)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, local_files_only=True, dtype=torch.float32
    )
    model = PeftModel.from_pretrained(model, adapter).eval()
    ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    if ids.shape[1] > 512:
        raise ValueError("Prompt exceeds demonstration context budget")
    with torch.no_grad():
        out = model.generate(
            ids,
            attention_mask=torch.ones_like(ids),
            do_sample=False,
            temperature=1.0,
            top_p=1.0,
            top_k=50,
            max_new_tokens=64,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0, ids.shape[1] :], skip_special_tokens=True)
