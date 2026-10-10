"""Conversation validation, stable identifiers, and completion-safe encoding."""

import hashlib
import re

import torch


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def parse_hh(text):
    # Spreadsheet exports sometimes store escaped newline characters.
    text = text.replace("\\n", "\n")
    parts = re.split(r"\n\n(Human|Assistant):", text)
    if len(parts) < 3 or parts[0].strip():
        raise ValueError("Not an HH conversation")
    messages = [
        {
            "role": "user" if parts[i] == "Human" else "assistant",
            "content": parts[i + 1].strip(),
        }
        for i in range(1, len(parts), 2)
    ]
    if messages[-1]["role"] != "assistant" or not messages[-1]["content"]:
        raise ValueError("Missing final assistant response")
    if any(
        m["role"] != ("user" if i % 2 == 0 else "assistant") or not m["content"]
        for i, m in enumerate(messages)
    ):
        raise ValueError("Invalid or empty conversation turn")
    return messages[:-1], messages[-1]["content"]


def preference_pair(chosen, rejected):
    a, c = parse_hh(chosen)
    b, r = parse_hh(rejected)
    if a != b or c == r:
        raise ValueError("Mismatched prompt or identical responses")
    prompt = str(a)
    return {
        "id": digest(prompt + c + r),
        "prompt_id": digest(prompt),
        "messages": a,
        "chosen": c,
        "rejected": r,
    }


def encode_response(tokenizer, messages, response, max_length=256):
    """Use a native chat prefix; preserve the complete response or reject."""
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True
    )
    full = tokenizer.apply_chat_template(
        messages + [{"role": "assistant", "content": response}],
        tokenize=True,
        add_generation_prompt=False,
    )
    if full[: len(prompt)] != prompt:
        raise ValueError("Chat template does not preserve the generation prefix")
    if len(full) > max_length or len(full) <= len(prompt):
        raise ValueError("Sequence too long or response absent")
    ids = torch.tensor([full])
    mask = torch.zeros_like(ids)
    mask[:, len(prompt) :] = 1
    return ids, mask
