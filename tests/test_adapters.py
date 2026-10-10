"""Exercise actual Qwen/PEFT behavior with a tiny offline model."""

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import Qwen2Config, Qwen2ForCausalLM


def tiny_base():
    torch.manual_seed(42)
    torch.set_num_threads(1)
    return Qwen2ForCausalLM(
        Qwen2Config(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_position_embeddings=64,
        )
    )


def test_policy_update_keeps_base_and_reference_unchanged(tmp_path):
    model = get_peft_model(
        tiny_base(),
        LoraConfig(
            r=2,
            lora_alpha=4,
            lora_dropout=0,
            target_modules=["q_proj", "v_proj"],
            task_type="CAUSAL_LM",
        ),
    )
    model.save_pretrained(tmp_path)
    model.load_adapter(tmp_path, adapter_name="reference", is_trainable=False)
    model.set_adapter("default")
    base = {
        k: v.detach().clone() for k, v in model.state_dict().items() if "lora_" not in k
    }
    reference = {
        k: v.detach().clone()
        for k, v in model.state_dict().items()
        if ".reference." in k
    }
    initial = {
        k: v.detach().clone() for k, v in model.state_dict().items() if ".default." in k
    }
    ids = torch.tensor([[1, 2, 3, 4]])
    model.eval()
    with torch.no_grad():
        model.set_adapter("reference")
        target = model(ids).logits.detach()
    model.set_adapter("default")
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.1)
    model(ids, labels=ids).loss.backward()
    opt.step()
    assert any(not torch.equal(model.state_dict()[k], v) for k, v in initial.items())
    assert all(torch.equal(model.state_dict()[k], v) for k, v in reference.items())
    assert all(torch.equal(model.state_dict()[k], v) for k, v in base.items())
    model.set_adapter("reference")
    with torch.no_grad():
        assert torch.equal(model(ids).logits, target)


def test_adapter_round_trip_preserves_outputs(tmp_path):
    model = get_peft_model(
        tiny_base(),
        LoraConfig(
            r=2,
            lora_alpha=4,
            lora_dropout=0,
            target_modules=["q_proj", "v_proj"],
            task_type="CAUSAL_LM",
        ),
    )
    ids = torch.tensor([[1, 2, 3, 4]])
    model.eval()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=0.1
    )
    model(ids, labels=ids).loss.backward()
    optimizer.step()
    with torch.no_grad():
        expected = model(ids).logits
    model.save_pretrained(tmp_path)
    reloaded = PeftModel.from_pretrained(tiny_base(), tmp_path).eval()
    with torch.no_grad():
        actual = reloaded(ids).logits
    assert torch.allclose(actual, expected, atol=1e-6)
