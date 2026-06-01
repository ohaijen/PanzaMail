import json
from types import SimpleNamespace

from panza.llm import mlx as mlx_module
from panza.llm.mlx import MLXLLM


class FakeTokenizer:
    has_chat_template = True

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["tokenize"] is False
        assert kwargs["add_generation_prompt"] is True
        return f"templated:{messages[0]['content']}"

    def encode(self, prompt):
        return [len(prompt)]


def test_mlx_llm_infers_adapter_base_model_and_batches(monkeypatch, tmp_path):
    adapter_path = tmp_path / "adapter"
    adapter_path.mkdir()
    (adapter_path / "adapters.safetensors").write_bytes(b"")
    (adapter_path / "adapter_config.json").write_text(
        json.dumps({"model": "mlx-community/test-model"}),
        encoding="utf-8",
    )

    calls = {}

    def fake_load(model_path, adapter_path=None, tokenizer_config=None):
        calls["load"] = (model_path, adapter_path, tokenizer_config)
        return object(), FakeTokenizer()

    def fake_batch_generate(model, tokenizer, prompts, **kwargs):
        calls["batch_generate"] = (prompts, kwargs)
        return SimpleNamespace(texts=["first", "second"])

    monkeypatch.setattr(mlx_module, "load", fake_load)
    monkeypatch.setattr(mlx_module, "generate", lambda *args, **kwargs: "")
    monkeypatch.setattr(mlx_module, "stream_generate", lambda *args, **kwargs: iter(()))
    monkeypatch.setattr(mlx_module, "batch_generate", fake_batch_generate)
    monkeypatch.setattr(mlx_module, "make_sampler", lambda **kwargs: "sampler")
    monkeypatch.setattr(mlx_module, "make_logits_processors", lambda **kwargs: [])

    llm = MLXLLM(
        name="local",
        checkpoint=str(adapter_path),
        sampling={"do_sample": False, "max_new_tokens": 7},
    )

    assert llm.model_path == "mlx-community/test-model"
    assert llm.adapter_path == str(adapter_path)
    assert calls["load"] == (
        "mlx-community/test-model",
        str(adapter_path),
        {"trust_remote_code": True},
    )

    outputs = llm.chat(
        [
            [{"role": "user", "content": "one"}],
            [{"role": "user", "content": "two"}],
        ]
    )

    assert outputs == ["first", "second"]
    assert calls["batch_generate"] == (
        [[13], [13]],
        {"max_tokens": 7, "verbose": False, "sampler": "sampler"},
    )
