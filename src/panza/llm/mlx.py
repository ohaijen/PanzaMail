import json
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from .base import ChatHistoryType, LLM

_MISSING_LIBRARIES = []

try:
    import mlx.core as mx
    from mlx_lm import generate, load, stream_generate
    from mlx_lm.sample_utils import make_sampler
except ImportError:
    mx = None
    batch_generate = None
    generate = None
    load = None
    stream_generate = None
    make_logits_processors = None
    make_sampler = None
    _MISSING_LIBRARIES.append("mlx_lm")
else:
    try:
        from mlx_lm import batch_generate
    except ImportError:
        batch_generate = None
    try:
        from mlx_lm.sample_utils import make_logits_processors
    except ImportError:
        make_logits_processors = None


class MLXLLM(LLM):
    def __init__(
        self,
        name: str,
        checkpoint: str,
        sampling: Dict[str, Any],
        model: str = "",
        adapter_path: str = "",
        device: str = "mlx",
        remove_prompt_from_stream: bool = True,
        tokenizer_config: Optional[Dict[str, Any]] = None,
    ):
        self._check_installation()
        self._check_device(device)

        super().__init__(name, sampling)
        self.checkpoint = checkpoint
        self.remove_prompt_from_stream = remove_prompt_from_stream

        self.model_path, self.adapter_path = self._resolve_model_and_adapter(
            checkpoint=checkpoint,
            model=model,
            adapter_path=adapter_path,
        )
        self.max_tokens, self.sampler, self.logits_processors = self._build_generation_args(
            sampling
        )

        tokenizer_config = {"trust_remote_code": True, **(tokenizer_config or {})}
        self.model, self.tokenizer = load(
            self.model_path,
            adapter_path=self.adapter_path,
            tokenizer_config=tokenizer_config,
        )

    def chat(self, messages: ChatHistoryType | List[ChatHistoryType]) -> List[str]:
        prompts = self._messages_to_prompt_tokens(messages)
        generation_kwargs = self._generation_kwargs()
        if batch_generate is not None:
            response = batch_generate(
                self.model,
                self.tokenizer,
                prompts,
                max_tokens=self.max_tokens,
                verbose=False,
                **generation_kwargs,
            )
            return response.texts

        return [
            generate(
                self.model,
                self.tokenizer,
                prompt,
                max_tokens=self.max_tokens,
                verbose=False,
                **generation_kwargs,
            )
            for prompt in prompts
        ]

    def chat_stream(self, messages: ChatHistoryType) -> Iterator[str]:
        if self._is_batched_messages(messages):
            raise TypeError("chat_stream does not support batched messages.")

        prompt = self._messages_to_prompt_tokens(messages)[0]
        for response in stream_generate(
            self.model,
            self.tokenizer,
            prompt,
            max_tokens=self.max_tokens,
            **self._generation_kwargs(),
        ):
            yield response.text

    def _check_installation(self) -> None:
        if load is None or generate is None or stream_generate is None:
            raise ImportError(
                "mlx-lm is not installed. Please install it with "
                "`pip install mlx mlx-lm` or `pip install .[inference_mlx]`."
            )

    def _check_device(self, device: str) -> None:
        normalized = device.lower().strip()
        if normalized not in {"mlx", "mps"}:
            raise ValueError("MLXLLM only supports device='mlx' or device='mps'.")

    def _resolve_model_and_adapter(
        self,
        checkpoint: str,
        model: str,
        adapter_path: str,
    ) -> Tuple[str, Optional[str]]:
        adapter = adapter_path or self._adapter_path_from_checkpoint(checkpoint)
        if adapter:
            base_model = model or self._base_model_from_adapter_config(adapter)
            if not base_model:
                raise ValueError(
                    "Could not infer the base model for the MLX adapter. "
                    "Set interfaces.writer.llm.model=<base-model>."
                )
            return base_model, adapter

        return model or checkpoint, None

    def _adapter_path_from_checkpoint(self, checkpoint: str) -> Optional[str]:
        path = Path(checkpoint).expanduser()
        if not path.is_dir():
            return None
        if (path / "adapter_config.json").exists() and (path / "adapters.safetensors").exists():
            return str(path)
        return None

    def _base_model_from_adapter_config(self, adapter_path: str) -> str:
        config_path = Path(adapter_path).expanduser() / "adapter_config.json"
        if not config_path.exists():
            return ""
        with open(config_path, "r") as file:
            config = json.load(file)
        return str(config.get("model", ""))

    def _build_generation_args(self, sampling: Dict[str, Any]):
        max_tokens = int(sampling.get("max_new_tokens", sampling.get("max_tokens", 128)))
        do_sample = bool(sampling.get("do_sample", True))
        temperature = float(sampling.get("temperature", 0.0 if not do_sample else 1.0))
        if not do_sample:
            temperature = 0.0

        sampler = make_sampler(
            temp=temperature,
            top_p=float(sampling.get("top_p", 1.0)),
            min_p=float(sampling.get("min_p", 0.0)),
            min_tokens_to_keep=int(sampling.get("min_tokens_to_keep", 1)),
            top_k=int(sampling.get("top_k", 0)),
        )
        logits_processors = []
        if make_logits_processors is not None:
            logits_processors = make_logits_processors(
                repetition_penalty=sampling.get("repetition_penalty"),
                repetition_context_size=sampling.get("repetition_context_size", 20),
                presence_penalty=sampling.get("presence_penalty"),
                presence_context_size=sampling.get("presence_context_size", 20),
                frequency_penalty=sampling.get("frequency_penalty"),
                frequency_context_size=sampling.get("frequency_context_size", 20),
            )

        if "seed" in sampling:
            mx.random.seed(int(sampling["seed"]))

        return max_tokens, sampler, logits_processors

    def _generation_kwargs(self) -> Dict[str, Any]:
        kwargs = {"sampler": self.sampler}
        if self.logits_processors:
            kwargs["logits_processors"] = self.logits_processors
        return kwargs

    def _messages_to_prompt_tokens(
        self,
        messages: ChatHistoryType | List[ChatHistoryType],
    ) -> List[List[int]]:
        batches = messages if self._is_batched_messages(messages) else [messages]
        return [self._chat_to_prompt_tokens(batch) for batch in batches]

    def _chat_to_prompt_tokens(self, messages: ChatHistoryType) -> List[int]:
        if getattr(self.tokenizer, "has_chat_template", False) or getattr(
            self.tokenizer, "chat_template", None
        ):
            prompt = self._apply_chat_template(messages)
            if not isinstance(prompt, str):
                return self._to_token_list(prompt)
            return self._encode(prompt)

        prompt = "\n".join(str(message["content"]) for message in messages)
        return self._encode(prompt)

    def _apply_chat_template(self, messages: ChatHistoryType) -> str | List[int]:
        try:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

    def _encode(self, prompt: str) -> List[int]:
        return self._to_token_list(self.tokenizer.encode(prompt))

    def _to_token_list(self, tokens) -> List[int]:
        if hasattr(tokens, "tolist"):
            tokens = tokens.tolist()
        return [int(token) for token in tokens]

    def _is_batched_messages(self, messages: Any) -> bool:
        return bool(messages) and isinstance(messages[0], list)
