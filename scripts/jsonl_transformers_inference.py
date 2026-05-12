#!/usr/bin/env python3
"""Run batched Hugging Face Transformers inference over a JSONL file."""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

try:
    import torch
except ImportError:
    torch = None

LOGGER = logging.getLogger(__name__)


def default_device() -> str:
    if torch is not None and torch.cuda.is_available():
        return "auto"
    return "cpu"


def strip_thinking_traces(text: str) -> str:
    """Remove thinking/reasoning traces from generated text.
    
    Strips content within tags like <think>...</think>, <thinking>...</thinking>,
    and other reasoning model output tags.
    """
    # Remove think tags (DeepSeek-R1, other reasoning models)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<thinking>.*?</thinking>", "", text, flags=re.DOTALL)
    # Remove other potential reasoning tags
    text = re.sub(r"<analysis>.*?</analysis>", "", text, flags=re.DOTALL)
    text = re.sub(r"<reflection>.*?</reflection>", "", text, flags=re.DOTALL)
    
    return text.strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read records from a JSONL file, generate text from a configurable query field "
            "with a local Hugging Face model, and write the augmented records back to JSONL."
        )
    )
    parser.add_argument("--input", required=True, help="Path to the input JSONL file.")
    parser.add_argument("--output", required=True, help="Path to the output JSONL file.")
    parser.add_argument(
        "--model-path",
        required=True,
        help="Path to the on-disk Hugging Face model directory or checkpoint.",
    )
    parser.add_argument(
        "--query-field",
        default="query",
        help="Field name that contains the prompt text to run through the model.",
    )
    parser.add_argument(
        "--output-field",
        default="generated_text",
        help="Field name to store the generated text under in the output JSONL.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Number of JSONL records to process at once.",
    )
    parser.add_argument(
        "--max-new-tokens",
        "--max-length",
        dest="max_new_tokens",
        type=int,
        default=2048,
        help="Maximum number of new tokens to generate per input.",
    )
    parser.add_argument(
        "--max-input-length",
        type=int,
        default=None,
        help="Optional tokenizer truncation length for the input prompt.",
    )
    parser.add_argument(
        "--device",
        default=default_device(),
        help='Inference device, for example "cpu", "cuda", "cuda:0", or "auto".',
    )
    parser.add_argument(
        "--dtype",
        choices=["auto", "fp32", "fp16", "bf16"],
        default="bf16",
        help="Torch dtype to use when loading the model.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True when loading the model and tokenizer.",
    )
    parser.add_argument(
        "--disable-chat-template",
        action="store_true",
        help="Tokenize the raw query directly instead of using the tokenizer chat template.",
    )
    parser.add_argument(
        "--do-sample",
        action="store_true",
        help="Use sampling instead of greedy decoding.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature used together with --do-sample.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Top-p value used together with --do-sample.",
    )
    parser.add_argument(
        "--disable-thinking",
        action="store_true",
        help="For reasoning models, disable output of thinking/reasoning traces.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of records to process from the input file.",
    )
    parser.add_argument(
        "--overwrite-output",
        action="store_true",
        help="Allow overwriting an existing output file.",
    )
    parser.add_argument(
        "--overwrite-output-field",
        action="store_true",
        help="Allow replacing an existing field on each record if it matches --output-field.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=1,
        help="Log progress every N processed batches.",
    )
    args = parser.parse_args()

    if args.batch_size < 1:
        parser.error("--batch-size must be a positive integer.")
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens/--max-length must be a positive integer.")
    if args.max_input_length is not None and args.max_input_length < 1:
        parser.error("--max-input-length must be a positive integer.")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be a positive integer.")
    if args.log_every < 1:
        parser.error("--log-every must be a positive integer.")
    if not args.do_sample and (
        args.temperature != parser.get_default("temperature")
        or args.top_p != parser.get_default("top_p")
    ):
        LOGGER.warning(
            "--temperature/--top-p are ignored unless --do-sample is enabled."
        )

    return args


def resolve_torch_dtype(dtype_name: str) -> torch.dtype | None:
    if dtype_name == "auto":
        return None
    if torch is None:
        raise ImportError("torch is not installed. Install it with `pip install torch`.")
    if dtype_name == "fp32":
        return torch.float32
    if dtype_name == "fp16":
        return torch.float16
    if dtype_name == "bf16":
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def iter_jsonl(path: str) -> Iterator[tuple[int, dict[str, Any]]]:
    with open(path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            stripped_line = line.strip()
            if not stripped_line:
                continue

            try:
                record = json.loads(stripped_line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} in {path}: {exc}") from exc

            if not isinstance(record, dict):
                raise ValueError(
                    f"Expected a JSON object on line {line_number} in {path}, got {type(record)}."
                )
            yield line_number, record


def batch_records(
    records: Iterator[tuple[int, dict[str, Any]]],
    batch_size: int,
    limit: int | None = None,
) -> Iterator[list[tuple[int, dict[str, Any]]]]:
    batch: list[tuple[int, dict[str, Any]]] = []
    processed = 0

    for item in records:
        if limit is not None and processed >= limit:
            break

        batch.append(item)
        processed += 1

        if len(batch) == batch_size:
            yield batch
            batch = []

    if batch:
        yield batch


class TransformersBatchGenerator:
    def __init__(
        self,
        model_path: str,
        *,
        device: str,
        dtype: str,
        max_new_tokens: int,
        max_input_length: int | None,
        trust_remote_code: bool,
        disable_chat_template: bool,
        do_sample: bool,
        temperature: float,
        top_p: float,
        disable_thinking: bool = False,
    ) -> None:
        if torch is None:
            raise ImportError("torch is not installed. Install it with `pip install torch`.")
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "transformers is not installed. Install it with `pip install transformers`."
            ) from exc

        self.device = device
        self.input_device = None if device == "auto" else torch.device(device)
        self.max_new_tokens = max_new_tokens
        self.max_input_length = max_input_length
        self.do_sample = do_sample
        self.temperature = temperature
        self.top_p = top_p
        self.disable_thinking = disable_thinking

        torch_dtype = resolve_torch_dtype(dtype)
        model_kwargs: dict[str, Any] = {
            "trust_remote_code": trust_remote_code,
            "low_cpu_mem_usage": True,
        }
        if torch_dtype is not None:
            model_kwargs["torch_dtype"] = torch_dtype
        if device == "auto":
            model_kwargs["device_map"] = "auto"

        LOGGER.info("Loading tokenizer from %s", model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code,
        )
        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token is None:
                raise ValueError(
                    "Tokenizer has no pad_token or eos_token; please configure one before inference."
                )
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        LOGGER.info("Loading model from %s", model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
        if self.input_device is not None:
            self.model.to(self.input_device)
        self.model.eval()

        self.use_chat_template = (
            not disable_chat_template and bool(getattr(self.tokenizer, "chat_template", None))
        )
        if self.use_chat_template:
            LOGGER.info("Using tokenizer chat template for prompt formatting.")
        else:
            LOGGER.info("Using raw query text without a chat template.")

    def _tokenize(self, prompts: Sequence[str]) -> dict[str, torch.Tensor]:
        if self.use_chat_template:
            conversations = [[{"role": "user", "content": prompt}] for prompt in prompts]
            kwargs: dict[str, Any] = {
                "tokenize": True,
                "add_generation_prompt": True,
                "return_tensors": "pt",
                "return_dict": True,
                "padding": True,
                "enable_thinking": False
            }
            if self.max_input_length is not None:
                kwargs["truncation"] = True
                kwargs["max_length"] = self.max_input_length
            return self.tokenizer.apply_chat_template(conversations, **kwargs)

        kwargs = {
            "return_tensors": "pt",
            "padding": True,
        }
        if self.max_input_length is not None:
            kwargs["truncation"] = True
            kwargs["max_length"] = self.max_input_length
        return self.tokenizer(list(prompts), **kwargs)

    def __call__(self, prompts: Sequence[str]) -> list[str]:
        model_inputs = self._tokenize(prompts)
        model_inputs = model_inputs.to('cuda')
        prompt_length = model_inputs["input_ids"].shape[1]

        if self.input_device is not None:
            model_inputs = {key: value.to(self.input_device) for key, value in model_inputs.items()}

        generation_kwargs: dict[str, Any] = {
            **model_inputs,
            "max_new_tokens": self.max_new_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if self.tokenizer.eos_token_id is not None:
            generation_kwargs["eos_token_id"] = self.tokenizer.eos_token_id
        if self.do_sample:
            generation_kwargs.update(
                {
                    "do_sample": True,
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                }
            )
        else:
            generation_kwargs["do_sample"] = False
        
        if self.disable_thinking:
            # For models that support disabling thinking output
            generation_kwargs["disable_thinking"] = True

        with torch.inference_mode():
            generated_ids = self.model.generate(**generation_kwargs)

        generated_text = self.tokenizer.batch_decode(
            generated_ids[:, prompt_length:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        
        # Strip thinking traces from generated text
        generated_text = [strip_thinking_traces(text) for text in generated_text]
        
        return generated_text


def process_jsonl(
    input_path: str,
    output_path: str,
    *,
    query_field: str,
    output_field: str,
    batch_size: int,
    generate_batch: Callable[[Sequence[str]], Sequence[str]],
    overwrite_output_field: bool = False,
    limit: int | None = None,
    log_every: int = 10,
) -> int:
    input_abs = os.path.abspath(input_path)
    output_abs = os.path.abspath(output_path)
    if input_abs == output_abs:
        raise ValueError("--input and --output must point to different files.")

    output_parent = Path(output_abs).parent
    output_parent.mkdir(parents=True, exist_ok=True)

    processed_records = 0

    with open(output_abs, "w", encoding="utf-8") as output_file:
        for batch_index, batch in enumerate(
            batch_records(iter_jsonl(input_abs), batch_size=batch_size, limit=limit),
            start=1,
        ):
            print("running batch")
            prompts: list[str] = []
            for line_number, record in batch:
                if query_field not in record:
                    raise KeyError(
                        f"Missing query field '{query_field}' on line {line_number} of {input_path}."
                    )

                prompt = record[query_field]
                if not isinstance(prompt, str):
                    raise TypeError(
                        f"Field '{query_field}' on line {line_number} of {input_path} must be a string."
                    )

                if output_field in record and not overwrite_output_field:
                    raise ValueError(
                        f"Field '{output_field}' already exists on line {line_number} of {input_path}. "
                        "Pass --overwrite-output-field to replace it."
                    )

                prompts.append(prompt)

            generations = list(generate_batch(prompts))
            if len(generations) != len(batch):
                raise RuntimeError(
                    "Model batch size mismatch: "
                    f"expected {len(batch)} generations, got {len(generations)}."
                )

            for (_, record), generation in zip(batch, generations):
                updated_record = dict(record)
                updated_record[output_field] = generation
                output_file.write(json.dumps(updated_record, ensure_ascii=False))
                output_file.write("\n")

            processed_records += len(batch)
            if batch_index % log_every == 0:
                LOGGER.info("Processed %s records so far.", processed_records)

    return processed_records


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    args = parse_args()

    if os.path.exists(args.output) and not args.overwrite_output:
        raise FileExistsError(
            f"Output file already exists: {args.output}. Pass --overwrite-output to replace it."
        )

    generator = TransformersBatchGenerator(
        args.model_path,
        device=args.device,
        dtype=args.dtype,
        max_new_tokens=args.max_new_tokens,
        max_input_length=args.max_input_length,
        trust_remote_code=args.trust_remote_code,
        disable_chat_template=args.disable_chat_template,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
        #disable_thinking=args.disable_thinking,
    )
    processed_records = process_jsonl(
        args.input,
        args.output,
        query_field=args.query_field,
        output_field=args.output_field,
        batch_size=args.batch_size,
        generate_batch=generator,
        overwrite_output_field=args.overwrite_output_field,
        limit=args.limit,
        log_every=args.log_every,
    )
    LOGGER.info("Finished processing %s records.", processed_records)


if __name__ == "__main__":
    main()
