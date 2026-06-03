"""Hydra entrypoint for MLX LoRA fine-tuning via mlx_lm."""

import json
import math
import random
import shlex
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import hydra
from hydra.core.global_hydra import GlobalHydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import torch
#from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = REPO_ROOT / "configs"
CONFIG_NAME = "panza_finetuning"


def load_preamble(path: str) -> str:
    with open(path, "r") as file:
        return file.read().strip()


def load_user_preamble(path: str) -> str:
    with open(path, "r") as file:
        lines = [line for line in file.readlines() if not line.strip().startswith("#")]
        preamble = "".join(lines)
        if "CHANGE ME" in preamble:
            print(
                "*" * 66
                + "\n* WARNING: User prompt preamble not customized.                  *\n* Please edit the preamble at prompt_preambles/user_preamble.txt *\n"
                + "*" * 66
            )
        return preamble


if not OmegaConf.has_resolver("load_preamble"):
    OmegaConf.register_new_resolver("load_preamble", load_preamble)
if not OmegaConf.has_resolver("load_user_preamble"):
    OmegaConf.register_new_resolver("load_user_preamble", load_user_preamble)


def compose_config(overrides: Optional[Sequence[str]] = None) -> DictConfig:
    overrides = list(overrides or [])
    if not any(override.lstrip("+").startswith("panza_workspace=") for override in overrides):
        overrides.insert(0, f"panza_workspace={REPO_ROOT}")
    if not any(override.lstrip("+").startswith("finetuning=") for override in overrides):
        overrides.insert(0, "finetuning=lora")

    if GlobalHydra.instance().is_initialized():
        cfg = hydra.compose(
            config_name=CONFIG_NAME,
            overrides=overrides,
            return_hydra_config=True,
        )
    else:
        with hydra.initialize_config_dir(version_base="1.1", config_dir=str(CONFIG_DIR)):
            cfg = hydra.compose(
                config_name=CONFIG_NAME,
                overrides=overrides,
                return_hydra_config=True,
            )

    HydraConfig.instance().set_config(cfg)
    OmegaConf.set_struct(cfg, False)
    del cfg["hydra"]
    OmegaConf.set_struct(cfg, True)
    return cfg


def create_lora_mlx_run_name(cfg: DictConfig) -> str:
    run_name = f"panza_{cfg.user.username}"
    model_name = cfg.finetuning.model_name_or_path.split("/")[-1]
    run_name += f"-{model_name}"
    run_name += f"-{cfg.model_precision}"
    run_name += f"-bs{get_train_batch_size(cfg.finetuning)}"
    run_name += "-lora-mlx"
    run_name += f"-lr{cfg.finetuning.lr}"
    run_name += f"-{cfg.finetuning.max_duration}"
    run_name += f"-seed{cfg.finetuning.seed}"
    return run_name


def parse_num_epochs(max_duration: Any) -> float:
    if isinstance(max_duration, (int, float)):
        return float(max_duration)
    if isinstance(max_duration, str) and max_duration.endswith("ep"):
        return float(max_duration[:-2])
    raise ValueError(
        f"Unsupported finetuning.max_duration value: {max_duration}. "
        "For MLX LoRA training, use values like '5ep'."
    )


def get_train_batch_size(finetuning_cfg: DictConfig) -> int:
    train_batch_size = int(
        finetuning_cfg.get(
            "train_batch_size",
            finetuning_cfg.get("batch_size", 1),
        )
    )
    if train_batch_size <= 0:
        raise ValueError("finetuning.train_batch_size must be >= 1 for MLX training.")
    return train_batch_size


def calculate_grad_accumulation_steps(
    train_batch_size: int,
    microbatch_size: int,
) -> int:
    if microbatch_size <= 0:
        raise ValueError(
            "finetuning.device_train_microbatch_size must be >= 1 for MLX training."
        )
    if train_batch_size <= 0:
        raise ValueError("finetuning.train_batch_size must be >= 1 for MLX training.")
    return max(1, int(math.ceil(train_batch_size / microbatch_size)))


def get_prompt_completion(example: Dict[str, Any]) -> Tuple[str, str]:
    if "prompt" in example and "completion" in example:
        return str(example["prompt"]), str(example["completion"])
    if "prompt" in example and "response" in example:
        return str(example["prompt"]), str(example["response"])
    if "summary" in example and "email" in example:
        prompt_raw = str(example["summary"])
        prompt = prompt_raw.split("\n\nInstruction: ")[-1]
        if prompt.startswith("Instruction: "):
            prompt = prompt[len("Instruction: ") :]
        return prompt, str(example["email"])
    if "summary" in example and "snippet_text" in example:
        prompt = str(example["summary"])
        return prompt, str(example["snippet_text"])
    raise ValueError(
        "Unsupported training sample format. Expected one of: "
        "{prompt,completion}, {prompt,response}, or {summary,email}."
        f" Got {[k for k in example.keys()]}"
    )


def load_jsonl_records(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with open(path, "r") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def convert_records(records: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    converted: List[Dict[str, str]] = []
    for record in records:
        prompt, completion = get_prompt_completion(record)
        converted.append({"prompt": prompt, "completion": completion})
    return converted


# def split_train_valid(
#     samples: List[Dict[str, str]],
#     validation_split: float,
#     seed: int,
# ) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
#     if validation_split <= 0 or len(samples) < 2:
#         return samples, []

#     num_valid = max(1, int(len(samples) * validation_split))
#     num_valid = min(num_valid, len(samples) - 1)

#     indices = list(range(len(samples)))
#     rng = random.Random(seed)
#     rng.shuffle(indices)

#     valid_indices = set(indices[:num_valid])
#     train_split: List[Dict[str, str]] = []
#     valid_split: List[Dict[str, str]] = []
#     for idx, sample in enumerate(samples):
#         if idx in valid_indices:
#             valid_split.append(sample)
#         else:
#             train_split.append(sample)
#     return train_split, valid_split


def write_jsonl(path: Path, records: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=True) + "\n")


def main(cfg: Optional[DictConfig] = None, overrides: Optional[Sequence[str]] = None) -> None:
    if cfg is None:
        cfg = compose_config(overrides)

    OmegaConf.set_struct(cfg, False)
    if "lora" not in cfg.finetuning:
        raise ValueError("This trainer only supports finetuning=lora.")
    if "rosa" in cfg.finetuning:
        raise ValueError("This trainer does not support RoSA.")

    if not cfg.finetuning.run_name:
        cfg.finetuning.run_name = create_lora_mlx_run_name(cfg)
    OmegaConf.resolve(cfg)

    data_dir = Path(cfg.user.data_dir)
    train_file = data_dir / "train.jsonl"
    #valid_file = data_dir / "valid.jsonl"
    test_file = data_dir / "test.jsonl"

    if not train_file.exists():
        raise FileNotFoundError(f"Training data not found at {train_file}")

    train_samples = convert_records(load_jsonl_records(train_file))
    if not train_samples:
        raise ValueError(f"No usable records found in {train_file}")

    # if valid_file.exists():
    #     valid_samples = convert_records(load_jsonl_records(valid_file))
    # else:
    #     valid_samples = None
        # train_samples, valid_samples = split_train_valid(
        #     samples=train_samples,
        #     validation_split=float(cfg.finetuning.mlx.validation_split),
        #     seed=int(cfg.finetuning.seed),
        # )

    test_samples: List[Dict[str, str]] = []
    if test_file.exists():
        test_samples = convert_records(load_jsonl_records(test_file))

    train_batch_size = get_train_batch_size(cfg.finetuning)
    batch_size = int(
        cfg.finetuning.get(
            "device_train_microbatch_size",
            cfg.finetuning.get("batch_size", train_batch_size),
        )
    )
    if batch_size <= 0:
        raise ValueError(
            "finetuning.device_train_microbatch_size must be >= 1 for MLX training."
        )
    if len(train_samples) < batch_size:
        warnings.warn(
            "MLX requires dataset size >= batch size. "
            f"Reducing batch size from {batch_size} to {len(train_samples)}."
        )
        batch_size = len(train_samples)

    grad_accumulation_steps = calculate_grad_accumulation_steps(
        train_batch_size=train_batch_size,
        microbatch_size=batch_size,
    )
    effective_train_batch_size = batch_size * grad_accumulation_steps
    if effective_train_batch_size != train_batch_size:
        warnings.warn(
            "finetuning.train_batch_size is not divisible by the MLX microbatch size. "
            f"Using effective train batch size {effective_train_batch_size} "
            f"(microbatch={batch_size}, grad_accumulation_steps={grad_accumulation_steps})."
        )

    mlx_cfg = cfg.finetuning.mlx
    if mlx_cfg.get("iters", None):
        iters = int(mlx_cfg.iters)
    else:
        num_epochs = parse_num_epochs(cfg.finetuning.max_duration)
        steps_per_epoch = max(
            1,
            math.ceil(len(train_samples) / effective_train_batch_size),
        )
        iters = max(1, int(math.ceil(num_epochs * steps_per_epoch)))

    mlx_data_dir = Path(cfg.checkpoint_dir) / "mlx_data" / cfg.finetuning.run_name
    write_jsonl(mlx_data_dir / "train.jsonl", train_samples)
    # if valid_samples:
    #     write_jsonl(mlx_data_dir / "valid.jsonl", valid_samples)
    if test_samples:
        write_jsonl(mlx_data_dir / "test.jsonl", test_samples)

    adapter_path = Path(cfg.finetuning.hf_save_path) / cfg.finetuning.run_name
    adapter_path.mkdir(parents=True, exist_ok=True)

    # Optional parameter counting to mirror HF trainer debug output.
    # Disabled by default to avoid heavy model downloads; enable with
    # finetuning.count_params=true in your config or CLI.
    # if True or bool(cfg.finetuning.get("count_params", False)):
    #     try:
    #         lora_cfg = cfg.finetuning.lora
    #         print("Loading model to count parameters (this may download weights)...")
    #         model_tmp = AutoModelForCausalLM.from_pretrained(
    #             cfg.finetuning.model_name_or_path,
    #             torch_dtype=torch.float32,
    #             low_cpu_mem_usage=True,
    #         )
    #         peft_cfg = LoraConfig(
    #             r=int(lora_cfg.get("r", 8)),
    #             lora_alpha=int(lora_cfg.get("lora_alpha", 16)),
    #             target_modules=lora_cfg.get("target_modules", "all-linear"),
    #             lora_dropout=float(lora_cfg.get("lora_dropout", 0.05)),
    #             bias=lora_cfg.get("bias", "none"),
    #             task_type=lora_cfg.get("task_type", "CAUSAL_LM"),
    #         )
    #         model_tmp = get_peft_model(model_tmp, peft_cfg)
    #         trainable = [(n, p.numel()) for n, p in model_tmp.named_parameters() if p.requires_grad]
    #         total_trainable = sum(nm for _, nm in trainable)
    #         total_params = sum(p.numel() for _, p in model_tmp.named_parameters())
    #         print(f"(MLX) Trainable parameters: {total_trainable} ({total_trainable/1e6:.3f}M)")
    #         print(f"(MLX) Total parameters: {total_params} ({total_params/1e6:.3f}M)")
    #         print("(MLX) First 40 trainable parameters (name, numel):")
    #         for n, nm in trainable[:4000]:
    #             print(f"  {n}: {nm}")
    #     except Exception as e:
    #         print(f"Parameter counting failed: {e}")

    cmd = [
        sys.executable,
        "-m",
        "mlx_lm",
        "lora",
        "--train",
        "--model",
        str(cfg.finetuning.model_name_or_path),
        "--data",
        str(mlx_data_dir),
        "--fine-tune-type",
        "lora",
        "--optimizer",
        str(mlx_cfg.optimizer),
        "--num-layers",
        str(int(mlx_cfg.num_layers)),
        "--batch-size",
        str(batch_size),
        "--iters",
        str(iters),
        "--val-batches",
        str(int(mlx_cfg.val_batches)),
        "--learning-rate",
        str(float(cfg.finetuning.lr)),
        "--steps-per-report",
        str(int(mlx_cfg.steps_per_report)),
        "--steps-per-eval",
        str(int(mlx_cfg.steps_per_eval)),
        "--grad-accumulation-steps",
        str(grad_accumulation_steps),
        "--adapter-path",
        str(adapter_path),
        "--save-every",
        str(int(mlx_cfg.save_every)),
        "--max-seq-length",
        str(int(cfg.finetuning.max_seq_len)),
        "-c",  "/Users/jen/Projects/PanzaMail-MLX/scripts/mlx_config.yaml",
        "--seed",
        str(int(cfg.finetuning.seed)),
    ]

    if bool(mlx_cfg.mask_prompt):
        cmd.append("--mask-prompt")
    if bool(mlx_cfg.grad_checkpoint):
        cmd.append("--grad-checkpoint")
    if bool(mlx_cfg.test_after_train) and test_samples:
        cmd.extend(["--test", "--test-batches", str(int(mlx_cfg.test_batches))])

    print(
        "Prepared MLX dataset: "
        f"train={len(train_samples)}, test={len(test_samples)}, "
        f"microbatch={batch_size}, train_batch_size={train_batch_size}, "
        f"grad_accumulation_steps={grad_accumulation_steps}"
    )
    print(f"Launching: {shlex.join(cmd)}")

    if bool(mlx_cfg.dry_run):
        print("Dry run enabled. Skipping MLX training command execution.")
        return

    subprocess.run(cmd, check=True)

    if bool(cfg.finetuning.get("save_merged_model", False)):
        merged_dir = adapter_path / "merged"
        fuse_cmd = [
            sys.executable,
            "-m",
            "mlx_lm",
            "fuse",
            "--model",
            str(cfg.finetuning.model_name_or_path),
            "--adapter-path",
            str(adapter_path),
            "--save-path",
            str(merged_dir),
        ]
        print(f"Fusing adapters: {shlex.join(fuse_cmd)}")
        subprocess.run(fuse_cmd, check=True)


@hydra.main(version_base="1.1", config_path="../../../configs", config_name=CONFIG_NAME)
def hydra_main(cfg: DictConfig) -> None:
    main(cfg)


if __name__ == "__main__":
    hydra_main()
