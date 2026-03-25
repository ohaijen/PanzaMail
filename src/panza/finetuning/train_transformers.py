# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import copy
import gc
import logging
import os
import tempfile
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import random
import numpy as np
from datetime import timedelta
import torch.distributed as dist
from datasets import disable_caching

# config helpers copied from llmfoundry
from omegaconf import DictConfig, ListConfig
from omegaconf import OmegaConf as om


def pop_config(cfg, key, must_exist=False, default_value=None, convert=False):
    if key in cfg:
        val = cfg.pop(key)
        if convert and isinstance(val, (DictConfig, ListConfig)):
            return OmegaConf.to_container(val, resolve=True)
        return val
    if must_exist:
        raise KeyError(f"Required config key {key} not found")
    return default_value


def update_batch_size_info(cfg):
    # noop placeholder
    return cfg


def process_init_device(model_config, fsdp_config):
    # no special FSDP context needed for HF Trainer
    from contextlib import nullcontext

    return nullcontext()


from peft import get_peft_model, LoraConfig
from rich.traceback import install
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, PreTrainedTokenizerBase
from transformers import Trainer as HfTrainer, TrainingArguments, DataCollatorWithPadding
from transformers import TrainerCallback
from datasets import load_dataset
from torch.utils.data import DataLoader

install()
# If certain ffn types require special handling, list them here (empty for now)
ffns_with_megablocks = []

# stub tokenizer builder
from transformers import AutoTokenizer


def build_tokenizer(name, kwargs):
    tokenizer = AutoTokenizer.from_pretrained(name, **kwargs)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def build_logger(name, cfg):
    return logging.getLogger(name)


def build_callback(name, cfg, logged_cfg):
    return None


# provide simplified logging util stub


def log_config(cfg):
    logging.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")


# simple dynamic import
import importlib.util


def import_file(path):
    spec = importlib.util.spec_from_file_location(Path(path).stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


import hydra
from omegaconf import DictConfig, OmegaConf

from panza import PanzaWriter  # The import also loads custom Hydra resolvers

log = logging.getLogger(__name__)

_FSDP_SHARDING_STRATEGY_MAP = {
    "FULL_SHARD": "full_shard",
    "SHARD_GRAD_OP": "shard_grad_op",
    "NO_SHARD": "no_shard",
    "HYBRID_SHARD": "hybrid_shard",
    "HYBRID_SHARD_ZERO2": "hybrid_shard_zero2",
}

_BACKWARD_PREFETCH_MAP = {
    "BACKWARD_PRE": "backward_pre",
    "BACKWARD_POST": "backward_post",
}

_FSDP_STATE_DICT_TYPE_MAP = {
    "FULL": "FULL_STATE_DICT",
    "FULL_STATE_DICT": "FULL_STATE_DICT",
    "SHARDED": "SHARDED_STATE_DICT",
    "SHARDED_STATE_DICT": "SHARDED_STATE_DICT",
    "LOCAL": "LOCAL_STATE_DICT",
    "LOCAL_STATE_DICT": "LOCAL_STATE_DICT",
}

try:
    from torch.distributed.fsdp import FullyShardedDataParallel as _TorchFSDP
except Exception:  # pragma: no cover - optional import guard for environments without FSDP deps
    _TorchFSDP = None


def _get_dist_env_info() -> Tuple[int, int, int]:
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    return rank, local_rank, world_size


def _maybe_init_process_group(dist_timeout: Union[int, float]) -> Tuple[int, int, int]:
    rank, local_rank, world_size = _get_dist_env_info()

    if world_size > 1 and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(
            backend=backend,
            timeout=timedelta(seconds=float(dist_timeout)),
        )

    if dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()

    if torch.cuda.is_available() and local_rank < 0 and world_size > 1:
        local_rank = rank % max(1, torch.cuda.device_count())
    if torch.cuda.is_available() and local_rank >= 0:
        torch.cuda.set_device(local_rank)

    return rank, local_rank, world_size


def _infer_transformer_layer_cls_to_wrap(model: torch.nn.Module) -> Optional[str]:
    class_names = {module.__class__.__name__ for module in model.modules()}

    # known_decoder_layers = [
    #     "LlamaDecoderLayer",
    #     "MistralDecoderLayer",
    #     "MixtralDecoderLayer",
    #     "GemmaDecoderLayer",
    #     "Qwen2DecoderLayer",
    #     "GPTNeoXLayer",
    #     "OPTDecoderLayer",
    #     "BloomBlock",
    #     "GPT2Block",
    # ]
    # for cls_name in known_decoder_layers:
    #     if cls_name in class_names:
    #         return cls_name

    for cls_name in sorted(class_names):
        if cls_name.endswith("DecoderLayer") or cls_name.endswith("Block"):
            return cls_name
    return None


def _build_hf_fsdp_args(
    model: torch.nn.Module,
    fsdp_config: Optional[Dict[str, Any]],
) -> Tuple[str, Dict[str, Any]]:
    config = dict(fsdp_config or {})
    sharding_strategy = str(config.get("sharding_strategy", "FULL_SHARD")).upper()
    fsdp_options = [_FSDP_SHARDING_STRATEGY_MAP.get(sharding_strategy, "full_shard"), "auto_wrap"]
    if config.get("activation_cpu_offload", False):
        fsdp_options.append("offload")
    # mixed_precision = str(config.get("mixed_precision", "")).upper()
    # if mixed_precision in {"BF16", "PURE_BF16"}:
    #     fsdp_options.append("mixed_precision")

    hf_fsdp_config: Dict[str, Any] = {}
    if config.get("activation_checkpointing", False):
        hf_fsdp_config["activation_checkpointing"] = True

    bool_passthrough_keys = [
        "limit_all_gathers",
        "use_orig_params",
        "forward_prefetch",
        "sync_module_states",
        "cpu_ram_efficient_loading",
    ]
    for key in bool_passthrough_keys:
        if key in config:
            hf_fsdp_config[key] = bool(config[key])

    if "backward_prefetch" in config:
        backward_prefetch = str(config["backward_prefetch"]).upper()
        hf_fsdp_config["backward_prefetch"] = _BACKWARD_PREFETCH_MAP.get(
            backward_prefetch,
            str(config["backward_prefetch"]),
        )

    if "state_dict_type" in config:
        state_dict_type = str(config["state_dict_type"]).upper()
        hf_fsdp_config["state_dict_type"] = _FSDP_STATE_DICT_TYPE_MAP.get(
            state_dict_type,
            str(config["state_dict_type"]),
        )

    layer_cls_to_wrap = config.get("transformer_layer_cls_to_wrap")
    if not layer_cls_to_wrap:
        layer_cls_to_wrap = _infer_transformer_layer_cls_to_wrap(model)
    if layer_cls_to_wrap:
        hf_fsdp_config["transformer_layer_cls_to_wrap"] = layer_cls_to_wrap
    else:
        hf_fsdp_config["min_num_params"] = int(config.get("min_num_params", 1_000_000))

    return " ".join(fsdp_options), hf_fsdp_config


def _enable_model_activation_checkpointing(
    model: torch.nn.Module,
    fsdp_config: Optional[Dict[str, Any]],
) -> bool:
    if not fsdp_config or not fsdp_config.get("activation_checkpointing", False):
        return False

    if not hasattr(model, "gradient_checkpointing_enable"):
        log.warning(
            "activation_checkpointing=True requested, but model %s does not expose gradient_checkpointing_enable().",
            type(model).__name__,
        )
        return False

    use_reentrant = fsdp_config.get("activation_checkpointing_reentrant", None)
    try:
        if use_reentrant is None:
            model.gradient_checkpointing_enable()
        else:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": bool(use_reentrant)}
            )
    except TypeError:
        model.gradient_checkpointing_enable()
        log.warning(
            "Model %s does not accept gradient_checkpointing_kwargs; enabled checkpointing with default settings.",
            type(model).__name__,
        )

    log.info(
        "Enabled model-level activation checkpointing via gradient_checkpointing_enable(use_reentrant=%s).",
        use_reentrant,
    )
    return True


def _is_fsdp_wrapped_model(model: Optional[torch.nn.Module]) -> bool:
    if model is None:
        return False
    if _TorchFSDP is not None and isinstance(model, _TorchFSDP):
        return True
    # Fallback check keeps this robust when torch FSDP is unavailable in tooling env.
    return any(cls.__name__ == "FullyShardedDataParallel" for cls in type(model).mro())


class _FSDPStateLoggingCallback(TrainerCallback):
    def __init__(self, expect_fsdp: bool, world_size: int):
        self.expect_fsdp = expect_fsdp
        self.world_size = world_size

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        rank = dist.get_rank() if dist.is_initialized() else 0
        is_wrapped = _is_fsdp_wrapped_model(model)
        model_type = type(model).__name__ if model is not None else "None"
        if self.expect_fsdp and not is_wrapped:
            log.warning(
                "Expected FSDP wrapping (world_size=%s, fsdp=%s), but model type at train begin is '%s'.",
                self.world_size,
                args.fsdp,
                model_type,
            )
        else:
            log.info(
                "Train begin rank=%s world_size=%s distributed_initialized=%s model_type=%s fsdp_wrapped=%s",
                rank,
                self.world_size,
                dist.is_initialized(),
                model_type,
                is_wrapped,
            )


def validate_config(cfg: DictConfig):
    """Validates compatible model and dataloader selection."""
    loaders = [cfg.train_loader]
    if "eval_loader" in cfg:
        eval_loader = cfg.eval_loader
        if isinstance(eval_loader, ListConfig):
            for loader in eval_loader:
                if loader.label is None:
                    raise ValueError(
                        "When specifying multiple evaluation datasets, each one must include the \
                            `label` attribute."
                    )
                loaders.append(loader)
        else:
            loaders.append(eval_loader)
    for loader in loaders:
        if loader.name == "text":
            if cfg.model.name == "hf_t5":
                raise ValueError(
                    f'Model type "{cfg.model.name}" is not supported when using the "text " '
                    + f"dataloader. Only finetuning is supported."
                )

    if "icl_tasks" in cfg:
        if cfg.model.name == "hf_t5":
            raise ValueError(
                'ICL evaluation does not currently support Encoder-Decoder models, such as "hf_t5".'
            )

    if (
        cfg.model.get("fc_type", "torch") != "te"
        and "te" not in cfg.model.get("ffn_config", {}).get("ffn_type", "mptmlp")
        and "fp8" in cfg.precision
    ):
        warnings.warn(
            "fp8 only supported for te.Linear layers. Either set `cfg.model.fc_typ='te'` or "
            + "`cfg.model.ffn_config.ffn_type='te_ln_mlp'` to enable layers using fp8 precision."
        )

    if cfg.model.get("fc_type", "torch") == "te" or "te" in cfg.model.get("ffn_config", {}).get(
        "ffn_type", "mptmlp"
    ):
        fsdp_config = cfg.get("fsdp_config", None)
        act_ckpt = fsdp_config.get("activation_checkpointing", False)
        act_ckpt_reentrant = fsdp_config.get("activation_checkpointing_reentrant", False)
        if fsdp_config is not None and act_ckpt == True and act_ckpt_reentrant == True:
            warnings.warn(
                "`te.Linear` layers do not support activation_checkpointing with "
                + "`activation_checkpointing_reentrant = True`. "
                + "Setting cfg.fsdp_config.activation_checkpointing_reentrant=False."
            )
            cfg.fsdp_config.activation_checkpointing_reentrant = False

    if cfg.model.get("ffn_config", {}).get("ffn_type", "mptmlp") == "te_ln_mlp":
        warnings.warn(
            "`te.LayerNormMLP` requires has issues with torch._dynamo. "
            + "Setting `torch._dynamo.config.suppress_errors = True` and falling back to eager."
        )
        torch._dynamo.config.suppress_errors = True  # type: ignore (third-party)

    if cfg.model.get("load_in_8bit", False):
        raise ValueError("`load_in_8bit` is only supported for evaluation rather than training.")

    if cfg.model.get("ffn_config", {}).get("ffn_type", "mptmlp") in ffns_with_megablocks:
        moe_world_size = cfg.model.get("ffn_config", {}).get("moe_world_size", 1)
        use_orig_params = cfg.get("fsdp_config", {}).get("use_orig_params", True)
        if moe_world_size > 1 and not use_orig_params:
            raise ValueError(
                f"MoEs with expert parallelism (moe_world_size {moe_world_size} > 1) require `use_orig_params=True`."
            )


def _load_and_prepare_dataset(
    loader_cfg: DictConfig,
    tokenizer,
    is_train: bool = True,
    global_max_len: Optional[int] = None,
):
    """Load HF dataset according to loader_cfg and tokenize examples.

    ``global_max_len`` is used as a fallback when the dataset configuration
    does not explicitly provide ``max_seq_len``.  This ensures the top-level
    ``finetuning.max_seq_len`` parameter is respected.
    """
    ds_conf = loader_cfg.dataset
    hf_name = ds_conf.hf_name
    split = ds_conf.get("split", "train")
    hf_kwargs = ds_conf.get("hf_kwargs", {})
    dataset = load_dataset(hf_name, split=split, **hf_kwargs)
    # apply preprocessing function if provided
    if "preprocessing_fn" in ds_conf and ds_conf.preprocessing_fn:
        module, fn = ds_conf.preprocessing_fn.split(":")
        func = getattr(__import__(module, fromlist=[fn]), fn)
        dataset = dataset.map(func, batched=False, remove_columns=dataset.column_names)
    # tokenization
    # prefer dataset-specific setting, otherwise fall back to global
    max_len = ds_conf.get("max_seq_len", None)
    if max_len is None:
        max_len = global_max_len

    def tokenize_fn(example):
        prompt = example.get("prompt", "")
        response = example.get("response", "")

        # Build token-level fields explicitly so mask length always matches input_ids length.
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        response_ids = tokenizer(response, add_special_tokens=False)["input_ids"]

        input_ids = (prompt_ids + response_ids)[:max_len]
        prompt_token_count = min(len(prompt_ids), len(input_ids))
        response_ids_truncated = input_ids[prompt_token_count:]

        # For causal LM, attention_mask marks real (non-pad) tokens.
        attention_mask = [1] * len(input_ids)
        # Supervise only the response tokens; ignore prompt tokens in loss.
        labels = ([-100] * prompt_token_count) + response_ids_truncated

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    # Preserve one dataset row as one training example.
    dataset = dataset.map(tokenize_fn, batched=False, remove_columns=dataset.column_names)
    if is_train and ds_conf.get("shuffle", True):
        dataset = dataset.shuffle()
    return dataset


def _make_dataloader(
    dataset, tokenizer, batch_size: int, is_train: bool = True, loader_cfg: DictConfig = None
):
    def collator(features):
        token_features = []
        for feature in features:
            token_features.append(
                {
                    "input_ids": feature["input_ids"],
                    "attention_mask": feature.get("attention_mask"),
                }
            )

        batch = tokenizer.pad(
            token_features,
            padding=True,
            return_tensors="pt",
        )

        if "labels" in features[0]:
            max_len = batch["input_ids"].shape[1]
            padded_labels = []
            for feature in features:
                labels = feature["labels"]
                if torch.is_tensor(labels):
                    labels = labels.tolist()
                else:
                    labels = list(labels)

                if len(labels) > max_len:
                    labels = labels[:max_len]
                else:
                    labels = labels + ([-100] * (max_len - len(labels)))
                padded_labels.append(labels)

            batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)

        return batch

    extra = {}
    if loader_cfg is not None:
        # pass through supported dataloader args
        for arg in [
            "num_workers",
            "pin_memory",
            "prefetch_factor",
            "persistent_workers",
            "drop_last",
        ]:
            if arg in loader_cfg:
                extra[arg] = loader_cfg[arg]
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collator, **extra)

    # if "icl_tasks" in cfg:
    #     if cfg.model.name == "hf_t5":
    #         raise ValueError(
    #             'ICL evaluation does not currently support Encoder-Decoder models, such as "hf_t5".'
    #         )

    # if (
    #     cfg.model.get("fc_type", "torch") != "te"
    #     and "te" not in cfg.model.get("ffn_config", {}).get("ffn_type", "mptmlp")
    #     and "fp8" in cfg.precision
    # ):
    #     warnings.warn(
    #         "fp8 only supported for te.Linear layers. Either set `cfg.model.fc_typ='te'` or "
    #         + "`cfg.model.ffn_config.ffn_type='te_ln_mlp'` to enable layers using fp8 precision."
    #     )

    # if cfg.model.get("fc_type", "torch") == "te" or "te" in cfg.model.get("ffn_config", {}).get(
    #     "ffn_type", "mptmlp"
    # ):
    #     fsdp_config = cfg.get("fsdp_config", None)
    #     act_ckpt = fsdp_config.get("activation_checkpointing", False)
    #     act_ckpt_reentrant = fsdp_config.get("activation_checkpointing_reentrant", False)
    #     if fsdp_config is not None and act_ckpt == True and act_ckpt_reentrant == True:
    #         warnings.warn(
    #             "`te.Linear` layers eo not support activation_checkpointing with "
    #             + "`activation_checkpointing_reentrant = True`. "
    #             + "Setting cfg.fsdp_config.activation_checkpointing_reentrant=False."
    #         )
    #         cfg.fsdp_config.activation_checkpointing_reentrant = False

    # if cfg.model.get("ffn_config", {}).get("ffn_type", "mptmlp") == "te_ln_mlp":
    #     warnings.warn(
    #         "`te.LayerNormMLP` requires has issues with torch._dynamo. "
    #         + "Setting `torch._dynamo.config.suppress_errors = True` and falling back to eager."
    #     )
    #     torch._dynamo.config.suppress_errors = True  # type: ignore (third-party)

    # if cfg.model.get("load_in_8bit", False):
    #     raise ValueError("`load_in_8bit` is only supported for evaluation rather than training.")

    # if cfg.model.get("ffn_config", {}).get("ffn_type", "mptmlp") in ffns_with_megablocks:
    #     moe_world_size = cfg.model.get("ffn_config", {}).get("moe_world_size", 1)
    #     use_orig_params = cfg.get("fsdp_config", {}).get("use_orig_params", True)
    #     if moe_world_size > 1 and not use_orig_params:
    #         raise ValueError(
    #             f"MoEs with expert parallelism (moe_world_size {moe_world_size} > 1) require `use_orig_params=True`."
    #         )


def create_run_name(cfg: DictConfig) -> str:
    # export RUN_NAME=panza_${PANZA_USERNAME}_${MODEL_TYPE}_${MODEL_PRECISION}-bs${BS}-fft-lr${LR}-epochs${NUM_EPOCHS}-wu${WARMUP}-seed${SEED}${PREAMBLE_STR}${RAFT_STR}

    run_name = f"panza_{cfg.user.username}"

    model_name = cfg.finetuning.model_name_or_path.split("/")[-1]
    run_name += f"-{model_name}"

    run_name += f"-{cfg.model_precision}"
    run_name += f"-bs{cfg.finetuning.train_batch_size}"

    if hasattr(cfg.finetuning, "lora"):
        run_name += "-lora"
        run_name += f"-r{cfg.finetuning.lora.rank}"
    else:
        run_name += "-fft"

    run_name += f"-lr{cfg.finetuning.lr}"
    run_name += f"-{cfg.finetuning.max_duration}"
    run_name += f"-seed{cfg.finetuning.seed}"

    return run_name


def create_checkpoint_dirs(cfg: DictConfig) -> None:
    # Create model directory
    os.makedirs(os.path.join(cfg.checkpoint_dir, "models"), exist_ok=True)


def get_hf_save_precision(cfg: DictConfig) -> str:
    if cfg.model_precision == "bf16":
        return "bfloat16"
    elif cfg.model_precision == "fp32":
        return "float32"
    else:
        raise ValueError(f"Unsupported model_precision: {cfg.model_precision}")


def override_config(cfg: DictConfig) -> None:
    # Disable struct mode to allow modifications
    OmegaConf.set_struct(cfg, False)

    if not cfg.finetuning.run_name:
        cfg.finetuning.run_name = create_run_name(cfg)

    # when not doing LoRA, set the HF checkpointer precision normally
    if not hasattr(cfg.finetuning, "lora"):
        cfg.finetuning.callbacks.hf_checkpointer.precision = get_hf_save_precision(cfg)

    # Re-enable struct mode to lock down the configuration
    OmegaConf.set_struct(cfg, True)


def save_config_to_yaml(cfg: DictConfig) -> str:
    cfg = OmegaConf.to_container(cfg, resolve=True)
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".yaml") as temp_file:
        OmegaConf.save(config=cfg, f=temp_file.name)
        return temp_file.name


def build_hf_peft_model(
    model_config: str,
    lora_config: Optional[Dict[str, Any]],
    tokenizer: PreTrainedTokenizerBase,
    is_fsdp: bool = False,
) -> torch.nn.Module:
    """Load a HuggingFace causal LM, optionally attach PEFT (LoRA) adapters.

    Returns the HF model instance ready for training.
    """
    print("Building model from HuggingFace checkpoint...")

    weight_bias_dtype = model_config.get("weight_bias_dtype", None)
    if weight_bias_dtype == "4bit":
        compute_dtype = torch.bfloat16
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    elif weight_bias_dtype == "bf16":
        compute_dtype = torch.bfloat16
        quant_config = None
    else:
        assert weight_bias_dtype == "fp32"
        compute_dtype = torch.float32
        quant_config = None

    model = AutoModelForCausalLM.from_pretrained(
        model_config.pretrained_model_name_or_path,
        device_map="cpu" if quant_config is None else "auto",
        torch_dtype=compute_dtype,
        quantization_config=quant_config,
        trust_remote_code=True,
        # use_auth_token=True,
        # use_cache=False,
        attn_implementation="eager",
    )

    print("Model built!")
    if lora_config is not None:
        print("Building LoRA config...")
        config = LoraConfig(
            r=lora_config.get("rank"),
            lora_alpha=lora_config.get("alpha", 16),
            target_modules=lora_config.get("target_modules", "all"),
            lora_dropout=lora_config.get("dropout", 0.05),
            bias=lora_config.get("bias", "none"),
            task_type="CAUSAL_LM",
        )
        print("Adding LoRA modules...")
        model = get_peft_model(model, config)
        print("LoRA modules added!")

    # model is ready (possibly with LoRA adapters). return it directly.
    return model


@hydra.main(version_base="1.1", config_path="../../../configs", config_name="panza_finetuning")
def main(cfg: DictConfig) -> HfTrainer:
    override_config(cfg)

    # Resolve all interpolation variables as early as possible
    om.resolve(cfg)

    # The preprocessing config is saved to a temporary directory
    # and accessed through an environment variable. Note that this
    # happens separately for each process (however, a collision should)
    # not be a problem, since the configs are the same.
    OmegaConf.set_struct(cfg, False)
    cfg.preprocessing.model = cfg.finetuning.model_name_or_path
    preprocessing_yaml = save_config_to_yaml(cfg.preprocessing)

    environment = os.environ
    environment["WANDB_PROJECT"] = f"panza-{cfg.user.username}"
    environment["WANDB_DISABLED"] = str(int(cfg.finetuning.wandb_disabled))
    environment["PANZA_PREPROCESSING_CONFIG"] = preprocessing_yaml

    cfg = cfg.finetuning

    # Make the config editable for popping.
    OmegaConf.set_struct(cfg, False)

    # Run user provided code if specified
    code_paths = pop_config(cfg, "code_paths", must_exist=False, default_value=[], convert=True)
    # Import any user provided code
    for code_path in code_paths:
        import_file(code_path)

    # Filter deprecation warning from torch internal usage
    warnings.filterwarnings(
        action="ignore",
        category=UserWarning,
        message="torch.distributed.*_base is a private function and will be deprecated.*",
    )

    # Check for incompatibilities between the model and data loaders
    validate_config(cfg)

    # Resolve all interpolation variables as early as possible
    om.resolve(cfg)

    # Create copy of config for logging
    logged_cfg: DictConfig = copy.deepcopy(cfg)

    cuda_alloc_conf = []
    # Get max split size mb
    max_split_size_mb: Optional[int] = cfg.pop("max_split_size_mb", None)
    if max_split_size_mb is not None:
        cuda_alloc_conf.append(f"max_split_size_mb:{max_split_size_mb}")

    # Expandable segments
    if cfg.pop("expandable_segments", False):
        cuda_alloc_conf.append("expandable_segments:True")

    if len(cuda_alloc_conf) > 0:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = ",".join(cuda_alloc_conf)

    # Set CUDA lazy loading
    # This can save a bit of memory if not all modules are needed
    cuda_load_lazy: bool = cfg.pop("cuda_load_lazy", False)
    if cuda_load_lazy:
        os.environ["CUDA_MODULE_LOADING"] = "LAZY"

    # Set seed first
    seed: int = pop_config(cfg, "seed", must_exist=True)
    # Set seeds for reproducibility
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Initialize distributed process groups and CUDA device placement.
    dist_timeout: Union[int, float] = pop_config(
        cfg, "dist_timeout", must_exist=False, default_value=600.0
    )
    rank, local_rank, world_size = _maybe_init_process_group(dist_timeout)
    if torch.cuda.is_available() and torch.cuda.device_count() > 1 and world_size == 1:
        warnings.warn(
            f"{torch.cuda.device_count()} CUDA devices are visible, but WORLD_SIZE is 1. "
            "Launch with torchrun --nproc_per_node=<num_gpus> to enable multi-GPU/FSDP."
        )

    save_merged_model: bool = pop_config(cfg, "save_merged_model", False)

    # Get global and device batch size information from distributed/single node setting
    cfg = update_batch_size_info(cfg)
    logged_cfg.update(cfg, merge=True)

    # Mandatory model training configs
    model_config: DictConfig = pop_config(cfg, "model", must_exist=True)
    tokenizer_config: Dict[str, Any] = pop_config(cfg, "tokenizer", must_exist=True, convert=True)
    optimizer_config: Dict[str, Any] = pop_config(cfg, "optimizer", must_exist=True, convert=True)
    scheduler_config: Dict[str, Any] = pop_config(cfg, "scheduler", must_exist=True, convert=True)
    train_loader_config: DictConfig = pop_config(cfg, "train_loader", must_exist=True)

    # Optional fsdp data, fine-tuning, and eval configs
    fsdp_config: Optional[Dict[str, Any]] = pop_config(
        cfg, "fsdp_config", must_exist=False, default_value=None, convert=True
    )

    ds_config: Optional[Dict[str, Any]] = pop_config(
        cfg, "ds_config", must_exist=False, default_value=None, convert=True
    )

    lora_config: Optional[Dict[str, Any]] = pop_config(
        cfg, "lora", must_exist=False, default_value=None, convert=True
    )

    hf_save_path: Union[int, str] = pop_config(cfg, "hf_save_path", must_exist=True)

    eval_loader_config: Optional[Union[DictConfig, ListConfig]] = pop_config(
        cfg, "eval_loader", must_exist=False, default_value=None
    )
    # icl_tasks_config: Optional[Union[ListConfig, str]] = pop_config(
    #     cfg, "icl_tasks", must_exist=False, default_value=None
    # )
    # eval_gauntlet_config: Optional[Union[DictConfig, str]] = pop_config(
    #     cfg, "eval_gauntlet", must_exist=False, default_value=None
    # )
    # icl_subset_num_batches: Optional[int] = pop_config(
    #     cfg, "icl_subset_num_batches", must_exist=False, default_value=None
    # )
    # icl_seq_len: Optional[int] = pop_config(
    #     cfg, "icl_seq_len", must_exist=False, default_value=None
    # )
    # # # Optional logging, evaluation and callback configs
    # # logger_configs: Optional[DictConfig] = pop_config(
    # #     cfg, "loggers", must_exist=False, default_value=None, convert=True
    # # )
    # # # callback configuration ignored
    # # callback_configs: Optional[DictConfig] = pop_config(
    # #     cfg, "callbacks", must_exist=False, default_value=None, convert=True
    # )
    # # algorithms config no longer used with HF Trainer
    # _unused_algorithm_configs: Optional[DictConfig] = pop_config(
    #     cfg, "algorithms", must_exist=False, default_value=None
    # )

    # Mandatory hyperparameters for training
    train_batch_size: int = pop_config(cfg, "train_batch_size", must_exist=True)
    device_eval_batch_size: int = pop_config(cfg, "device_eval_batch_size", must_exist=True)
    max_duration: Union[int, str] = pop_config(cfg, "max_duration", must_exist=True)
    eval_interval: Union[int, str] = pop_config(
        cfg, "eval_interval", default_value=1, must_exist=False
    )
    precision: str = pop_config(cfg, "precision", must_exist=True)
    max_seq_len: int = pop_config(cfg, "max_seq_len", must_exist=True)

    # Optional parameters will be set to default values if not specified.
    default_run_name: str = os.environ.get("RUN_NAME", "llm")
    run_name: str = pop_config(cfg, "run_name", must_exist=False, default_value=default_run_name)
    save_folder: Optional[str] = pop_config(
        cfg, "save_folder", must_exist=False, default_value=None
    )
    is_state_dict_sharded: bool = (
        (fsdp_config.get("state_dict_type", "full") == "sharded") if fsdp_config else False
    )
    # save_latest_filename: str = pop_config(
    #     cfg,
    #     "save_latest_filename",
    #     must_exist=False,
    #     default_value=(
    #         "latest-sharded-rank{rank}" if is_state_dict_sharded else "latest-rank{rank}.pt"
    #     ),
    # )
    # save_overwrite: bool = pop_config(cfg, "save_overwrite", must_exist=False, default_value=False)
    # save_weights_only: bool = pop_config(
    #     cfg, "save_weights_only", must_exist=False, default_value=False
    # )
    # save_filename: str = pop_config(
    #     cfg, "save_filename", must_exist=False, default_value="ep{epoch}-ba{batch}-rank{rank}.pt"
    # )
    save_interval: Union[str, int] = pop_config(
        cfg, "save_interval", must_exist=False, default_value="10000ba"
    )
    save_num_checkpoints_to_keep: int = pop_config(
        cfg, "save_num_checkpoints_to_keep", must_exist=False, default_value=-1
    )
    progress_bar = pop_config(cfg, "progress_bar", must_exist=False, default_value=False)
    # log_to_console: bool = pop_config(cfg, "log_to_console", must_exist=False, default_value=True)
    python_log_level: Optional[str] = pop_config(
        cfg, "python_log_level", must_exist=False, default_value="debug"
    )
    console_log_interval: Union[int, str] = pop_config(
        cfg, "console_log_interval", must_exist=False, default_value="1ba"
    )
    device_train_microbatch_size: Union[str, int] = pop_config(
        cfg, "device_train_microbatch_size", must_exist=False, default_value="auto"
    )
    # eval_subset_num_batches: int = pop_config(
    #     cfg, "eval_subset_num_batches", must_exist=False, default_value=-1
    # )
    # eval_first: bool = pop_config(cfg, "eval_first", must_exist=False, default_value=False)
    # load_path: str = pop_config(cfg, "load_path", must_exist=False, default_value=None)
    # load_weights_only: bool = pop_config(
    #     cfg, "load_weights_only", must_exist=False, default_value=False
    # )
    # load_strict_model_weights: bool = pop_config(
    #     cfg, "load_strict_model_weights", must_exist=False, default_value=True
    # )
    # load_ignore_keys: Optional[List[str]] = pop_config(
    #     cfg, "load_ignore_keys", must_exist=False, default_value=None
    # )
    # save_ignore_keys: Optional[List[str]] = pop_config(
    #     cfg, "save_ignore_keys", must_exist=False, default_value=None
    # )
    # compile_config: Optional[Dict[str, Any]] = pop_config(
    #     cfg, "compile_config", must_exist=False, default_value=None
    # )
    # metadata: Optional[Dict[str, str]] = pop_config(
    #     cfg, "metadata", must_exist=False, default_value=None, convert=True
    # )
    should_log_config: bool = pop_config(cfg, "log_config", must_exist=False, default_value=True)

    num_cpu_threads: Optional[int] = cfg.pop("num_cpu_threads", 0)
    if num_cpu_threads > 0:
        print(f"Setting number of CPU threads to {num_cpu_threads}")
        torch.set_num_threads(num_cpu_threads)

    # # Enable autoresume from model checkpoints if possible
    # autoresume_default: bool = False
    # if (
    #     logged_cfg.get("run_name", None) is not None
    #     and save_folder is not None
    #     and not save_overwrite
    #     and not save_weights_only
    # ):
    #     autoresume_default = True

    # if cfg.get("autoresume") is None and autoresume_default:
    #     log.info(
    #         "As run_name, save_folder, and save_latest_filename are set, \
    #             changing autoresume default to True..."
    #     )

    # autoresume: bool = pop_config(
    #     cfg, "autoresume", must_exist=False, default_value=autoresume_default
    # )

    # Pop known unused parameters that are used as interpolation variables or
    # created by update_batch_size_info.
    # pop_config(cfg, "data_local", must_exist=False)
    # pop_config(cfg, "data_remote", must_exist=False)
    # pop_config(cfg, "global_seed", must_exist=False)
    # pop_config(cfg, "global_train_batch_size", must_exist=False)
    # pop_config(cfg, "n_gpus", must_exist=False)
    # pop_config(cfg, "device_train_grad_accum", must_exist=False)

    assert fsdp_config is None or ds_config is None, "fsdp and deepspeed are not supported together"

    # Warn users for unused parameters
    for key in cfg:
        warnings.warn(
            f"Unused parameter {key} found in cfg. Please check your yaml to ensure this parameter is necessary."
        )

    if fsdp_config is None and torch.cuda.is_available() and world_size > 1:
        fsdp_config = {}
        log.info("WORLD_SIZE=%s with CUDA detected. Enabling FSDP defaults.", world_size)

    if fsdp_config is not None and not torch.cuda.is_available():
        warnings.warn("FSDP requires CUDA. Reverting to non-FSDP training.")
        fsdp_config = None
    elif fsdp_config is not None and world_size == 1:
        warnings.warn(
            "FSDP requires distributed launch (WORLD_SIZE > 1). "
            "Launch with torchrun --nproc_per_node=<num_gpus> to enable FSDP."
        )
        fsdp_config = None

    # set logging level
    if python_log_level is not None:
        logging.basicConfig(
            # Example of format string
            # 2022-06-29 11:22:26,152: rank0[822018][MainThread]: INFO: Message here
            format=f"%(asctime)s: rank{rank}[%(process)d][%(threadName)s]: %(levelname)s: %(name)s: %(message)s"
        )
        logging.getLogger(__name__).setLevel(python_log_level.upper())  # Train script

    # Initialize context
    init_context = process_init_device(model_config, fsdp_config)
    logged_cfg.update({"fsdp_config": fsdp_config}, merge=True)

    # Build tokenizer
    log.info("Building tokenizer...")
    tokenizer_name = tokenizer_config["name"]
    tokenizer_kwargs = tokenizer_config.get("kwargs", {})
    tokenizer_kwargs["num_proc"] = 1
    tokenizer = build_tokenizer(tokenizer_name, tokenizer_kwargs)

    # Scheduler configuration is ignored when using HF Trainer; parameters are passed via TrainingArguments
    # (scheduler_config remains available if needed for custom logic)

    # loggers and callbacks are not supported in this simplified version
    # loggers = []
    # mosaicml_logger = None
    # metadata logging skipped

    # # Callbacks not implemented; ignore configs
    # callbacks: List[Any] = []

    print("LORA CONFIG", lora_config)
    # Build Model
    print("Initializing model...")
    with init_context:
        model = build_hf_peft_model(
            model_config, lora_config, tokenizer, is_fsdp=fsdp_config is not None
        )
    activation_checkpointing_via_model = _enable_model_activation_checkpointing(model, fsdp_config)

    # Dataloaders
    log.info("Building train dataset and loader...")
    try:
        disable_caching()
        train_dataset = _load_and_prepare_dataset(
            train_loader_config,
            tokenizer,
            is_train=True,
            global_max_len=max_seq_len,
        )
        # optional: create PyTorch DataLoader if desired for debugging

    except Exception as e:
        # if mosaicml_logger is not None:
        #     mosaicml_logger.log_exception(e)
        raise e

    # do not remove - debugging code to check the data.
    # train_loader = _make_dataloader(train_dataset, tokenizer, 2, is_train=True, loader_cfg=train_loader_config)
    # for batch in train_loader:
    #     print(batch)
    # sys.exit()

    # if mosaicml_logger is not None:
    #     mosaicml_logger.log_metrics({"data_validated": time.time()})

    # Evaluation dataset
    eval_dataset = None
    if eval_loader_config is not None:
        log.info("Building eval dataset...")
        eval_dataset = _load_and_prepare_dataset(
            eval_loader_config,
            tokenizer,
            is_train=False,
            global_max_len=max_seq_len,
        )
        # optionally build a DataLoader if desired for debugging
        # eval_loader = _make_dataloader(eval_dataset, tokenizer, device_eval_batch_size, is_train=False, loader_cfg=eval_loader_config)
        # gauntlet callbacks are not supported

    # if mosaicml_logger is not None:
    #     log_train_analytics(
    #         mosaicml_logger,
    #         model_config,
    #         train_loader_config,
    #         eval_loader_config,
    #         callback_configs,
    #         tokenizer_name,
    #         load_path,
    #         icl_tasks_config,
    #         eval_gauntlet_config,
    #     )
    # Log number of parameters
    if hasattr(model, "n_total_params"):
        n_params = model.n_total_params
        n_trainable_params = n_params  # We currently assume all parameters are trainable.
    else:
        n_params = sum(p.numel() for p in model.parameters())
        n_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if hasattr(model, "n_active_params"):
        n_active_params = model.n_active_params
    else:
        n_active_params = n_params
    logged_cfg.update(
        {
            "n_params": n_params,
            "n_active_params": n_active_params,
            "n_trainable_params": n_trainable_params,
        }
    )

    # Optimizer - Extract parameters from config
    optimizer_name: str = optimizer_config.pop("name")
    opt_kwargs = dict(optimizer_config)
    base_lr = opt_kwargs.pop("lr", 5e-5)

    log.info(f"Optimizer: {optimizer_name}")
    log.info(f"Optimizer kwargs: {opt_kwargs}")

    # Map optimizer name to HF format (e.g., "decoupled_adamw" -> "adamw_torch")
    optim_type = "adamw_torch"
    if "adamw" in optimizer_name.lower() or "adam" in optimizer_name.lower():
        optim_type = "adamw_torch"

    # Create custom optimizer with LoRA parameter groups if needed
    custom_optimizer = None
    if lora_config and "lora_lr" in lora_config:
        log.info(f"Using different learning rate for LoRA params: {lora_config['lora_lr']}")
        lora_params = []
        other_params = []
        for name, param in model.named_parameters():
            if "lora_" in name:
                lora_params.append(param)
            else:
                other_params.append(param)

        param_groups = []
        if other_params:
            param_groups.append({"params": other_params, "lr": base_lr})
        if lora_params:
            param_groups.append({"params": lora_params, "lr": lora_config["lora_lr"]})

        # Create optimizer with parameter groups
        custom_optimizer = torch.optim.AdamW(param_groups, **opt_kwargs)

    # Scheduler
    scheduler_name: str = scheduler_config.pop("name")
    sched_kwargs = dict(scheduler_config)
    log.info(f"Scheduler: {scheduler_name}")
    log.info(f"Scheduler kwargs: {sched_kwargs}")

    # Determine learning rate scheduler type based on config
    lr_scheduler_type = "linear"
    if "constant" in scheduler_name.lower():
        lr_scheduler_type = "constant_with_warmup"
    elif sched_kwargs.get("alpha_f", None) == 0:
        # alpha_f=0 means no decay (constant LR after warmup)
        lr_scheduler_type = "constant_with_warmup"
    elif "linear" in scheduler_name.lower():
        lr_scheduler_type = "linear"
    elif "cosine" in scheduler_name.lower():
        lr_scheduler_type = "cosine"

    # (Skipping composer-style eval-metrics augmentation; evaluators remain as built)

    # Build the HuggingFace Trainer and training arguments
    log.info("Building HuggingFace Trainer...")
    # helper to parse numeric part of duration strings like '1000ba' or '1ep'
    def _parse_int(val):
        if isinstance(val, int):
            return val
        if isinstance(val, str):
            digits = "".join(ch for ch in val if ch.isdigit())
            return int(digits) if digits else None
        return None

    # determine number of epochs or steps
    num_train_epochs = None
    max_steps = None
    if isinstance(max_duration, str) and max_duration.endswith("ep"):
        num_train_epochs = int(max_duration[:-2])
    elif isinstance(max_duration, str) and max_duration.endswith("ba"):
        max_steps = int(max_duration[:-2])
    elif isinstance(max_duration, (int, float)):
        num_train_epochs = int(max_duration)

    # Calculate gradient accumulation steps
    # If microbatch_size is specified and differs from batch_size, compute accumulation steps
    gradient_accumulation_steps = 1
    if device_train_microbatch_size != "auto" and device_train_microbatch_size is not None:
        if isinstance(device_train_microbatch_size, int) and device_train_microbatch_size > 0:
            # gradient_accumulation_steps = batch_size / microbatch_size
            gradient_accumulation_steps = max(1, train_batch_size // (device_train_microbatch_size * world_size))
            if gradient_accumulation_steps > 1:
                log.info(
                    f"Using gradient accumulation: {gradient_accumulation_steps} steps (batch_size={train_batch_size}, microbatch_size={device_train_microbatch_size})"
                )

    per_device_train_batch_size = (
        device_train_microbatch_size
        if isinstance(device_train_microbatch_size, int) and device_train_microbatch_size > 0
        else train_batch_size
    )
    if per_device_train_batch_size != device_train_microbatch_size:
        log.info(
            "device_train_microbatch_size=%s is not an integer; falling back to train_batch_size=%s",
            device_train_microbatch_size,
            train_batch_size,
        )

    hf_fsdp_mode = None
    hf_fsdp_config = None
    if fsdp_config is not None:
        hf_fsdp_mode, hf_fsdp_config = _build_hf_fsdp_args(model, fsdp_config)
        if activation_checkpointing_via_model:
            hf_fsdp_config.pop("activation_checkpointing", None)
        log.info("Using FSDP mode '%s' with config: %s", hf_fsdp_mode, hf_fsdp_config)
        logged_cfg.update(
            {
                "fsdp": hf_fsdp_mode,
                "fsdp_config": hf_fsdp_config,
            },
            merge=True,
        )

    training_args_kwargs = {
        "output_dir": os.path.join(hf_save_path, run_name),
        "per_device_train_batch_size": per_device_train_batch_size,
        "per_device_eval_batch_size": device_eval_batch_size,
        "learning_rate": base_lr,
        "weight_decay": opt_kwargs.get("weight_decay", 0.0),
        "adam_beta1": opt_kwargs.get("betas", [0.9, 0.999])[0] if "betas" in opt_kwargs else 0.9,
        "adam_beta2": opt_kwargs.get("betas", [0.9, 0.999])[1] if "betas" in opt_kwargs else 0.999,
        "adam_epsilon": opt_kwargs.get("eps", 1e-8),
        "optim": optim_type,
        "logging_steps": _parse_int(console_log_interval) or 1,
        # Disable Trainer checkpointing (optimizer/scheduler/FSDP artifacts).
        # We save only the final runnable model explicitly after training.
        "save_strategy": "no",
        "save_only_model": True,
        # "evaluation_strategy": "steps" if eval_dataset is not None else "no",
        "eval_steps": _parse_int(eval_interval),
        "fp16": "fp16" in precision,
        "bf16": "bf16" in precision,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "seed": seed,
        "disable_tqdm": not progress_bar,
        "warmup_steps": _parse_int(sched_kwargs.get("t_warmup", 0)) or 0,
        "lr_scheduler_type": lr_scheduler_type,
    }
    if local_rank >= 0:
        training_args_kwargs["local_rank"] = local_rank
    if hf_fsdp_mode is not None:
        training_args_kwargs["fsdp"] = hf_fsdp_mode
        training_args_kwargs["fsdp_config"] = hf_fsdp_config
    if num_train_epochs is not None:
        training_args_kwargs["num_train_epochs"] = num_train_epochs
    if max_steps is not None:
        training_args_kwargs["max_steps"] = max_steps

    training_args = TrainingArguments(
        **{k: v for k, v in training_args_kwargs.items() if v is not None}
    )
    log.info(
        "TrainingArguments distributed summary: local_rank=%s world_size=%s ddp_find_unused_parameters=%s fsdp=%s fsdp_config=%s",
        training_args.local_rank,
        world_size,
        getattr(training_args, "ddp_find_unused_parameters", None),
        getattr(training_args, "fsdp", None),
        hf_fsdp_config,
    )

    # Build trainer with custom optimizer if LoRA has different LR
    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        # "data_collator": DataCollatorWithPadding(tokenizer=tokenizer, padding=True, return_tensors="pt"),
        # "tokenizer": tokenizer,
    }

    if custom_optimizer is not None:
        trainer_kwargs["optimizers"] = (custom_optimizer, None)  # (optimizer, scheduler)

    print(trainer_kwargs)
    # raise ValueError(trainer_kwargs)

    trainer = HfTrainer(**trainer_kwargs)
    if fsdp_config is not None:
        trainer.add_callback(
            _FSDPStateLoggingCallback(expect_fsdp=world_size > 1, world_size=world_size)
        )
        pre_train_wrapped = _is_fsdp_wrapped_model(getattr(trainer, "model_wrapped", None))
        log.info(
            "Pre-train wrapper state: model_type=%s model_wrapped_type=%s fsdp_wrapped_pre_train=%s "
            "(HF usually wraps at train() time).",
            type(trainer.model).__name__,
            type(getattr(trainer, "model_wrapped", None)).__name__
            if getattr(trainer, "model_wrapped", None) is not None
            else "None",
            pre_train_wrapped,
        )

    if should_log_config:
        log.info("Logging config")
        log_config(logged_cfg)
    torch.cuda.empty_cache()
    gc.collect()

    output_dir = os.path.join(hf_save_path, run_name)
    try:
        log.info("Starting training...")
        trainer.train()

        # All ranks must participate in save under distributed/FSDP.
        trainer.save_model(output_dir)
        if trainer.is_world_process_zero():
            tokenizer.save_pretrained(output_dir)

        log.info(f"Done. Model saved to {output_dir}")
        return trainer
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
