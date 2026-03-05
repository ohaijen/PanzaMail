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
from typing import Any, Dict, List, Optional, Union

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
from transformers import Trainer as HfTrainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset
from torch.utils.data import DataLoader
from dataclasses import dataclass
from typing import Dict

install()
# If certain ffn types require special handling, list them here (empty for now)
ffns_with_megablocks = []

@dataclass
class CustomDataCollator:
    """Custom collator that pads sequences to max length in batch."""
    tokenizer: PreTrainedTokenizerBase
    mlm: bool = False

    def __call__(self, batch: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        """Pad batch to max sequence length."""
        # Find max length in this batch
        max_len = max(len(item["input_ids"]) for item in batch)
        
        # Pad all sequences to max length
        padded_batch = {
            "input_ids": [],
            "labels": [],
        }
        if "attention_mask" in batch[0]:
            padded_batch["attention_mask"] = []
        
        for item in batch:
            pad_len = max_len - len(item["input_ids"])
            padded_batch["input_ids"].append(
                item["input_ids"] + [self.tokenizer.pad_token_id] * pad_len
            )
            padded_batch["labels"].append(
                item["labels"] + [-100] * pad_len  # -100 tokens are ignored in loss
            )
            if "attention_mask" in item:
                padded_batch["attention_mask"].append(
                    item["attention_mask"] + [0] * pad_len
                )
        
        # Convert to tensors
        return {
            k: torch.tensor(v) if isinstance(v, list) else v 
            for k, v in padded_batch.items()
        }

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
    def tokenize_fn(examples):
        text = examples.get("prompt", "") + examples.get("response", "")
        tokens = tokenizer(
            text,
            truncation=True,
            max_length=max_len,
        )
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens
    dataset = dataset.map(tokenize_fn, batched=True, remove_columns=dataset.column_names)
    if is_train and ds_conf.get("shuffle", True):
        dataset = dataset.shuffle()
    return dataset


def _make_dataloader(dataset, tokenizer, batch_size: int, is_train: bool = True, loader_cfg: DictConfig = None):
    collator = CustomDataCollator(tokenizer=tokenizer, mlm=False)
    extra = {}
    if loader_cfg is not None:
        # pass through supported dataloader args
        for arg in ["num_workers", "pin_memory", "prefetch_factor", "persistent_workers", "drop_last"]:
            if arg in loader_cfg:
                extra[arg] = loader_cfg[arg]
    return DataLoader(dataset, batch_size=batch_size, shuffle=is_train, collate_fn=collator, **extra)

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
    #             "`te.Linear` layers do not support activation_checkpointing with "
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
        use_auth_token=True,
        use_cache=False,
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

    # Initialize pytorch distributed training process groups
    dist_timeout: Union[int, float] = pop_config(
        cfg, "dist_timeout", must_exist=False, default_value=600.0
    )
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1 and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        try:
            dist.init_process_group(backend=backend, timeout=timedelta(seconds=dist_timeout))
        except Exception:
            pass

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
    #log_to_console: bool = pop_config(cfg, "log_to_console", must_exist=False, default_value=True)
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

    # Warn if fsdp is enabled but user only has 1 GPU
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    if world_size == 1 and fsdp_config is not None:
        warnings.warn("FSDP is not applicable for single-GPU training. Reverting to DDP.")
        fsdp_config = None

    # set logging level
    if python_log_level is not None:
        rank = dist.get_rank() if dist.is_initialized() else 0
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
    #loggers = []
    #mosaicml_logger = None
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
        # train_loader = _make_dataloader(train_dataset, tokenizer, device_train_batch_size, is_train=True, loader_cfg=train_loader_config)
    except Exception as e:
        # if mosaicml_logger is not None:
        #     mosaicml_logger.log_exception(e)
        raise e

    # if mosaicml_logger is not None:
    #     mosaicml_logger.log_metrics({"data_validated": time.time()})

    # Evaluation dataset
    eval_dataset = None
    if eval_loader_config is not None:
        log.info("Building eval dataset...")
        eval_dataset = _load_and_prepare_dataset(
            eval_loader_config,
            tokenizer,
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
            digits = ''.join(ch for ch in val if ch.isdigit())
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
            gradient_accumulation_steps = max(1, train_batch_size // device_train_microbatch_size)
            if gradient_accumulation_steps > 1:
                log.info(f"Using gradient accumulation: {gradient_accumulation_steps} steps (batch_size={train_batch_size}, microbatch_size={device_train_microbatch_size})")

    training_args_kwargs = {
        "output_dir": os.path.join(hf_save_path, run_name),
        "per_device_train_batch_size": device_train_microbatch_size,
        "per_device_eval_batch_size": device_eval_batch_size,
        "learning_rate": base_lr,
        "weight_decay": opt_kwargs.get("weight_decay", 0.0),
        "adam_beta1": opt_kwargs.get("betas", [0.9, 0.999])[0] if "betas" in opt_kwargs else 0.9,
        "adam_beta2": opt_kwargs.get("betas", [0.9, 0.999])[1] if "betas" in opt_kwargs else 0.999,
        "adam_epsilon": opt_kwargs.get("eps", 1e-8),
        "optim": optim_type,
        "logging_steps": _parse_int(console_log_interval) or 1,
        "save_steps": _parse_int(save_interval),
        "evaluation_strategy": "steps" if eval_dataset is not None else "no",
        "eval_steps": _parse_int(eval_interval),
        "fp16": "fp16" in precision,
        "bf16": "bf16" in precision,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "save_total_limit": save_num_checkpoints_to_keep if save_num_checkpoints_to_keep > 0 else None,
        "seed": seed,
        "disable_tqdm": not progress_bar,
        "warmup_steps": _parse_int(sched_kwargs.get("t_warmup", 0)) or 0,
        "lr_scheduler_type": lr_scheduler_type,
    }
    if num_train_epochs is not None:
        training_args_kwargs["num_train_epochs"] = num_train_epochs
    if max_steps is not None:
        training_args_kwargs["max_steps"] = max_steps

    training_args = TrainingArguments(**{k: v for k, v in training_args_kwargs.items() if v is not None})
    
    # Build trainer with custom optimizer if LoRA has different LR
    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "data_collator": CustomDataCollator(tokenizer=tokenizer, mlm=False),
        "tokenizer": tokenizer,
    }
    
    if custom_optimizer is not None:
        trainer_kwargs["optimizers"] = (custom_optimizer, None)  # (optimizer, scheduler)
    
    trainer = HfTrainer(**trainer_kwargs)

    if should_log_config:
        log.info("Logging config")
        log_config(logged_cfg)
    torch.cuda.empty_cache()
    gc.collect()

    log.info("Starting training...")
    trainer.train()

    # Save final model and tokenizer
    rank = dist.get_rank() if dist.is_initialized() else 0
    if rank == 0:
        output_dir = os.path.join(hf_save_path, run_name)
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)

    log.info(f"Done. Model saved to {output_dir}")
    return trainer


if __name__ == "__main__":
    main()
