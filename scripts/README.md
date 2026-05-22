# Scripts

This directory contains the supported training and serving scripts for PanzaMail-MLX.

## Supported Scripts

- `prepare_hf_panza_emails.py`
  - Builds `train.jsonl`, `valid.jsonl`, `test.jsonl` from `ISTA-DASLab/Panza-emails`.
  - Output format is prompt/completion JSONL for LoRA training.

- `train_lora_mlx.sh`
  - Runs LoRA training through `src/panza/finetuning/train_lora_mlx.py`.
  - Backend: `mlx_lm` (Apple Silicon / Metal).

- `train_lora.sh`
  - Runs LoRA training through `src/panza/finetuning/train_lora_hf.py`.
  - Backend: Hugging Face Transformers + PEFT (MPS/CUDA/CPU).

- `runner.py` / `runner.sh`
  - Panza serving entrypoint (CLI/GUI/JSON/Web per config).

## Unsupported in This Fork

- `train_rosa.sh`
  - Present only as an explicit unsupported stub.
  - RoSA/llm-foundry/spops dependencies are not part of PanzaMail-MLX.

## Typical Workflow

1. Prepare dataset:

```bash
python3 scripts/prepare_hf_panza_emails.py \
  --config david \
  --merge-splits \
  --out-dir data/panza_emails_hf_david
```

2. Train with MLX:

```bash
cd scripts
./train_lora_mlx.sh \
  user.data_dir=../data/panza_emails_hf_david \
  finetuning.model_name_or_path=mlx-community/Qwen3-4B-4bit \
  finetuning.run_name=panza-david-qwen3-4b-4bit-lora
```

Alternative (HF/PEFT on MPS/CUDA):

```bash
cd scripts
./train_lora.sh \
  user.data_dir=../data/panza_emails_hf_david \
  finetuning.model_name_or_path=Qwen/Qwen3-4B-Instruct-2507 \
  finetuning.run_name=panza-david-qwen3-4b-hf-lora
```

3. Generate with adapter:

```bash
python3 -m mlx_lm generate \
  --model mlx-community/Qwen3-4B-4bit \
  --adapter-path checkpoints/models/panza-david-qwen3-4b-4bit-lora \
  --prompt "Write a concise professional email for this subject: Meeting follow-up"
```
