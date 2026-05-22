#!/usr/bin/env bash
# This repository variant is MLX/MPS-focused and does not include RoSA support.

set -e

echo "RoSA training is not supported in this repository."
echo "Use one of the following instead:"
echo "  - ./train_lora.sh      (Hugging Face Transformers + PEFT on MPS/CUDA)"
echo "  - ./train_lora_mlx.sh  (native MLX LoRA on Apple Silicon)"
exit 1
