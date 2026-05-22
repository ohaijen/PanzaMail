#!/usr/bin/env bash
# This repository variant is MLX/MPS-focused and does not include RoSA support.

set -e

vars=()
idx=1

# process input arguments
for argument in "$@"
do
   key=$(echo $argument | cut -f1 -d=)
    vars[idx]=$argument
    idx+=1
done

echo "RoSA training is not supported in this repository."
echo "Use one of the following instead:"
echo "  - ./train_lora.sh      (Hugging Face Transformers + PEFT on MPS/CUDA)"
echo "  - ./train_lora_mlx.sh  (native MLX LoRA on Apple Silicon)"

python3 ../src/panza/finetuning/train_lora_hf.py \
    finetuning=lora "${vars[@]}"
