#!/usr/bin/env bash

# Convenience script for running MLX LoRA finetuning on Apple Silicon.
# All arguments to the python script can be provided
# here exactly in the form they would be passed to the
# python script directly.
#
# Example usage:
# ./train_lora_mlx.sh user=alonso finetuning.max_duration=3ep

set -e

vars=()
idx=1

# process input arguments
for argument in "$@"
do
   key=$(echo $argument | cut -f1 -d=)

   if [[ $key == finetuning ]]; then
    echo "The 'finetuning' argument is already set and should not be overridden here; override is ignored."
   else
    vars[idx]=$argument
    idx+=1
   fi
done

python3 ../src/panza/finetuning/train_lora_mlx.py \
    finetuning=lora "${vars[@]}"
