# Convenience script for running RoSA finetuning.
# All arguments to the python script can be provided
# here exactly in the form they would be passed to the
# python script directly.
#
# Example usage:
# ./train_rosa.sh user=alonso trainer.optimizer.lr=0.1

# Certain parameters are saved here: /nfs/scistore19/alistgrp/eiofinov/.cache/huggingface/accelerate/default_config.yaml 

set -e

vars=()
idx=1

# process input arguments
for argument in "$@"
do
   key=$(echo $argument | cut -f1 -d=)
   
   if [[ $key == finetuning ]]; then
    echo "The 'finetuning' argument is already set and should not be overridden here; override is ignored."
   elif [[ $key == finetuning.rosa.masks_only ]]; then
    echo "The 'finetuning.rosa.masks_only' argument is already set and should not be overridden here; override is ignored."
   else
    vars[idx]=$argument
    idx+=1
   fi
done

# torchrun defaults to a single process unless nproc_per_node is specified.
# Prefer explicit override via env, otherwise derive from visible GPUs.
if [[ -n "${NPROC_PER_NODE}" ]]; then
  nproc_per_node="${NPROC_PER_NODE}"
elif [[ -n "${CUDA_VISIBLE_DEVICES}" ]]; then
  # Count comma-separated devices in CUDA_VISIBLE_DEVICES
  IFS=',' read -r -a visible_devices <<< "${CUDA_VISIBLE_DEVICES}"
  nproc_per_node="${#visible_devices[@]}"
else
  nproc_per_node=$(python -c "import torch; print(torch.cuda.device_count())")
fi

if [[ -z "${nproc_per_node}" || "${nproc_per_node}" -lt 1 ]]; then
  nproc_per_node=1
fi

echo "Launching accelerate with num_processes=${nproc_per_node}"

# Then train the weights.
accelerate launch \
    --num_processes "${nproc_per_node}" \
    ../src/panza/finetuning/train_transformers.py \
    finetuning=lora ${vars[@]}
