# Convenience script for combining all data preparation, model training
# and model evaluation with json
# All arguments to the python script can be provided
# here exactly in the form they would be passed to the
# python script directly.
#
# Example usage:
# CUDA_VISIBLE_DEVICES=x ./prepare_train_eval.sh user=alonso finetuning=rosa

set -e

# process input arguments
# training_mode="tbd" # training_mode to be determined later.
# test_split="0"
# for argument in "$@"
# do
#     key=$(echo $argument | cut -f1 -d=)
#     if [[ $key == test_split ]]; then
#         test_split=${argument#*=}
#         echo "Setting the test_split to ${test_split}"
#     elif [[ $key == finetuning ]]; then
#         training_mode=${argument#*=}
#         echo "Setting finetuning mode to ${training_mode}"
#     elif [[ $training_mode == "rosa" ]] && [[ $key == finetuning.rosa.masks_only ]];then
#         echo "The 'finetuning.rosa.masks_only' argument is already set and should not be overridden here; override is ignored."
#     else
#         vars[idx]=$argument
#         idx+=1
#     fi
# done

export CUDA_VISIBLE_DEVICES=$1
models=(Qwen/Qwen3.5-4B Qwen/Qwen3.5-9B Qwen/Qwen3.5-0.8B Qwen/Qwen3-4B-Instruct-2507 meta-llama/Llama-3.2-3B-Instruct meta-llama/Llama-3.2-1B-Instruct Qwen/Qwen3.5-27B)
model_names=(Qwen3.5-4B Qwen3.5-9B Qwen3.5-0.8B Qwen3-4B-Instruct-2507 Llama-3.2-3B-Instruct Llama-3.2-1B-Instruct Qwen3.5-27B)
users=(david isabel marcus)
#users=(isabel marcus)
lrs=( 3.6e-6 6.3e-6 1.1e-5 2e-5 3.6e-5 6.3e-5 1.1e-4 2e-4)
lrs=(2e-05 3.6e-05 6.3e-05 0.00011 0.0002)
lrs=( 1.1e-05 6.3e-06 3.6e-06)
lrs=( 3.6e-06 6.3e-06 1.1e-05 2e-05 3.6e-05 6.3e-05 0.00011 0.0002)
lora_ranks=(8 16 32)
batch_sizes=(4 8 16)
batch_sizes=(8)
epochs=(4 5)
epochs=(5 7)
epochs=(9)

cuda_device_index="${CUDA_VISIBLE_DEVICES%%,*}"
if [[ "${cuda_device_index}" =~ ^[0-9]+$ ]] && (( cuda_device_index >= 0 )) && (( cuda_device_index < ${#models[@]} )); then
  model="${models[cuda_device_index]}"
  model_name="${model_names[cuda_device_index]}"
else
  model="${models[0]}"
  model_name="${model_names[0]}"
fi

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


seed=44
for var in "${vars[@]}"; do
  if [[ "${var}" == seed=* ]]; then
    seed="${var#*=}"
    break
  fi
done

for user in "${users[@]}"; do
  for lr in "${lrs[@]}"; do
    for lora_r in "${lora_ranks[@]}"; do
      for batch_size in "${batch_sizes[@]}"; do
        for n_epochs in "${epochs[@]}"; do
          echo "Training user=${user} lr=${lr} lora_rank=${lora_r} batch_size=${batch_size} epochs=${n_epochs}"

         accelerate launch \
            --num_processes "${nproc_per_node}" \
            ../src/panza/finetuning/train_transformers.py \
            finetuning=lora \
            user=${user} \
            finetuning.lr=${lr} \
            finetuning.lora.lr=${lr} \
            finetuning.optimizer.lr=${lr} \
            finetuning.train_batch_size=${batch_size} \
            finetuning.max_duration=${n_epochs}ep \
            finetuning.lora.alpha=${lora_r} \
            finetuning.lora.rank=${lora_r} \
            finetuning.model_name_or_path=${model} 

          echo "Generating json evaluation for user=${user} lr=${lr} lora_rank=${lora_r} batch_size=${batch_size} epochs=${n_epochs}"
          python runner.py \
            interfaces=json \
            writer/llm=peft \
            user=${user} \
            checkpoint=/nfs/scistore19/alistgrp/eiofinov/PanzaMail/scripts/..//checkpoints/models/panza_${user}_anonymous-${model_name}-bf16-bs${batch_size}-lora-r${lora_r}-lr${lr}-${n_epochs}ep-seed${seed} \
            interfaces.input_file=/nfs/scistore19/alistgrp/eiofinov/PanzaMail/data/${user}_anonymous/test.jsonl

        done
      done
    done
  done
done
