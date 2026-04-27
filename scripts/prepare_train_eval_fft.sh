#####
# Convenience script for combining all data preparation, model training
# and model evaluation with json
# All arguments to the python script can be provided
# here exactly in the form they would be passed to the
# python script directly.
#
# Example usage:
# CUDA_VISIBLE_DEVICES=x ./prepare_train_eval.sh user=alonso finetuning=rosa

#set -e


export CUDA_VISIBLE_DEVICES=$1
models=(Qwen/Qwen3.5-4B Qwen/Qwen3.5-9B Qwen/Qwen3.5-0.8B Qwen/Qwen3-4B-Instruct-2507 meta-llama/Llama-3.2-3B-Instruct meta-llama/Llama-3.2-1B-Instruct Qwen/Qwen3.5-27B)
model_names=(Qwen3.5-4B Qwen3.5-9B Qwen3.5-0.8B Qwen3-4B-Instruct-2507 Llama-3.2-3B-Instruct Llama-3.2-1B-Instruct Qwen3.5-27B)
users=(snippets2)
#users=(snippets3)
#users=(isabel marcus)
lrs=( 3.6e-6 6.3e-6 1.1e-5 2e-5 3.6e-5 6.3e-5 1.1e-4 2e-4)
lrs=(2e-05 3.6e-05 6.3e-05 0.00011 0.0002)
lrs=( 1.1e-05 6.3e-06 3.6e-06)
lrs=( 3.6e-06 6.3e-06 1.1e-05 2e-05 3.6e-05 6.3e-05 0.00011 0.0002)
lrs=( 3.6e-06 6.3e-06 1.1e-05 2e-05 3.6e-05 6.3e-05 0.00011)
lrs=( 3.6e-06 6.3e-06 1.1e-05 2e-05 3.6e-05) # 6.3e-05 0.00011)
#lrs=(2e-05 3.6e-05) # 6.3e-05 0.00011)
#lrs=( 1.1e-05)
batch_sizes=(4 8 16)
batch_sizes=(6)
epochs=(4 5)
#batch_sizes=(8)
#epochs=(4)

# cuda_device_index="${CUDA_VISIBLE_DEVICES%%,*}"
# if [[ "${cuda_device_index}" =~ ^[0-9]+$ ]] && (( cuda_device_index >= 0 )) && (( cuda_device_index < ${#models[@]} )); then
#   model="${models[cuda_device_index]}"
#   model_name="${model_names[cuda_device_index]}"
# else
#   model="${models[0]}"
#   model_name="${model_names[0]}"
# fi
model="${models[$2]}"
model_name="${model_names[$2]}"

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


seed=41
for var in "${vars[@]}"; do
  if [[ "${var}" == seed=* ]]; then
    seed="${var#*=}"
    break
  fi
done

for user in "${users[@]}"; do
  for lr in "${lrs[@]}"; do
    for batch_size in "${batch_sizes[@]}"; do
      for n_epochs in "${epochs[@]}"; do
        filename=/nfs/scistore19/alistgrp/eiofinov/PanzaMail-snippets/scripts/..//checkpoints/models/panza_${user}-${model_name}-bf16-bs${batch_size}-fft-lr${lr}-${n_epochs}ep-seed${seed}
        echo $filename

        if [[ ! -e "${filename}/model.safetensors" ]]; then
        echo "Training user=${user} lr=${lr} batch_size=${batch_size} epochs=${n_epochs}"
        accelerate launch \
          --num_processes "${nproc_per_node}" \
          ../src/panza/finetuning/train_transformers.py \
          finetuning=full \
          user=${user} \
          finetuning.lr=${lr} \
          finetuning.optimizer.lr=${lr} \
          finetuning.train_batch_size=${batch_size} \
          finetuning.max_duration=${n_epochs}ep \
          finetuning.model_name_or_path=${model} 

        elif [[ ! -e "${filename}/test_outputs.json" ]]; then
        echo "Generating json evaluation for user=${user} lr=${lr} batch_size=${batch_size} epochs=${n_epochs}"
        python runner.py \
          interfaces=json \
          writer/llm=transformers \
          user=${user} \
          checkpoint="${filename}" \
          interfaces.input_file=/nfs/scistore19/alistgrp/eiofinov/PanzaMail-snippets/data/${user}/test.jsonl
        
        else
          echo "Skipping training user=${user} lr=${lr} batch_size=${batch_size} epochs=${n_epochs}; ${filename} already exists"
        fi

      done
    done
  done
done
