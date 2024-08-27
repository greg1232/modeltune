#!/bin/bash

# Define variables
NNODES=1
NPROC_PER_NODE=8
MODEL_NAME="meta-llama/Llama-Guard-3-8B"
DATASET="mlc_dataset_lg3"
OUTPUT_DIR="trained_model_checkpoints"
EXPERIMENT_NAME="MLC_V_LG3_05_HP"
PATH_TO_FINETUNE="model/mlc_llama_guard/meta_llama_recipes/recipes/quickstart/finetuning/"
VAL_BATCH_SIZE=4

OUTPUT_DIR="$OUTPUT_DIR/$EXPERIMENT_NAME"

# Create the output directory if not already created
if [ ! -d $OUTPUT_DIR ]; then
  mkdir -p $OUTPUT_DIR;
fi

# For multi GPU parameter efficient finetuning
torchrun --nnodes $NNODES --nproc_per_node $NPROC_PER_NODE $PATH_TO_FINETUNE/finetuning.py \
  --enable_fsdp \
  --model_name $MODEL_NAME \
  --pure_bf16 \
  --use_fast_kernels \
  --use_peft \
  --batch_size_training 6 \
  --peft_method lora \
  --num_workers_dataloader 64 \
  --val_batch_size $VAL_BATCH_SIZE \
  --dataset $DATASET \
  --output_dir $OUTPUT_DIR \
  --save_metrics \
  --save_model