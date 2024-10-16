#!/usr/bin/env bash

gpu_list="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

OUTPUT_DIR=$1
CKPT=$(basename ${OUTPUT_DIR})

SPLIT="OCRBench"
for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.eval_ocrbench \
        --model-path ${OUTPUT_DIR} \
        --question-file .playground/data/OCRBench/$SPLIT.jsonl \
        --image-folder .playground/data/OCRBench/OCRBench_Images \
        --answers-file .playground/data/OCRBench/result/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode vicuna_v1 &
done

wait

output_file=.playground/data/OCRBench/result/$SPLIT/$CKPT/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat .playground/data/OCRBench/result/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python scripts/convert_ocrbench_for_eval.py --src $output_file | tee .playground/data/OCRBench/result/$SPLIT/$CKPT/result.log
