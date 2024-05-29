#!/bin/bash

MODEL_PATH=./checkpoints/llava-mmfuser

python -m llava.serve.inference \
     --model-path ${MODEL_PATH} \
     --image-file ./images/example.png \
     --from-human "Describe this image." "What are the people in the image doing?" "How many people are there in the image?"\
     --load-4bit
