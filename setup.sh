#!/bin/bash
uv sync
uv run pre-commit install

export MODEL_NAME=mistralai/Ministral-8B-Instruct-2410
export LOCAL_PATH=/home/onyxia/.cache/huggingface/hub
export QDRANT_API_KEY=***

export VLLM_USE_V1=1
export VLLM_USE_FLASHINFER_SAMPLER=0
# export TORCH_CUDA_ARCH_LIST=9.0

./bash/fetch_model_s3.sh $MODEL_NAME $LOCAL_PATH
