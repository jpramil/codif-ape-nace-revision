#!/bin/bash
pip install uv
uv pip install -r requirements.txt

pre-commit install

export MODEL_NAME=mistralai/Ministral-8B-Instruct-2410
export LOCAL_PATH=/home/onyxia/.cache/huggingface/hub
export QDRANT_API_KEY=***

./bash/fetch_model_s3.sh $MODEL_NAME $LOCAL_PATH
