#!/bin/bash

pip install -r requirements.txt

pre-commit install

MODEL_NAME=mistralai/Ministral-8B-Instruct-2410
LOCAL_PATH=~/.cache/huggingface/hub

./bash/fetch_model_s3.sh $MODEL_NAME $LOCAL_PATH
