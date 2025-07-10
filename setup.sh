#!/bin/bash
uv sync
uv run pre-commit autoupdate
uv run pre-commit install

export MODEL_NAME=Qwen/Qwen2.5-0.5B

uv run huggingface-cli download $MODEL_NAME
