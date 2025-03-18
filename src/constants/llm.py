LLM_MODEL = "Qwen/Qwen2.5-32B-Instruct"
MAX_NEW_TOKEN = 100
TEMPERATURE = 0.01

MODEL_TO_ARGS = {
    "mistralai/Ministral-8B-Instruct-2410": {
        "tokenizer_mode": "mistral",
        "config_format": "mistral",
        "load_format": "mistral",
    },
    "mistralai/Mistral-Small-3.1-24B-Instruct-2503": {
        "tokenizer_mode": "mistral",
        "config_format": "mistral",
        "load_format": "mistral",
    },
    "Qwen/Qwen2.5-32B-Instruct": {
        "max_model_len": 20000,
        "gpu_memory_utilization": 0.95,
        "enforce_eager": True,
    },
    "google/gemma-3-27b-it": {
        "gpu_memory_utilization": 1.0,
        "enable_chunked_prefill": True,
    },
}
