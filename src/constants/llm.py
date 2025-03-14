LLM_MODEL = "Qwen/Qwen2.5-32B-Instruct"
MAX_NEW_TOKEN = 100
TEMPERATURE = 0.01
TOP_P = 0.8
REP_PENALTY = 1.05

MODEL_TO_ARGS = {
    "mistralai/Mistral-7B-Instruct-v0.3": {
        "tokenizer_mode": "mistral",
        "config_format": "mistral",
        "load_format": "mistral",
        "enforce_eager": True,
    },
    "mistralai/Ministral-8B-Instruct-2410": {
        "tokenizer_mode": "mistral",
        "config_format": "mistral",
        "load_format": "mistral",
        "enforce_eager": True,
    },
    "mistralai/Mistral-Small-Instruct-2409": {
        "tokenizer_mode": "mistral",
        "config_format": "mistral",
        "load_format": "mistral",
        "enforce_eager": True,
    },
    "mistralai/Mistral-Small-24B-Instruct-2501": {
        "tokenizer_mode": "mistral",
        "config_format": "mistral",
        "load_format": "mistral",
        "enforce_eager": True,
    },
    "Qwen/Qwen2.5-1.5B-Instruct": {},
    "Qwen/Qwen2.5-32B-Instruct": {
        "max_model_len": 20000,
        "gpu_memory_utilization": 0.95,
        "enforce_eager": True,
    },
    "Qwen/Qwen2.5-72B-Instruct-GPTQ-Int4": {
        "max_model_len": 8192,
        "gpu_memory_utilization": 1.0,
        "enforce_eager": True,
        "enable_chunked_prefill": True,
        "max_num_batched_tokens": 2048,
    },
    "meta-llama/Meta-Llama-3.1-8B-Instruct": {},
    "google/gemma-3-27b-it": {
        "gpu_memory_utilization": 1.0,
        "enforce_eager": True,
        "enable_chunked_prefill": True,
    },
}
