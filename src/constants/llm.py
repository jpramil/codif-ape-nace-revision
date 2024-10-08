LLM_MODEL = "Qwen/Qwen2.5-32B-Instruct"
MAX_NEW_TOKEN = 100
TEMPERATURE = 0.01
TOP_P = 0.8
REP_PENALTY = 1.05

MODEL_TO_ARGS = {
    "mistralai/Mistral-7B-Instruct-v0.3": {"tokenizer_mode":"mistral", "config_format":"mistral", "load_format":"mistral"},
    "Qwen/Qwen2.5-1.5B-Instruct": {},
    "Qwen/Qwen2.5-32B-Instruct": {"max_model_len": 20000, "gpu_memory_utilization": 0.95},
}