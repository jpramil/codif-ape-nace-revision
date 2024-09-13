from transformers import AutoTokenizer, AutoModelForCausalLM
import os


def get_model(model_name: str, device: str = "cuda") -> tuple:
    """
    Initializes a HuggingFace tokenizer and model.

    Parameters:
    ----------
    model_name : str
        The name or path of the pre-trained model to load from the HuggingFace Hub.
        For example, 'meta-llama/Meta-Llama-3.1-8B-Instruct'.

    device : str, optional, default="cuda"
        The device on which to load the model. Typically, this will be 'cuda' for GPUs or 'cpu' for CPU execution.
    Returns:
    -------
    tokenizer : transformers.PreTrainedTokenizer
        The tokenizer associated with the specified model, used for encoding and decoding text.

    model : transformers.PreTrainedModel
        The pre-trained causal language model loaded on the specified device.
    """

    hf_token = os.getenv("HF_TOKEN")

    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(model_name, token=hf_token).to(device)

    return tokenizer, model
