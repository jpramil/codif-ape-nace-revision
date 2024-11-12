import os
import subprocess

from transformers import AutoModelForCausalLM

from src.utils.data import get_file_system


def cache_model_from_hf_hub(
    model_name,
    s3_bucket="models-hf",
    s3_cache_dir="hf_hub",
    s3_token=None,
    hf_token=None
):
    """Use S3 as proxy cache from HF hub if a model is not already cached locally.

    Args:
        model_name (str): Name of the model on the HF hub.
        s3_bucket (str): Name of the S3 bucket to use.
        s3_cache_dir (str): Path of the cache directory on S3.
    """
    assert (
        "MC_HOST_s3" in os.environ
    ), "Please set the MC_HOST_s3 environment variable."

    # Local cache config
    LOCAL_HF_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
    model_name_hf_cache = "models--" + "--".join(model_name.split("/"))
    dir_model_local = os.path.join(LOCAL_HF_CACHE_DIR, model_name_hf_cache)

    # Remote cache config
    fs = get_file_system(token=s3_token)
    available_models_s3 = [
        os.path.basename(path) for path in fs.ls(os.path.join(s3_bucket, s3_cache_dir))
    ]
    dir_model_s3 = os.path.join(s3_bucket, s3_cache_dir, model_name_hf_cache)

    if model_name_hf_cache not in os.listdir(LOCAL_HF_CACHE_DIR):
        # Try fetching from S3 if available
        if model_name_hf_cache in available_models_s3:
            print(f"Fetching model {model_name} from S3.")
            cmd = [
                "mc",
                "cp",
                "-r",
                f"s3/{dir_model_s3}",
                f"{LOCAL_HF_CACHE_DIR}/"
            ]
            with open("/dev/null", "w") as devnull:
                subprocess.run(cmd, check=True, stdout=devnull, stderr=devnull)
        # Else, fetch from HF Hub and push to S3
        else:
            print(f"Model {model_name} not found on S3, fetching from HF hub.")
            AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype="auto",
                token=hf_token
            )
            print(f"Putting model {model_name} on S3.")
            cmd = [
                "mc",
                "cp",
                "-r",
                f"{dir_model_local}/",
                f"s3/{dir_model_s3}",           
            ]
            with open("/dev/null", "w") as devnull:
                subprocess.run(cmd, check=True, stdout=devnull, stderr=devnull)
    else:
        print(f"Model {model_name} found in local cache. ")
        if model_name_hf_cache not in available_models_s3:
            # Push from local HF cache to S3
            print(f"Putting model {model_name} on S3.")
            cmd = [
                "mc",
                "cp",
                "-r",
                f"{dir_model_local}/",
                f"s3/{dir_model_s3}",
            ]
            with open("/dev/null", "w") as devnull:
                subprocess.run(cmd, check=True, stdout=devnull, stderr=devnull)
