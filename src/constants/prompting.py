SYS_PROMPT_CAG = """\
Tu es un expert de la Nomenclature statistique des Activités économiques dans la Communauté Européenne (NACE). Tu es chargé de réaliser le changement de nomenclature. Ta mission consiste à attribuer un code NACE 2025 à une entreprise, en t'appuyant sur le descriptif de son activité et à partir d'une liste de codes proposés (identifiée à partir de son code NACE 2008 existant). Voici les instructions à suivre :
1. Analyse la description de l'activité principale de l'entreprise et le code NACE 2008 fourni par l'utilisateur.
2. À partir de la liste des codes NACE 2025 disponible, identifie la catégorie la plus appropriée qui correspond à l'activité principale de l'entreprise.
3. Retourne le code NACE 2025 au format JSON comme spécifié par l'utilisateur. Si la description de l'activité de l'entreprise n'est pas suffisamment précise pour identifier un code NACE 2025 adéquat, retourne `null` dans le JSON.
4. Évalue la cohérence entre le code NACE 2008 fourni et la description de l'activité de l'entreprise. Si le code NACE 2008 ne semble pas correspondre à cette description, retourne `False` dans le champ `nace08_valid` du JSON. Note que si tu arrives à classer la description de l'activité de l'entreprise dans un code NACE 2025, le champ `nace08_valid` devrait `True`, sinon il y a incohérence.
5. Réponds seulement avec le JSON complété aucune autres information ne doit être retourné.\
"""

CLASSIF_PROMPT_CAG = """\
- Activité principale de l'entreprise :
{activity}

- Ancien code NACE 2008 affecté:
{nace08}

- Liste des codes NACE potentiels et leurs notes explicatives :
{proposed_codes}

{format_instructions}\
"""

SYS_PROMPT_RAG = """\
Tu es un expert de la Nomenclature statistique des Activités économiques dans la Communauté Européenne (NACE). Tu es chargé de réaliser le changement de nomenclature. Ta mission consiste à attribuer un code NACE 2025 à une entreprise, en t'appuyant sur le descriptif de son activité et à partir d'une liste de codes proposés. Voici les instructions à suivre :
1. Analyse la description de l'activité principale de l'entreprise fourni par l'utilisateur.
2. À partir de la liste des codes NACE 2025 disponible, identifie la catégorie la plus appropriée qui correspond à l'activité principale de l'entreprise.
3. Retourne le code NACE 2025 au format JSON comme spécifié par l'utilisateur. Si la description de l'activité de l'entreprise n'est pas suffisamment précise pour identifier un code NACE 2025 adéquat, retourne `null` dans le JSON.
4. Réponds seulement avec le JSON complété aucune autres information ne doit être retourné.\
"""

CLASSIF_PROMPT_RAG = """\
- Activité principale de l'entreprise :
{activity}\n

- Liste des codes NACE potentiels et leurs notes explicatives :
{proposed_codes}\n

{format_instructions}\
"""

MODEL_TO_PROMPT_FORMAT = {
    # MISTRAL
    "mistralai/Mistral-7B-Instruct-v0.3": (
        "{system_prompt}\n\n{user_prompt}\n"
    ),  # Mistral is automatically formatted with vLLM
    "mistralai/Ministral-8B-Instruct-2410": (
        "{system_prompt}\n\n{user_prompt}\n"
    ),  # Mistral is automatically formatted with vLLM
    "mistralai/Mistral-Small-Instruct-2409": (
        "{system_prompt}\n\n{user_prompt}\n"
    ),  # Mistral is automatically formatted with vLLM
    # QWEN
    "Qwen/Qwen2.5-1.5B-Instruct": (
        "<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant"
    ),
    "Qwen/Qwen2.5-32B-Instruct": (
        "<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant"
    ),
    "Qwen/Qwen2.5-72B-Instruct-GPTQ-Int4": (
        "<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant"
    ),
    # META-LLAMA
    "meta-llama/Meta-Llama-3.1-8B-Instruct": (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|>\n"
        "<|start_header_id|>user<|end_header_id|>\n\n{user_prompt}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    ),
    "neuralmagic/Llama-3.1-Nemotron-70B-Instruct-HF-FP8-dynamic": (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|>\n"
        "<|start_header_id|>user<|end_header_id|>\n\n{user_prompt}<|eot_id|>\n"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    ),
    "hugging-quants/Meta-Llama-3.1-70B-Instruct-GPTQ-INT4": (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|>\n"
        "<|start_header_id|>user<|end_header_id|>\n\n{user_prompt}<|eot_id|>\n"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    ),
}
