# 📌 CAG vs RAG: Centralized Repository for NACE Revision

## 📑 Table of Contents

- [📌 CAG vs RAG: Centralized Repository for NACE Revision](#-cag-vs-rag-centralized-repository-for-nace-revision)
  - [📑 Table of Contents](#-table-of-contents)
  - [📖 Overview](#-overview)
  - [🚀 Getting Started](#-getting-started)
    - [🛠 Installation](#-installation)
    - [🏗 Pre-commit Setup](#-pre-commit-setup)
    - [Cache LLM model from S3 Bucket (Optionnal)](#cache-llm-model-from-s3-bucket-optionnal)
  - [📜 Running the Scripts](#-running-the-scripts)
    - [✅ 1. Build Vector Database (if you are in the RAG case)](#-1-build-vector-database-if-you-are-in-the-rag-case)
    - [🏷 2. Encode Business Activity Codes](#-2-encode-business-activity-codes)
    - [🔬 3. Evaluate Classification Strategies](#-3-evaluate-classification-strategies)
    - [📊 4. Build the NACE 2025 Dataset](#-4-build-the-nace-2025-dataset)
  - [📡 LLM Integration](#-llm-integration)
  - [🏗 Argo Workflows](#-argo-workflows)
  - [📄 License](#-license)


## 📖 Overview
This repository is dedicated to the revision of the **Nomenclature statistique des Activités économiques dans la Communauté Européenne (NACE)**.

It provides tools for **automated classification and evaluation of business activity codes** using **Large Language Models (LLMs)** and vector-based retrieval systems.


## 🚀 Getting Started

### 🛠 Installation
Ensure you have **Python 3.12** and **uv**:

```bash
uv sync
```

### 🏗 Pre-commit Setup
Set up linting and formatting checks using `pre-commit`:

```bash
uv run pre-commit install
```

### Cache LLM model from S3 Bucket (Optionnal)
If you want to use a model available in the SSPCloud you can execute this command:

```bash
MODEL_NAME=mistralai/Ministral-8B-Instruct-2410
LOCAL_PATH=~/.cache/huggingface/hub

./bash/fetch_model_s3.sh $MODEL_NAME $LOCAL_PATH
```

## 📜 Running the Scripts

### ✅ 1. Build Vector Database (if you are in the RAG case)

To create a searchable database of NACE 2025 codes:

```bash
uv run build-vector-db.py
```

### 🏷 2. Encode Business Activity Codes

For **unambiguous** classification:

```bash
uv run encode-univoque.py
```

For **ambiguous** classification using an LLM:

```bash
uv run encode-multivoque.py --experiment_name NACE2025_DATASET --llm_name Ministral-8B-Instruct-2410
```

### 🔬 3. Evaluate Classification Strategies

Compare different classification models:

```bash
uv run evaluate_strategies.py
```

### 📊 4. Build the NACE 2025 Dataset

Once all unique ambiguous cases have been recoded using the best strategy, you can rebuild the entire dataset with NACE 2025 labels:

```bash
uv run build_nace2025_sirene4.py
```

## 📡 LLM Integration
This repository leverages **Large Language Models (LLMs)** to assist in classifying business activities. The supported models include are the one available on the SSPCloud platform. One can also use a model directly from HuggingFace.


## 🏗 Argo Workflows
This project supports **automated workflows** via [Argo Workflows](https://argoproj.github.io/argo-workflows/).
To trigger a workflow, execute:

```yaml
argo submit argo-workflows/relabel-naf08-to-naf25.yaml
```

Or use the **Argo Workflow UI**.


## 📄 License
This project is licensed under the **Apache License 2.0**. See the [LICENSE](LICENSE) file for details.
