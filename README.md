This is anonymious repository that contains code for the paper **Think When Needed: Model-Aware Reasoning Routing for LLM-based Ranking** submitted to WWW 2026.

# Modules Overview
## ./data_process/
Purpose: We offer an example for processing three public datasets: MovieLens, Amazon-Game and Rankzephyr_IR.

## unified_runner.py
Purpose: A one-stop runner integrating the entire workflow — from prompt generation and LLM inference to feature extraction, checklist probing, and embedding recomputation.

Usage:

python unified_runner.py --dataset ml-1m --model Qwen3-4B --enable_embed --enable_checklist

Key features:

● Unified argument control (generation, embedding, checklist)

● Dataset-agnostic (MovieLens, Amazon, IR benchmarks)

● Single LLM forward pass (no vLLM required)

## template.py

Purpose: Builds consistent, dataset-aware prompt templates for both recommendation and IR tasks.

Usage:

Imported automatically by unified_runner.py; can also be used independently to construct structured prompts with system/user roles.

Example:

from template import generate_llm_inputs

## pipeline_utils.py
Purpose: Contains core helper functions shared across all modules — prompt formatting, token mapping, embedding pooling, and feature extraction routines.

Usage:

Automatically called within unified_runner.py and item_embed_extractor.

## regression_utils.py
   
Purpose: Provides lightweight configuration dataclasses, helper utilities, and regression helpers (advantage labeling, LightGBM regression, isotonic calibration, and Pareto-based policy selection).

Used by:regression_runner.py

## regression_runner.py

Purpose: Trains and evaluates the Adaptive CoT Router, learning when to invoke reasoning to balance accuracy and token cost.

Usage:

python regression_runner.py --dataset ml-1m --model Qwen3-4B --lambda_cost 0.1

Output: Policy curves, gate configurations, SHAP analyses, and CSV summaries in ./outputs/{dataset}/{model}/results.

# Typical Workflow
## 1. Generate LLM outputs
python unified_runner.py --dataset ml-1m --model Qwen3-4B --enable_gen

## 2. Extract embeddings and compute features
python embed_feature_extractor.py --dataset ml-1m --model Qwen3-4B

## 3. Train regression router and evaluate policies
python regression_runner.py --dataset ml-1m --model Qwen3-4B

# Dependencies
● Python ≥ 3.9

● PyTorch ≥ 2.0

● Transformers ≥ 4.40

● LightGBM, scikit-learn, pandas, numpy

# License
MIT License — freely available for research and educational use.
