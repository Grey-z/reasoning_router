# unified_runner.py
import os
import sys
import json
import argparse
import pickle
from datetime import datetime
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ---- External modules you provide ----
# - template.py : prompt builders (kept from your code, English only)
# - pipeline_utils.py : embedding extraction + feature computation utils
from template import (
    direct_system, CoT_system, agent_prefix, direct_prefix, CoT_prefix,
    get_agent_system, generate_llm_inputs, generate_llm_inputs_rank
)

# Optional: checklist probe (keep import name to match your existing file)
try:
    from check_list_probe import (
        one_shot_dual_checklist_with_blockmask,
        ProbeConfig,
        filter_margin_raw_features
    )
    HAS_CHECKLIST = True
except Exception:
    HAS_CHECKLIST = False

# Optional: embedding extraction + feature computation from your utils
try:
    from pipeline_utils import (
        # Embedding extraction
        extract_item_embeddings,       # recommendation path (history/candidates) - expected in your utils
        extract_item_embeddings_ir,    # IR path (query + candidate passages)
        # Feature computation
        compute_rec_features_from_jsonl,  # optional convenience functions (batch to CSV)
        compute_ir_features_from_jsonl
    )
    HAS_PIPE_UTILS = True
except Exception:
    HAS_PIPE_UTILS = False


# =========================
# CLI
# =========================
def parse_args():
    p = argparse.ArgumentParser(description="Unified pipeline: build prompts -> HF generation -> embed extraction -> checklist -> feature stats.")
    # Core
    p.add_argument("--dataset_name", type=str, default="ml-1m", choices=["ml-1m", "amazon-game", "rank_ir"])
    p.add_argument("--split", type=str, default="train", choices=["train", "test", "debug"])
    p.add_argument("--system", type=str, default="agent", choices=["direct", "CoT", "agent"])
    p.add_argument("--model_name_or_path", type=str, required=True, help="HF model path or hub id")
    p.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--seed", type=int, default=2025)

    # IO roots (anonymized)
    p.add_argument("--data_root", type=str, default="./data")
    p.add_argument("--out_root", type=str, default="./outputs")

    # Feature toggles
    p.add_argument("--do_generate", type=int, default=1, help="Run HF generation to produce outputs/meta")
    p.add_argument("--do_extract", type=int, default=1, help="Extract embeddings from prompts")
    p.add_argument("--do_checklist", type=int, default=0, help="Run checklist probing and save features")
    p.add_argument("--do_features", type=int, default=0, help="Compute embedding statistical features to CSV")

    # Generation params
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--max_new_tokens", type=int, default=1024)
    p.add_argument("--num_return_sequences", type=int, default=1)

    # Extraction params
    p.add_argument("--pooling", type=str, default="mean", choices=["mean", "start", "end", "both"])
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--max_length", type=int, default=4096)
    p.add_argument("--print_every", type=int, default=50)

    # Logging
    p.add_argument("--log", type=int, default=1)
    return p.parse_args()


# =========================
# Utils
# =========================
def set_seed(seed: int):
    np.random.seed(seed)
    try:
        import random
        random.seed(seed)
    except Exception:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dirs(out_root: str, dataset: str, model_name: str) -> Dict[str, str]:
    """
    Prepare directory structure:
      out_root/{dataset}/{safe_model}/saved_data
      out_root/{dataset}/{safe_model}/saved_result
    """
    safe_model = model_name.replace("/", "_")
    base = os.path.join(out_root, dataset, safe_model)
    saved_data = os.path.join(base, "saved_data")
    saved_result = os.path.join(base, "saved_result")
    os.makedirs(saved_data, exist_ok=True)
    os.makedirs(saved_result, exist_ok=True)
    return {"base": base, "saved_data": saved_data, "saved_result": saved_result}


def load_dataset(dataset: str, split: str, data_root: str):
    """
    Load raw data per dataset:
      - ml-1m: item_df from movies.dat; pickle train/test
      - amazon-game: item_df from item_info.csv; pickle train/test
      - rank_ir: train/test JSON
    """
    if dataset == "ml-1m":
        item_path = os.path.join(data_root, "ml-1m", "movies.dat")
        item_df = pd.read_csv(
            item_path, sep="::", engine="python", encoding="ISO-8859-1",
            names=["item_id", "title", "genres"]
        )
        tr_path = os.path.join(data_root, "ml-1m", "processed_data", "ml-1m_train.txt")
        te_path = os.path.join(data_root, "ml-1m", "processed_data", "ml-1m_test.txt")
        with open(tr_path, "rb") as f:
            train_data = list(pickle.load(f))
        with open(te_path, "rb") as f:
            test_data = list(pickle.load(f))
        if split == "train":
            return item_df, train_data
        if split == "test":
            return item_df, test_data
        return item_df, train_data[:10]

    if dataset == "amazon-game":
        item_path = os.path.join(data_root, "amazon-game", "processed_data", "item_info.csv")
        item_df = pd.read_csv(item_path)
        tr_path = os.path.join(data_root, "amazon-game", "processed_data", "amazon-game_train.txt")
        te_path = os.path.join(data_root, "amazon-game", "processed_data", "amazon-game_test.txt")
        with open(tr_path, "rb") as f:
            train_data = list(pickle.load(f))
        with open(te_path, "rb") as f:
            test_data = list(pickle.load(f))
        if split == "train":
            return item_df, train_data
        if split == "test":
            return item_df, test_data
        return item_df, train_data[:10]

    if dataset == "rank_ir":
        tr_path = os.path.join(data_root, "rank_ir", "processed_data", "train_data.json")
        te_path = os.path.join(data_root, "rank_ir", "processed_data", "test_data.json")
        with open(tr_path, "r", encoding="utf-8") as f:
            train_data = json.load(f)
        with open(te_path, "r", encoding="utf-8") as f:
            test_data = json.load(f)
        if split == "train":
            return None, train_data
        if split == "test":
            return None, test_data
        return None, train_data[:10]

    raise ValueError(f"Unsupported dataset: {dataset}")


def select_system_prompt(dataset: str, system: str) -> str:
    """
    Select a system prompt:
      - For rec datasets and explicit 'direct'/'CoT': use the exact system you provided
      - Otherwise: use agent-style default by dataset
    """
    name = (dataset or "").lower()
    if name in {"ml-1m", "amazon-game"} and system in {"direct", "CoT"}:
        return direct_system if system == "direct" else CoT_system
    return get_agent_system(dataset)


def select_assistant_prefix(system: str) -> str:
    if system == "direct":
        return direct_prefix
    if system == "CoT":
        return CoT_prefix
    return agent_prefix


def hf_generate_for_chats(
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    chat_inputs: List[List[Dict[str, str]]],
    temperature: float = 0.8,
    top_p: float = 0.9,
    max_new_tokens: int = 1024,
    num_return_sequences: int = 1,
) -> List[Dict[str, Any]]:
    """
    Minimal chat generation using HF without vLLM.
    Returns a list of dicts: {"request_id": i, "prompt": rendered_chat_text, "outputs":[{"text":..., "num_tokens":...}, ...]}
    """
    results: List[Dict[str, Any]] = []
    eos_id = tokenizer.eos_token_id

    for i, messages in enumerate(chat_inputs):
        # render chat to string with chat template
        rendered = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        # tokenize
        enc = tokenizer(
            rendered, return_tensors="pt", add_special_tokens=False
        ).to(model.device)

        # generate
        gen_out = model.generate(
            **enc,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            num_return_sequences=num_return_sequences,
            eos_token_id=eos_id,
            pad_token_id=eos_id,
        )

        # decode one or multiple sequences
        # only decode the newly generated tokens (slice)
        outputs = []
        input_len = enc["input_ids"].shape[1]
        for b in range(gen_out.shape[0]):
            gen_seq = gen_out[b, input_len:]
            text = tokenizer.decode(gen_seq, skip_special_tokens=True)
            outputs.append({"text": text, "num_tokens": int(gen_seq.numel())})

        results.append({"request_id": f"req_{i}", "prompt": rendered, "outputs": outputs})

    return results


def write_json(fp: str, obj: Any):
    with open(fp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def read_meta_prompts(saved_dir: str, prefix: str) -> List[str]:
    meta_path = os.path.join(saved_dir, f"{prefix}_meta.json")
    if not os.path.exists(meta_path):
        return []
    meta = json.load(open(meta_path, "r", encoding="utf-8"))
    return [m.get("prompt", "") for m in meta]


# =========================
# Main
# =========================
def main():
    args = parse_args()
    set_seed(args.seed)

    # Prepare dirs
    d = ensure_dirs(args.out_root, args.dataset_name, args.model_name_or_path)
    saved_data = d["saved_data"]
    saved_result = d["saved_result"]

    # Logging to file
    if args.log:
        log_dir = os.path.join(d["base"], "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(
            log_dir,
            f"{args.dataset_name}_{args.system}_{args.split}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
        sys.stdout = open(log_file, "w")
        sys.stderr = sys.stdout
        print(f"[Log] to {log_file}")

    print(f"[Config] dataset={args.dataset_name} split={args.split} system={args.system}")
    print(f"[Paths] data_root={args.data_root} out_root={args.out_root}")
    print(f"[Model] {args.model_name_or_path} device={args.device}")

    # Load data
    item_df, data_list = load_dataset(args.dataset_name, args.split, args.data_root)

    # Build chat inputs (end-to-end prompt creation)
    system_text = select_system_prompt(args.dataset_name, args.system)
    assistant_prefix = select_assistant_prefix(args.system)

    if args.dataset_name in {"ml-1m", "amazon-game"}:
        chat_inputs = generate_llm_inputs(
            data_list, item_df, system_text, assistant_prefix, args.dataset_name
        )
    elif args.dataset_name in {"rank_ir"}:
        chat_inputs = generate_llm_inputs_rank(
            data_list, args.system, assistant_prefix, args.dataset_name
        )
    else:
        raise ValueError("Unsupported dataset")

    # Save constructed chat inputs for reproducibility
    prefix = f"{args.system}_{args.dataset_name}_{args.split}"
    chat_inputs_path = os.path.join(saved_data, f"{prefix}_llm_input.jsonl")
    write_json(chat_inputs_path, chat_inputs)
    print(f"[Saved] chat inputs -> {chat_inputs_path}")

    # ===== Step 1: HF generation (optional) =====
    if args.do_generate:
        print("[Gen] loading HF model/tokenizer ...")
        tok = AutoTokenizer.from_pretrained(
            args.model_name_or_path, trust_remote_code=True, use_fast=True
        )
        if not getattr(tok, "is_fast", False):
            raise ValueError("A fast tokenizer is required (use_fast=True).")
        mdl = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path, trust_remote_code=True,
            torch_dtype=torch.float16 if args.device == "cuda" else torch.float32
        ).to(args.device)
        mdl.eval()

        print("[Gen] generating ...")
        gen_results = hf_generate_for_chats(
            tokenizer=tok,
            model=mdl,
            chat_inputs=chat_inputs,
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
            num_return_sequences=args.num_return_sequences,
        )
        out_file = os.path.join(saved_data, f"{prefix}_llm_output.jsonl")
        write_json(out_file, gen_results)
        print(f"[Saved] generation outputs -> {out_file}")

        # Build meta.json (request_id + prompt)
        meta = [{"request_id": r["request_id"], "prompt": r["prompt"]} for r in gen_results]
        meta_path = os.path.join(saved_data, f"{prefix}_meta.json")
        write_json(meta_path, meta)
        print(f"[Saved] meta.json -> {meta_path}")

        # Release memory
        del mdl
        torch.cuda.empty_cache()
    else:
        print("[Gen] skipped")

    # ===== Step 2: Checklist probing (optional) =====
    if args.do_checklist:
        if not HAS_CHECKLIST:
            print("[Checklist] skipped (check_list_probe not available).")
        else:
            print("[Checklist] running probe ...")
            # checklist uses the chat_inputs list and the backbone model for logits
            cfg = ProbeConfig(dtype=torch.float16, use_blockmask=True)
            features_list, chat_filled = one_shot_dual_checklist_with_blockmask(
                chat_input_list=chat_inputs,
                model_name_or_path=args.model_name_or_path,
                dataset_name=args.dataset_name,
                cfg=cfg
            )
            full_df = pd.DataFrame(features_list)
            full_df_path = os.path.join(saved_result, f"{prefix}_prompt_feature_origin.csv")
            full_df.to_csv(full_df_path, index=False)
            print(f"[Checklist] saved raw features -> {full_df_path}")

            sel_df = filter_margin_raw_features(
                features_list,
                keep_aggregates=("margin_mean_raw", "win_count", "top3_margin_mean", "gap_mean"),
                topk=3,
                rename_prefix="mraw"
            )
            sel_df_path = os.path.join(saved_result, f"{prefix}_prompt_feature.csv")
            sel_df.to_csv(sel_df_path, index=False)
            print(f"[Checklist] saved selected features -> {sel_df_path}")
    else:
        print("[Checklist] skipped")

    # ===== Step 3: Embedding extraction (optional) =====
    if args.do_extract:
        if not HAS_PIPE_UTILS:
            print("[Extract] skipped (pipeline_utils not available).")
        else:
            print("[Extract] loading HF model/tokenizer for hidden states ...")
            tok = AutoTokenizer.from_pretrained(
                args.model_name_or_path, trust_remote_code=True, use_fast=True
            )
            if not getattr(tok, "is_fast", False):
                raise ValueError("A fast tokenizer is required (use_fast=True).")
            mdl = AutoModelForCausalLM.from_pretrained(
                args.model_name_or_path, trust_remote_code=True,
                torch_dtype=torch.float16 if args.device == "cuda" else torch.float32,
                device_map="auto" if args.device == "cuda" else None
            )
            mdl.eval()

            # prefer prompts from meta (identical to those used for generation)
            prompts = read_meta_prompts(saved_data, prefix)
            if not prompts:
                # fallback to user messages from chat_inputs
                prompts = [next(m["content"] for m in chat if m["role"] == "user") for chat in chat_inputs]

            print(f"[Extract] prompts: {len(prompts)}")
            if args.dataset_name == "rank_ir":
                entries = extract_item_embeddings_ir(
                    tokenizer=tok,
                    model=mdl.to(args.device),
                    prompts=prompts,
                    pooling=args.pooling,
                    max_length=args.max_length,
                    batch_size=args.batch_size,
                    print_every=args.print_every,
                )
                entries_path = os.path.join(saved_data, f"{prefix}_ir_entry_lists.jsonl")
            else:
                # recommendation path
                entries = extract_item_embeddings(
                    tokenizer=tok,
                    model=mdl.to(args.device),
                    prompts=prompts,
                    dataset_name=args.dataset_name,
                    pooling=args.pooling,
                    max_length=args.max_length,
                    batch_size=args.batch_size,
                    print_every=args.print_every,
                )
                entries_path = os.path.join(saved_data, f"{prefix}_entry_lists.jsonl")

            write_json(entries_path, entries)
            print(f"[Extract] saved entries -> {entries_path}")

            del mdl
            torch.cuda.empty_cache()
    else:
        print("[Extract] skipped")

    # ===== Step 4: Embedding feature computation (optional) =====
    if args.do_features:
        if not HAS_PIPE_UTILS:
            print("[Features] skipped (pipeline_utils not available).")
        else:
            if args.dataset_name == "rank_ir":
                in_jsonl = os.path.join(saved_data, f"{prefix}_ir_entry_lists.jsonl")
                if os.path.exists(in_jsonl):
                    out_csv = os.path.join(saved_result, f"{prefix}_embed_features.csv")
                    try:
                        compute_ir_features_from_jsonl(in_jsonl, out_csv, batch_size=256)
                        print(f"[Features] saved -> {out_csv}")
                    except Exception as e:
                        print(f"[Features] IR computation failed: {e}")
                else:
                    print(f"[Features] missing entries jsonl: {in_jsonl}")
            else:
                in_jsonl = os.path.join(saved_data, f"{prefix}_entry_lists.jsonl")
                if os.path.exists(in_jsonl):
                    out_csv = os.path.join(saved_result, f"{prefix}_embed_features.csv")
                    try:
                        compute_rec_features_from_jsonl(in_jsonl, out_csv, batch_size=256)
                        print(f"[Features] saved -> {out_csv}")
                    except Exception as e:
                        print(f"[Features] REC computation failed: {e}")
                else:
                    print(f"[Features] missing entries jsonl: {in_jsonl}")
    else:
        print("[Features] skipped")

    # Close log
    if args.log:
        sys.stdout.close()
        sys.stderr = sys.stdout = sys.__stdout__
        print(f"[Done] Log saved.")


if __name__ == "__main__":
    main()
