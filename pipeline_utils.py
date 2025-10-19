#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import random
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import re

# -----------------------------
# basic utils
# -----------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dirs(*paths: str) -> None:
    for p in paths:
        os.makedirs(p, exist_ok=True)


def open_log(log_dir: str, dataset: str, model: str, system: str, split: str) -> Tuple[Any, str]:
    ensure_dirs(log_dir)
    log_file = os.path.join(
        log_dir,
        f"{dataset}_{os.path.basename(model)}_{system}_{split}_{pd.Timestamp.now():%Y%m%d_%H%M%S}.log",
    )
    f = open(log_file, "w")
    sys.stdout = f
    sys.stderr = f
    return f, log_file


# -----------------------------
# io helpers
# -----------------------------
def load_chat_inputs(chat_inputs_json: str) -> List[List[Dict[str, str]]]:
    if not os.path.exists(chat_inputs_json):
        raise FileNotFoundError(f"chat_inputs_json not found: {chat_inputs_json}")
    with open(chat_inputs_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list) or (len(data) > 0 and not isinstance(data[0], list)):
        raise ValueError("chat_inputs_json must be List[List[{'role','content'}]].")
    return data


def save_json(obj: Any, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# -----------------------------
# hf model
# -----------------------------
def load_tokenizer_model(
    path: str, dtype: torch.dtype = torch.float16
) -> Tuple[AutoTokenizer, AutoModelForCausalLM, str]:
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=True)
    if not getattr(tok, "is_fast", False):
        raise ValueError("Fast tokenizer required for offset_mapping.")
    mdl = AutoModelForCausalLM.from_pretrained(
        path, trust_remote_code=True, torch_dtype=dtype, device_map="auto"
    )
    mdl.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return tok, mdl, device


# -----------------------------
# generation (transformers only)
# -----------------------------
def hf_chat_generate(
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    chat_inputs: List[List[Dict[str, str]]],
    n: int = 5,
    max_new_tokens: int = 1024,
    temperature: float = 0.8,
    top_p: float = 0.9,
) -> List[Dict[str, Any]]:
    results = []
    for i, msgs in enumerate(chat_inputs):
        prompt = tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        gen_out = model.generate(
            **inputs,
            do_sample=True,
            top_p=top_p,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            num_return_sequences=n,
            pad_token_id=tokenizer.eos_token_id,
        )
        texts = tokenizer.batch_decode(gen_out, skip_special_tokens=True)
        outs = [{"text": t} for t in texts]
        results.append({"request_id": f"req_{i}", "prompt": prompt, "outputs": outs})
        if (i % 10) == 0:
            print(f"[Gen] {i+1}/{len(chat_inputs)}")
    return results


# -----------------------------
# embedding extraction (Reco)
# -----------------------------
class DatasetSpec:
    def __init__(self, name: str, line_re: "re.Pattern", id_group: int, content_group_for_pooling: Optional[int] = None):
        self.name = name
        self.line_re = line_re
        self.id_group = id_group
        self.content_group_for_pooling = content_group_for_pooling


def _make_specs() -> Dict[str, DatasetSpec]:
    import re
    mark = re.escape("§")
    specs: Dict[str, DatasetSpec] = {}
    ml1m_re = re.compile(rf"(?m)^{mark}\s*ID\s*:\s*(\d+)\s+Title\s*:\s*(.+?)(?:\s*;\s*Genres\s*:\s*.*)?\s*$")
    specs["ml-1m"] = DatasetSpec("ml-1m", ml1m_re, id_group=1, content_group_for_pooling=None)
    ag_re = re.compile(rf"(?m)^{mark}\s*ID\s*:\s*(\d+)\s+Title\s*:\s*(.+?)\s*$")
    specs["amazon-game"] = DatasetSpec("amazon-game", ag_re, id_group=1, content_group_for_pooling=None)
    generic_re = re.compile(rf"(?m)^{mark}\s*ID\s*:\s*(\d+)\s*$")
    specs["generic-id-only"] = DatasetSpec("generic-id-only", generic_re, id_group=1, content_group_for_pooling=None)
    return specs


DATASET_SPECS = _make_specs()


def get_spec(dataset: str) -> DatasetSpec:
    key = (dataset or "").lower()
    if key in DATASET_SPECS:
        return DATASET_SPECS[key]
    if "amazon-game" in key:
        return DATASET_SPECS["amazon-game"]
    if "ml-1m" in key or "ml1m" in key:
        return DATASET_SPECS["ml-1m"]
    return DATASET_SPECS["generic-id-only"]


def find_sections(prompt: str) -> Dict[str, Optional[Tuple[int, int]]]:
    import re
    his = re.compile(r"(?mi)^[ \t]*User's[ \t]+watched[ \t]+history:\s*$")
    cand = re.compile(r"(?mi)^[ \t]*Candidate[ \t]+items:\s*$")
    L = len(prompt)
    m_h, m_c = his.search(prompt), cand.search(prompt)

    def start(m): return m.end()

    if m_h and m_c:
        return {"history": (start(m_h), m_c.start()), "candidate": (start(m_c), L)}
    if m_c:
        return {"history": (0, m_c.start()), "candidate": (start(m_c), L)}
    if m_h:
        return {"history": (start(m_h), L), "candidate": None}
    return {"history": None, "candidate": None}


def parse_item_blocks(prompt: str, spec: DatasetSpec) -> List[Dict[str, Any]]:
    blocks = []
    for m in spec.line_re.finditer(prompt):
        try:
            item_id = int(m.group(spec.id_group))
        except Exception:
            continue
        row_l, row_r = m.start(), m.end()
        if spec.content_group_for_pooling is not None:
            try:
                pool_l, pool_r = m.span(spec.content_group_for_pooling)
            except Exception:
                pool_l, pool_r = row_l, row_r
        else:
            pool_l, pool_r = row_l, row_r
        blocks.append({
            "item_id": item_id,
            "char_start": row_l,
            "char_end": row_r,
            "pool_char_start": pool_l,
            "pool_char_end": pool_r,
        })
    return blocks


def char_to_token_span(offsets: List[Tuple[int, int]], char_start: int, char_end: int) -> Optional[Tuple[int, int]]:
    ts = te = None
    for i, (s, e) in enumerate(offsets):
        if s == 0 and e == 0:
            continue
        if not (e <= char_start or s >= char_end):
            if ts is None:
                ts = i
            te = i
    return None if ts is None or te is None else (int(ts), int(te))


def align_blocks(
    blocks: List[Dict[str, Any]],
    sections: Dict[str, Optional[Tuple[int, int]]],
    offsets: List[Tuple[int, int]],
) -> List[Dict[str, Any]]:
    def in_rng(pos: int, rg: Optional[Tuple[int, int]]) -> bool:
        return rg is not None and rg[0] <= pos < rg[1]
    out = []
    for b in blocks:
        sp = char_to_token_span(offsets, b["pool_char_start"], b["pool_char_end"])
        if sp is None:
            continue
        sec = None
        if in_rng(b["char_start"], sections.get("history")):
            sec = "history"
        elif in_rng(b["char_start"], sections.get("candidate")):
            sec = "candidate"
        out.append({"item_id": b["item_id"], "token_start": sp[0], "token_end": sp[1], "section": sec})
    return out


def pool_hidden(hidden_last: np.ndarray, s: int, e: int, strategy: str = "mean") -> np.ndarray:
    if strategy == "start":
        return hidden_last[s]
    if strategy == "end":
        return hidden_last[e]
    if strategy == "both":
        return np.concatenate([hidden_last[s], hidden_last[e]], axis=-1)
    l = max(0, s)
    r = max(l, e)
    return hidden_last[l:r + 1].mean(axis=0)


def extract_item_embeddings(
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    chat_inputs: List[List[Dict[str, str]]],
    dataset_name: str,
    pooling: str,
    emb_dtype: str,
    max_length: int,
    batch_size: int,
    print_every: int,
    save_dir: str,
    save_prefix: str,
    save_format: str,
    shard_size: int,
) -> str:
    spec = get_spec(dataset_name)
    ensure_dirs(save_dir)

    jsonl_f = None
    jsonl_path = None
    entries_buf: List[dict] = []
    shard_idx = 0

    def flush_npz():
        nonlocal entries_buf, shard_idx
        if not entries_buf:
            return
        npz_path = os.path.join(save_dir, f"{save_prefix}_entry_lists_shard{shard_idx:03d}.npz")
        np.savez_compressed(
            npz_path,
            entries=np.array(entries_buf, dtype=object),
            pooling=np.asarray([pooling], dtype=object),
            emb_dtype=np.asarray([emb_dtype], dtype=object),
            dataset_spec=np.asarray([spec.name], dtype=object),
        )
        print(f"[Shard] Saved {len(entries_buf)} -> {npz_path}")
        shard_idx += 1
        entries_buf.clear()

    if save_format == "jsonl":
        jsonl_path = os.path.join(save_dir, f"{save_prefix}_entry_lists.jsonl")
        jsonl_f = open(jsonl_path, "w", encoding="utf-8")

    with torch.no_grad():
        for start in range(0, len(chat_inputs), batch_size):
            end = min(start + batch_size, len(chat_inputs))
            batch = chat_inputs[start:end]
            prompts = [
                tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False) for m in batch
            ]
            tok = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
                return_offsets_mapping=True,
            )
            offsets_b = tok.pop("offset_mapping")
            tok = {k: v.to(model.device) for k, v in tok.items()}

            out = model(**tok, output_hidden_states=True, return_dict=True)
            last_hidden_b = out.hidden_states[-1]
            input_ids_b = tok["input_ids"]

            for bi in range(last_hidden_b.size(0)):
                i = start + bi
                prompt = prompts[bi]
                offsets = [tuple(map(int, p)) for p in offsets_b[bi].tolist()]
                hidden_last = last_hidden_b[bi].detach().cpu().numpy()

                sections = find_sections(prompt)
                blocks = parse_item_blocks(prompt, spec)
                aligned = align_blocks(blocks, sections, offsets)

                history_list, candidates_list = [], []
                for a in aligned:
                    emb = pool_hidden(hidden_last, a["token_start"], a["token_end"], strategy=pooling)
                    if save_format == "npz":
                        emb_to_save = emb.astype(emb_dtype)
                    else:
                        emb_to_save = emb.astype("float32").tolist()
                    item_obj = {"item_id": a["item_id"], "embedding": emb_to_save}
                    if a["section"] == "history":
                        history_list.append(item_obj)
                    elif a["section"] == "candidate":
                        candidates_list.append(item_obj)
                    else:
                        candidates_list.append(item_obj)

                entry = {
                    "entry_id": i,
                    "request_id": f"req_{i}",
                    "history": history_list,
                    "candidates": candidates_list,
                }

                if save_format == "npz":
                    entries_buf.append(entry)
                    if (i + 1) % shard_size == 0:
                        flush_npz()
                else:
                    assert jsonl_f is not None
                    jsonl_f.write(json.dumps(entry, ensure_ascii=False) + "\n")

                if (i % print_every) == 0:
                    print(f"[Embed] {i+1}/{len(chat_inputs)}  h={len(history_list)} c={len(candidates_list)}")

            del out, last_hidden_b, input_ids_b
            torch.cuda.empty_cache()

    if save_format == "npz":
        flush_npz()
        return os.path.join(save_dir, f"{save_prefix}_entry_lists_shard*.npz")
    else:
        assert jsonl_f is not None
        jsonl_f.close()
        print(f"[Done] Saved JSONL -> {jsonl_path}")
        return jsonl_path


# -----------------------------
# checklist probe (Direct vs CoT)
# -----------------------------
_MOVIE_DUAL_GROUPS = [
    {"dir": "Are candidates highly homogeneous in genre/year (e.g., median genre-overlap ≥ 0.8 OR release-year IQR ≤ 3)?",
     "cot": "Are candidates thematically diverse across genres/eras/directors (e.g., median genre-overlap ≤ 0.4 OR release-year IQR ≥ 10)?"},
    {"dir": "Does the watch history show a dominant genre cluster (≥70% of last 20 movies share the same top-2 genres)?",
     "cot": "Does the watch history split across ≥3 unrelated genre clusters with none >40% (multi-intent)?"},
    {"dir": "Do candidate titles/genres lexically match the history keywords strongly (e.g., ≥70% top keywords overlap)?",
     "cot": "Is lexical overlap weak (≤20%) so semantic bridging beyond keywords is needed?"},
    {"dir": "Are there effectively no hard constraints implied?",
     "cot": "Must strict constraints be inferred and enforced?"},
    {"dir": "Are near-duplicates negligible?",
     "cot": "Are many near-duplicate or tightly related entries where subtle tie-breaking is required?"},
    {"dir": "Is franchise order irrelevant to ranking?",
     "cot": "Does correct watch order matter?"},
    {"dir": "Are most candidates mainstream?",
     "cot": "Are many candidates long-tail where popularity heuristics are unreliable?"},
    {"dir": "Does the user show a stable recent-era preference?",
     "cot": "Is there explicit tension requiring trade-off reasoning to balance eras?"},
    {"dir": "Can coarse genre signals alone likely determine ranking?",
     "cot": "Do fine-grained cues need to be articulated to separate very similar candidates?"},
    {"dir": "Is metadata clean?",
     "cot": "Is metadata noisy requiring explicit reasoning?"},
]
_GAME_DUAL_GROUPS = [
    {"dir": "Are platform and hardware constraints obvious and satisfied?",
     "cot": "Do platform/VR/GPU/OS constraints need to be inferred or disambiguated?"},
    {"dir": "Is the solo vs co-op preference clear?",
     "cot": "Is solo vs co-op preference ambiguous/contradictory and critical?"},
    {"dir": "Do session-length patterns match candidates directly?",
     "cot": "Do loop depth/grind length/endgame needs require explicit reasoning?"},
    {"dir": "Are monetization/DLC issues irrelevant?",
     "cot": "Are DLC bundles/editions/paywalls decisive?"},
    {"dir": "Is franchise continuity irrelevant?",
     "cot": "Does correct entry-point in a franchise matter?"},
    {"dir": "Is online population/latency not a deciding factor?",
     "cot": "Is community health/region/latency critical?"},
    {"dir": "Are age rating/toxicity concerns irrelevant?",
     "cot": "Must age-rating/toxicity constraints be enforced?"},
    {"dir": "Do mainstream titles dominate?",
     "cot": "Are many indie/long-tail titles where naive priors fail?"},
    {"dir": "Are control/accessibility requirements non-critical?",
     "cot": "Do control/accessibility constraints drive ranking?"},
    {"dir": "Can coarse genre tags alone decide ranking?",
     "cot": "Do sub-genre/mechanics nuances require explicit reasoning?"},
]
_IR_DUAL_GROUPS = [
    {"dir": "Is the query single-intent and unambiguous?",
     "cot": "Is the query ambiguous or multi-intent?"},
    {"dir": "Do top candidates have strong lexical overlap with the query?",
     "cot": "Is there a lexical–semantic mismatch requiring paraphrase/bridging?"},
    {"dir": "Can a single passage fully answer the query?",
     "cot": "Is the answer scattered and requires aggregation?"},
    {"dir": "Are near-duplicates rare?",
     "cot": "Are many near-duplicate/tie passages requiring subtle weighing?"},
    {"dir": "Is temporal reasoning irrelevant?",
     "cot": "Is temporal reasoning essential?"},
    {"dir": "Does the query avoid negation/exception/scope?",
     "cot": "Does ranking depend on handling negation/exception/scope precisely?"},
    {"dir": "Do domain terms/acronyms map unambiguously?",
     "cot": "Do they require disambiguation?"},
    {"dir": "Is jurisdiction/entity-type constraint irrelevant?",
     "cot": "Must such constraints be inferred and enforced?"},
    {"dir": "Are off-topic/adversarial passages minimal?",
     "cot": "Is there substantial noise to down-weight?"},
    {"dir": "Does the query avoid numeric thresholds/comparatives?",
     "cot": "Is numeric reasoning required?"},
]


def get_dual_checklist(dataset_name: str, flatten: bool = False) -> List:
    key = (dataset_name or "").strip().lower()
    if key in {"ml-1m", "movielens", "movie"}:
        groups = _MOVIE_DUAL_GROUPS
    elif key in {"amazon-game", "steam", "game"}:
        groups = _GAME_DUAL_GROUPS
    elif key in {"rank_ir", "retrieval", "ir", "search"}:
        groups = _IR_DUAL_GROUPS
    else:
        groups = _IR_DUAL_GROUPS
    if not flatten:
        return groups
    flat = []
    for g in groups:
        flat.append(("dir", g["dir"]))
        flat.append(("cot", g["cot"]))
    return flat


def _first_subtoken_ids(tokenizer: AutoTokenizer, variants: Tuple[str, ...]) -> List[int]:
    ids = []
    for v in variants:
        tok = tokenizer.encode(v, add_special_tokens=False)
        if len(tok) >= 1:
            ids.append(tok[0])
    seen, uniq = set(), []
    for t in ids:
        if t not in seen:
            uniq.append(t)
            seen.add(t)
    return uniq


def _encode_chat(
    tokenizer: AutoTokenizer, system_text: str, user_text: str, assistant_text: Optional[str] = None
) -> Tuple[List[int], str]:
    msgs = [{"role": "system", "content": system_text}, {"role": "user", "content": user_text}]
    if assistant_text is not None:
        msgs.append({"role": "assistant", "content": assistant_text})
    txt = tokenizer.apply_chat_template(msgs, add_generation_prompt=(assistant_text is None), tokenize=False)
    ids = tokenizer.encode(txt, add_special_tokens=False)
    return ids, txt


def _margin_from_hidden_subset(
    h_last: torch.Tensor, lm_head, yes_ids: List[int], no_ids: List[int]
) -> float:
    h = h_last.to(torch.float32)
    W = lm_head.weight
    b = lm_head.bias
    y_idx = torch.tensor(yes_ids, dtype=torch.long, device=W.device)
    n_idx = torch.tensor(no_ids, dtype=torch.long, device=W.device)
    W_y = W.index_select(0, y_idx).to(torch.float32)
    W_n = W.index_select(0, n_idx).to(torch.float32)
    by = (b.index_select(0, y_idx).to(torch.float32) if b is not None else None)
    bn = (b.index_select(0, n_idx).to(torch.float32) if b is not None else None)
    logits_y = (W_y @ h) + (by if by is not None else 0.0)
    logits_n = (W_n @ h) + (bn if bn is not None else 0.0)
    mY = torch.logsumexp(logits_y, dim=0)
    mN = torch.logsumexp(logits_n, dim=0)
    return float((mY - mN).item())


def _prob_from_margin(m: float) -> float:
    return float(torch.sigmoid(torch.tensor(m)).item())


@torch.inference_mode()
def checklist_probe(
    chat_input_list: List[List[Dict[str, Any]]],
    model_name_or_path: str,
    dataset_name: str,
    dtype: torch.dtype = torch.float16,
) -> Tuple[List[Dict[str, float]], List[List[Dict[str, Any]]]]:
    tok = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True, use_fast=True)
    mdl = AutoModelForCausalLM.from_pretrained(
        model_name_or_path, trust_remote_code=True, torch_dtype=dtype, device_map="auto"
    )
    mdl.eval()

    # verbalizers
    verbalizer_sets = {
        "yn": ((" Yes", "Yes", " YES", " yes"), (" No", "No", " NO", " no")),
        "tf": ((" True", "True"), (" False", "False")),
        "ab": ((" A", "A"), (" B", "B")),
        "01": ((" 1", "1"), (" 0", "0")),
    }
    vz_map: Dict[str, Tuple[List[int], List[int]]] = {}
    for k, (Y, N) in verbalizer_sets.items():
        yids = _first_subtoken_ids(tok, Y)
        nids = _first_subtoken_ids(tok, N)
        if yids and nids:
            vz_map[k] = (yids, nids)
    if not vz_map:
        raise RuntimeError("no valid verbalizer set")

    q_pairs = get_dual_checklist(dataset_name)
    features_list: List[Dict[str, float]] = []
    chat_filled: List[List[Dict[str, Any]]] = []

    for chat in chat_input_list:
        sys_msg = next((m["content"] for m in chat if m["role"] == "system"), None)
        usr_msg = next((m["content"] for m in chat if m["role"] == "user"), None)
        if sys_msg is None or usr_msg is None:
            raise ValueError("Each chat must include both system and user")

        lines = ["[Checklist]", "Please answer each question with exactly one token: 'Yes' or 'No'. If uncertain, answer 'No'.", ""]
        for i, (qd, qc) in enumerate(q_pairs, 1):
            lines.append(f"Q{i} (Direct-friendly): {qd}  Answer:")
            lines.append(f"Q{i} (CoT-friendly): {qc}  Answer:")
        lines.append("Prefix:")
        block_text = "\n".join(lines)

        filled = []
        injected = False
        for m in chat:
            if m.get("role") == "assistant":
                filled.append({"role": "assistant", "content": block_text})
                injected = True
            else:
                filled.append(dict(m))
        if not injected:
            filled.append({"role": "assistant", "content": block_text})
        chat_filled.append(filled)

        # prefix KV
        pref_ids, _ = _encode_chat(tok, sys_msg, usr_msg, assistant_text=None)
        pref_t = torch.tensor([pref_ids], dtype=torch.long, device=mdl.device)
        pref_out = mdl(input_ids=pref_t, use_cache=True, return_dict=True)
        past = pref_out.past_key_values
        lm_head = mdl.get_output_embeddings()

        # neutral bias
        neu_full, _ = _encode_chat(tok, sys_msg, usr_msg, assistant_text="Answer:")
        neu_delta = neu_full[len(pref_ids):]
        neu_t = torch.tensor([neu_delta], dtype=torch.long, device=mdl.device)
        neu_out = mdl(
            input_ids=neu_t, past_key_values=past, output_hidden_states=True, use_cache=False, return_dict=True
        )
        h_neu = neu_out.hidden_states[-1][0, -1, :]
        bias_by_set = {k: _margin_from_hidden_subset(h_neu, lm_head, y, n) for k, (y, n) in vz_map.items()}

        feat: Dict[str, float] = {}
        margins_raw: List[float] = []

        for qi, (qd, qc) in enumerate(q_pairs, 1):
            d_full, _ = _encode_chat(tok, sys_msg, usr_msg, assistant_text=f"Q{qi} (Direct-friendly): {qd}\nAnswer:")
            c_full, _ = _encode_chat(tok, sys_msg, usr_msg, assistant_text=f"Q{qi} (CoT-friendly): {qc}\nAnswer:")
            d_delta = d_full[len(pref_ids):]
            c_delta = c_full[len(pref_ids):]

            d_t = torch.tensor([d_delta], dtype=torch.long, device=mdl.device)
            c_t = torch.tensor([c_delta], dtype=torch.long, device=mdl.device)
            d_out = mdl(input_ids=d_t, past_key_values=past, output_hidden_states=True, use_cache=False, return_dict=True)
            c_out = mdl(input_ids=c_t, past_key_values=past, output_hidden_states=True, use_cache=False, return_dict=True)
            h_d = d_out.hidden_states[-1][0, -1, :]
            h_c = c_out.hidden_states[-1][0, -1, :]

            probs_dir, probs_cot, raw_m_dir, raw_m_cot = [], [], [], []
            for k, (yids, nids) in vz_map.items():
                m_dir = _margin_from_hidden_subset(h_d, lm_head, yids, nids) - bias_by_set[k]
                m_cot = _margin_from_hidden_subset(h_c, lm_head, yids, nids) - bias_by_set[k]
                probs_dir.append(_prob_from_margin(m_dir))
                probs_cot.append(_prob_from_margin(m_cot))
                raw_m_dir.append(m_dir)
                raw_m_cot.append(m_cot)

            p_dir = float(np.mean(probs_dir))
            p_cot = float(np.mean(probs_cot))
            raw_dir = float(np.mean(raw_m_dir))
            raw_cot = float(np.mean(raw_m_cot))
            margin_raw = raw_dir - raw_cot

            feat[f"p_dir_q{qi}"] = p_dir
            feat[f"p_cot_q{qi}"] = p_cot
            feat[f"margin_q{qi}"] = p_dir - p_cot
            feat[f"margin_raw_q{qi}"] = margin_raw
            margins_raw.append(margin_raw)

        dir_mean = float(np.mean([feat[f"p_dir_q{i}"] for i in range(1, len(q_pairs) + 1)]))
        cot_mean = float(np.mean([feat[f"p_cot_q{i}"] for i in range(1, len(q_pairs) + 1)]))
        feat["dir_mean"] = dir_mean
        feat["cot_mean"] = cot_mean
        feat["gap_mean"] = dir_mean - cot_mean

        margins_raw_arr = np.array(margins_raw, dtype=np.float32)
        feat["margin_mean_raw"] = float(margins_raw_arr.mean())
        feat["win_count"] = float((margins_raw_arr > 0).sum())
        k = min(3, len(margins_raw_arr))
        topk_idx = np.argsort(np.abs(margins_raw_arr))[-k:]
        feat["top3_margin_mean"] = float(margins_raw_arr[topk_idx].mean())

        features_list.append(feat)

    return features_list, chat_filled


def filter_margin_raw_features(
    features_list: List[Dict[str, float]],
    keep_aggregates: Iterable[str] = ("margin_mean_raw", "win_count", "top3_margin_mean", "gap_mean"),
    topk: int = 3,
    rename_prefix: Optional[str] = "mraw",
) -> pd.DataFrame:
    rows = []
    all_q_indices = set()
    parsed = []
    for rec in features_list:
        q2val: Dict[int, float] = {}
        for k, v in rec.items():
            if k.startswith("margin_raw_q"):
                try:
                    q_idx = int(k.split("margin_raw_q")[1])
                except Exception:
                    continue
                q2val[q_idx] = float(v)
                all_q_indices.add(q_idx)
        parsed.append((rec, q2val))

    q_indices_sorted = sorted(all_q_indices)
    for rec, q2val in parsed:
        out: Dict[str, float] = {}
        for q in q_indices_sorted:
            val = q2val.get(q, np.nan)
            if rename_prefix:
                out[f"{rename_prefix}_q{q}"] = val
            else:
                out[f"margin_raw_q{q}"] = val

        margins = np.array([q2val[q] for q in q_indices_sorted if q in q2val], dtype=float)
        for agg in keep_aggregates:
            if agg in rec:
                out[agg] = rec[agg]
                continue
            if agg == "margin_mean_raw":
                out[agg] = float(np.nanmean(margins)) if margins.size else np.nan
            elif agg == "win_count":
                out[agg] = float(np.sum(margins > 0)) if margins.size else 0.0
            elif agg == "top3_margin_mean":
                if margins.size:
                    smallest = np.sort(margins)[:max(1, topk)]
                    out[agg] = float(np.mean(smallest))
                else:
                    out[agg] = np.nan
            elif agg == "gap_mean":
                out[agg] = float(rec.get("dir_mean", np.nan) - rec.get("cot_mean", np.nan))
            else:
                out[agg] = rec.get(agg, np.nan)
        rows.append(out)

    margin_cols = [f"{rename_prefix}_q{q}" for q in q_indices_sorted] if rename_prefix \
        else [f"margin_raw_q{q}" for q in q_indices_sorted]
    agg_cols = list(keep_aggregates)
    df = pd.DataFrame(rows)[margin_cols + agg_cols]
    return df


# -----------------------------
# feature computation from embeddings (Reco + IR)
# -----------------------------
def _norm_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + eps)


def _safe_var(x: np.ndarray) -> float:
    return float(np.var(x)) if x.size else 0.0


def _quantiles(a: np.ndarray, qs: List[float]) -> List[float]:
    if a.size == 0:
        return [0.0] * len(qs)
    return [float(q) for q in np.quantile(a, qs)]


def features_from_entry_reco(entry: Dict[str, Any]) -> Dict[str, float]:
    H = np.array([i["embedding"] for i in entry.get("history", []) if i.get("embedding") is not None], dtype=np.float32)
    C = np.array([i["embedding"] for i in entry.get("candidates", []) if i.get("embedding") is not None], dtype=np.float32)

    res = {
        "entry_id": entry.get("entry_id", ""),
        "hist_len": float(len(H)),
        "hist_embedding_variance": _safe_var(H),
        "cand_embedding_dispersion": _safe_var(C),

        "history_candidate_sim_max": 0.0,
        "history_candidate_sim_mean": 0.0,
        "history_candidate_sim_min": 0.0,
        "history_candidate_sim_var": 0.0,
        "history_candidate_sim_p50": 0.0,
        "history_candidate_sim_p75": 0.0,
        "history_candidate_sim_p90": 0.0,
        "history_candidate_sim_p95": 0.0,
        "history_candidate_sim_top1_top2_gap": 0.0,
        "top1_sim": 0.0,
        "top10_sim": 0.0,
        "top_k_sim_gap": 0.0,

        "cand_pairwise_cos_mean": 0.0,
        "cand_pairwise_cos_std": 0.0,
        "cand_pairwise_cos_min": 0.0,
        "cand_centroid_norm": 0.0,
        "cand_cov_trace": 0.0,
        "cand_pca_singular_top3_sum": 0.0,

        "hist_adj_cos_mean": 0.0,
        "hist_adj_cos_std": 0.0,
        "hist_drift_cos": 0.0,
        "hist_cov_trace": 0.0,
        "hist_pca_singular_top3_sum": 0.0,

        "cand_sim_>0.3_ratio": 0.0,
        "cand_sim_>0.5_ratio": 0.0,
        "cand_sim_>0.7_ratio": 0.0,

        "cand_cluster_size_entropy_k3": 0.0,
        "ambiguous_cluster_score": 0.0,

        "hist_cand_centroid_cos": 0.0,
        "hist_cand_centroid_dist": 0.0,
        "c2h_recentK_mean": 0.0,
        "c2h_recentK_max": 0.0,
        "cand_density_wrt_hist": 0.0,
    }

    if H.size == 0 or C.size == 0:
        return res

    res["hist_embedding_variance"] = float(np.var(H))
    Hc = H - H.mean(axis=0, keepdims=True)
    res["hist_cov_trace"] = float(np.trace(Hc.T @ Hc) / max(len(H) - 1, 1))
    try:
        s = np.linalg.svd(Hc, full_matrices=False)[1]
        res["hist_pca_singular_top3_sum"] = float(np.sum(s[:3]))
    except Exception:
        pass

    if len(H) >= 2:
        Hn = _norm_rows(H)
        adj_sim = np.sum(Hn[:-1] * Hn[1:], axis=1)
        res["hist_adj_cos_mean"] = float(adj_sim.mean())
        res["hist_adj_cos_std"] = float(adj_sim.std())
    if len(H) >= 4:
        m = len(H) // 2
        h1 = H[:m].mean(axis=0, keepdims=True)
        h2 = H[m:].mean(axis=0, keepdims=True)
        h1n = h1 / (np.linalg.norm(h1) + 1e-12)
        h2n = h2 / (np.linalg.norm(h2) + 1e-12)
        res["hist_drift_cos"] = float(np.sum(h1n * h2n))

    res["cand_embedding_dispersion"] = float(np.var(C))
    Cc = C - C.mean(axis=0, keepdims=True)
    res["cand_cov_trace"] = float(np.trace(Cc.T @ Cc) / max(len(C) - 1, 1))
    try:
        s = np.linalg.svd(Cc, full_matrices=False)[1]
        res["cand_pca_singular_top3_sum"] = float(np.sum(s[:3]))
    except Exception:
        pass
    res["cand_centroid_norm"] = float(np.linalg.norm(C.mean(axis=0)))

    if len(C) >= 2:
        Cn = _norm_rows(C)
        S = Cn @ Cn.T
        iu = np.triu_indices(len(C), k=1)
        pair = S[iu]
        res["cand_pairwise_cos_mean"] = float(pair.mean())
        res["cand_pairwise_cos_std"] = float(pair.std())
        res["cand_pairwise_cos_min"] = float(pair.min())

    Hn = _norm_rows(H)
    Cn = _norm_rows(C)
    sim_CH = Cn @ Hn.T
    candidate_mean_sims = sim_CH.mean(axis=1)

    res["history_candidate_sim_max"] = float(np.max(candidate_mean_sims))
    res["history_candidate_sim_mean"] = float(np.mean(candidate_mean_sims))
    res["history_candidate_sim_min"] = float(np.min(candidate_mean_sims))
    res["history_candidate_sim_var"] = float(np.var(candidate_mean_sims))

    qs = _quantiles(candidate_mean_sims, [0.5, 0.75, 0.9, 0.95])
    res["history_candidate_sim_p50"] = qs[0]
    res["history_candidate_sim_p75"] = qs[1]
    res["history_candidate_sim_p90"] = qs[2]
    res["history_candidate_sim_p95"] = qs[3]

    sorted_sims = np.sort(candidate_mean_sims)[::-1]
    res["top1_sim"] = float(sorted_sims[0])
    res["top10_sim"] = float(sorted_sims[9] if len(sorted_sims) >= 10 else sorted_sims[-1])
    res["top_k_sim_gap"] = float(res["top1_sim"] - res["top10_sim"])
    if len(sorted_sims) >= 2:
        res["history_candidate_sim_top1_top2_gap"] = float(sorted_sims[0] - sorted_sims[1])

    for th in (0.3, 0.5, 0.7):
        res[f"cand_sim_>{th}_ratio"] = float(np.mean(candidate_mean_sims > th))

    K = min(3, len(H))
    if K >= 1:
        H_recent = H[-K:]
        Hrn = _norm_rows(H_recent)
        sim_recent = (Cn @ Hrn.T).mean(axis=1)
        res["c2h_recentK_mean"] = float(sim_recent.mean())
        res["c2h_recentK_max"] = float(sim_recent.max())

    h_cent = H.mean(axis=0, keepdims=True)
    c_cent = C.mean(axis=0, keepdims=True)
    hc = h_cent / (np.linalg.norm(h_cent) + 1e-12)
    cc = c_cent / (np.linalg.norm(c_cent) + 1e-12)
    res["hist_cand_centroid_cos"] = float(np.sum(hc * cc))
    res["hist_cand_centroid_dist"] = float(np.linalg.norm(c_cent - h_cent))

    dists = np.linalg.norm(C - h_cent, axis=1)
    res["cand_density_wrt_hist"] = float(np.mean(dists))

    # light k-means (k=3) if available
    try:
        from sklearn.cluster import KMeans
        from sklearn.metrics.pairwise import euclidean_distances
        if len(C) >= 3:
            try:
                km = KMeans(n_clusters=3, n_init="auto", random_state=42)
            except TypeError:
                km = KMeans(n_clusters=3, n_init=10, random_state=42)
            labels = km.fit_predict(C)
            intra_means = []
            for i in range(3):
                idx = np.where(labels == i)[0]
                if len(idx) >= 2:
                    Xj = C[idx]
                    d = euclidean_distances(Xj)
                    intra_means.append(float(np.mean(d)))
            res["ambiguous_cluster_score"] = float(np.std(intra_means)) if intra_means else 0.0
            counts = np.array([(labels == i).sum() for i in range(3)], dtype=float)
            p = counts / counts.sum()
            p = p[p > 0]
            entropy = -np.sum(p * np.log(p))
            res["cand_cluster_size_entropy_k3"] = float(entropy)
    except Exception:
        pass

    return res


def features_from_entry_ir(entry: Dict[str, Any]) -> Dict[str, float]:
    # This variant expects: entry["query"]["embedding"] and entry["candidates"][*]["embedding"] already provided.
    q = entry.get("query", {}).get("embedding", None)
    C = np.array([i["embedding"] for i in entry.get("candidates", []) if i.get("embedding") is not None], dtype=np.float32)

    res = {
        "entry_id": entry.get("entry_id", ""),
        "query_norm": 0.0,

        "cand_embedding_dispersion": 0.0,
        "cand_pairwise_cos_mean": 0.0,
        "cand_pairwise_cos_std": 0.0,
        "cand_pairwise_cos_min": 0.0,
        "cand_centroid_norm": 0.0,
        "cand_cov_trace": 0.0,
        "cand_pca_singular_top3_sum": 0.0,

        "query_candidate_sim_max": 0.0,
        "query_candidate_sim_mean": 0.0,
        "query_candidate_sim_min": 0.0,
        "query_candidate_sim_var": 0.0,
        "query_candidate_sim_p50": 0.0,
        "query_candidate_sim_p75": 0.0,
        "query_candidate_sim_p90": 0.0,
        "query_candidate_sim_p95": 0.0,
        "top1_sim": 0.0,
        "top10_sim": 0.0,
        "top_k_sim_gap": 0.0,
        "qc_centroid_cos": 0.0,
        "qc_centroid_dist": 0.0,

        "cand_sim_>0.3_ratio": 0.0,
        "cand_sim_>0.5_ratio": 0.0,
        "cand_sim_>0.7_ratio": 0.0,
        "cand_cluster_size_entropy_k3": 0.0,
        "ambiguous_cluster_score": 0.0,
        "cand_density_wrt_query": 0.0,
    }

    if q is None or C.size == 0:
        return res

    q = np.array(q, dtype=np.float32).reshape(-1)
    res["query_norm"] = float(np.linalg.norm(q))
    res["cand_embedding_dispersion"] = float(np.var(C))

    Cc = C - C.mean(axis=0, keepdims=True)
    denom = max(len(C) - 1, 1)
    res["cand_cov_trace"] = float(np.trace(Cc.T @ Cc) / denom)
    try:
        s = np.linalg.svd(Cc, full_matrices=False)[1]
        res["cand_pca_singular_top3_sum"] = float(np.sum(s[:3]))
    except Exception:
        pass

    c_cent = C.mean(axis=0, keepdims=True)
    res["cand_centroid_norm"] = float(np.linalg.norm(c_cent))

    if len(C) >= 2:
        Cn = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-12)
        S = Cn @ Cn.T
        iu = np.triu_indices(len(C), k=1)
        pair = S[iu]
        res["cand_pairwise_cos_mean"] = float(pair.mean())
        res["cand_pairwise_cos_std"] = float(pair.std())
        res["cand_pairwise_cos_min"] = float(pair.min())

    qn = q / (np.linalg.norm(q) + 1e-12)
    Cn = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-12)
    sims = Cn @ qn

    res["query_candidate_sim_max"] = float(np.max(sims))
    res["query_candidate_sim_mean"] = float(np.mean(sims))
    res["query_candidate_sim_min"] = float(np.min(sims))
    res["query_candidate_sim_var"] = float(np.var(sims))

    qs = _quantiles(sims, [0.5, 0.75, 0.9, 0.95])
    res["query_candidate_sim_p50"] = qs[0]
    res["query_candidate_sim_p75"] = qs[1]
    res["query_candidate_sim_p90"] = qs[2]
    res["query_candidate_sim_p95"] = qs[3]

    sorted_sims = np.sort(sims)[::-1]
    res["top1_sim"] = float(sorted_sims[0])
    res["top10_sim"] = float(sorted_sims[9] if len(sorted_sims) >= 10 else sorted_sims[-1])
    res["top_k_sim_gap"] = float(res["top1_sim"] - res["top10_sim"])

    hc = qn.reshape(1, -1)
    cc = c_cent / (np.linalg.norm(c_cent) + 1e-12)
    res["qc_centroid_cos"] = float(np.sum(hc * cc))
    res["qc_centroid_dist"] = float(np.linalg.norm(c_cent - q.reshape(1, -1)))

    dists = np.linalg.norm(C - q.reshape(1, -1), axis=1)
    res["cand_density_wrt_query"] = float(np.mean(dists))

    try:
        from sklearn.cluster import KMeans
        from sklearn.metrics.pairwise import euclidean_distances
        if len(C) >= 3:
            try:
                km = KMeans(n_clusters=3, n_init="auto", random_state=42)
            except TypeError:
                km = KMeans(n_clusters=3, n_init=10, random_state=42)
            labels = km.fit_predict(C)
            intra_means = []
            for i in range(3):
                idx = np.where(labels == i)[0]
                if len(idx) >= 2:
                    Xj = C[idx]
                    d = euclidean_distances(Xj)
                    intra_means.append(float(np.mean(d)))
            res["ambiguous_cluster_score"] = float(np.std(intra_means)) if intra_means else 0.0
            counts = np.array([(labels == i).sum() for i in range(3)], dtype=float)
            p = counts / counts.sum()
            p = p[p > 0]
            entropy = -np.sum(p * np.log(p))
            res["cand_cluster_size_entropy_k3"] = float(entropy)
    except Exception:
        pass

    return res


def compute_features_from_entries_jsonl(
    entries_jsonl_path: str,
    dataset_kind: str,
) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    with open(entries_jsonl_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            entry = json.loads(line)
            if dataset_kind in {"ml-1m", "amazon-game"}:
                rows.append(features_from_entry_reco(entry))
            elif dataset_kind in {"rank_ir", "ir", "retrieval", "search"}:
                rows.append(features_from_entry_ir(entry))
            else:
                rows.append(features_from_entry_reco(entry))
            if (i % 200) == 0:
                print(f"[Feat] {i} entries processed")
    df = pd.DataFrame(rows)
    return df

# ---------- Safe IR parsing: slice candidate block only ----------
_HDR_CANDIDATES = re.compile(
    r'(?im)^\s*-\s*\*\*Candidate\s+Passages\*\*\s*:\s*$|^\s*\*\*Candidate\s+Passages\*\*\s*:\s*$',
    re.M
)
_HDR_END_BLOCK = re.compile(
    r'(?im)^\s*###\s*Constraints\s*$|^\s*###\s*Output\s*Format\s*$',
    re.M
)

def _slice_candidates_block(full_text: str) -> Tuple[str, int, int]:
    m = _HDR_CANDIDATES.search(full_text)
    if not m:
        return "", -1, -1
    start = m.end()
    me = _HDR_END_BLOCK.search(full_text, start)
    end = me.start() if me else len(full_text)
    return full_text[start:end], start, end

# ---------- Query line ----------
_IR_QUERY_RE = re.compile(r'(?im)^\s*(?:-\s*)?Search\s*Query\s*:\s*(.+?)\s*$')

def _extract_query(full_text: str) -> Tuple[str, Tuple[int, int]]:
    m = _IR_QUERY_RE.search(full_text)
    if not m:
        return "", (-1, -1)
    return m.group(1).strip(), m.span(1)

# ---------- Candidate blocks ----------
_IR_PASSAGE_BLOCK_RE = re.compile(
    r'(?ms)^\s*(?:§\s*)?\[(?P<ID>[A-Za-z0-9]{1,3})\]\s*(?!>)'
    r'(?P<TEXT>.+?)'
    r'(?=^\s*(?:§\s*)?\[[A-Za-z0-9]{1,3}\]\s*(?!>)|\Z)'
)

def _extract_candidates(cblock: str, block_abs_start: int) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for m in _IR_PASSAGE_BLOCK_RE.finditer(cblock):
        cid = m.group('ID').upper()
        l_rel, r_rel = m.span('TEXT')
        out.append({
            'id': cid,
            'text': m.group('TEXT'),
            'abs_span': (block_abs_start + l_rel, block_abs_start + r_rel)
        })
    return out

_COUNT_RE = re.compile(r'(?i)I\s+will\s+provide\s+you\s+with\s+(\d+)\s+passages')

def parse_ir_prompt_safely(full_text: str) -> Dict[str, object]:
    cnt_expect = None
    m_cnt = _COUNT_RE.search(full_text)
    if m_cnt:
        try:
            cnt_expect = int(m_cnt.group(1))
        except Exception:
            pass

    q_text, q_span = _extract_query(full_text)
    cblock, c_start, _ = _slice_candidates_block(full_text)
    raw_items = _extract_candidates(cblock, c_start) if c_start != -1 else []

    seen = set()
    cleaned = []
    for c in raw_items:
        tid = c['id']
        ttx = (c['text'] or '').strip()
        if not ttx or tid in seen:
            continue
        seen.add(tid)
        cleaned.append({'id': tid, 'text': ttx, 'abs_span': c['abs_span']})

    warn = None
    if cnt_expect is not None and cnt_expect != len(cleaned):
        warn = f"Expected {cnt_expect} candidates but parsed {len(cleaned)}."

    return {
        'query_text': q_text,
        'query_span': q_span,
        'candidates': cleaned,
        'n_expected': cnt_expect,
        'n_parsed': len(cleaned),
        'warning': warn
    }

# ---------- Char span -> token span (reuse your existing helper if present) ----------
def char_span_to_token_span(offsets: List[Tuple[int, int]], char_start: int, char_end: int) -> Optional[Tuple[int, int]]:
    ts = te = None
    for i, (s, e) in enumerate(offsets):
        if s == 0 and e == 0:
            continue
        if not (e <= char_start or s >= char_end):
            if ts is None:
                ts = i
            te = i
    return None if ts is None else (ts, te)

def pool_embedding(hidden_last, token_start: int, token_end: int, strategy: str = "mean"):
    if strategy == "start":
        return hidden_last[token_start]
    if strategy == "end":
        return hidden_last[token_end]
    if strategy == "both":
        import numpy as np
        return np.concatenate([hidden_last[token_start], hidden_last[token_end]], axis=-1)
    l = max(0, token_start)
    r = max(l, token_end)
    return hidden_last[l:r+1].mean(axis=0)

def process_one_entry_ir_safe(idx: int, prompt: str, hidden_last, offsets,
                              pooling: str = "mean", print_warn_every: int = 50) -> Dict[str, object]:
    parsed = parse_ir_prompt_safely(prompt)

    # Query vector
    q_text = parsed['query_text']
    q_span = parsed['query_span']
    query_obj = {"text": q_text, "embedding": None}
    if q_span and q_span[0] >= 0:
        q_tok = char_span_to_token_span(offsets, q_span[0], q_span[1])
        if q_tok is not None:
            qs, qe = q_tok
            query_obj["embedding"] = pool_embedding(hidden_last, qs, qe, strategy=pooling)
        elif (idx % print_warn_every) == 0:
            print(f"[IR {idx}] WARN: Query span not covered by tokens (truncated?)")

    # Candidate vectors
    cand_list = []
    for c in parsed['candidates']:
        l_abs, r_abs = c['abs_span']
        tspan = char_span_to_token_span(offsets, l_abs, r_abs)
        if tspan is not None:
            s, e = tspan
            vec = pool_embedding(hidden_last, s, e, strategy=pooling)
            cand_list.append({"id": c["id"], "text": c["text"], "embedding": vec})
        else:
            cand_list.append({"id": c["id"], "text": c["text"], "embedding": None})

    if parsed.get("warning") and (idx % print_warn_every) == 0:
        print(f"[IR {idx}] NOTE: {parsed['warning']}")

    return {"entry_id": idx, "query": query_obj, "candidates": cand_list}

# ---------- HF forward and IR embedding extraction ----------
def extract_item_embeddings_ir(
    tokenizer,
    model,
    prompts: List[str],
    pooling: str = "mean",
    max_length: int = 4096,
    batch_size: int = 8,
    print_every: int = 50,
):
    """
    Tokenize prompts (with offset_mapping), run model to get last hidden state,
    then pool query/candidate spans into embeddings using the safe IR parser.
    Returns a list of entries: {'entry_id', 'query': {'text','embedding'}, 'candidates': [...] }.
    """
    import torch
    import numpy as np

    results: List[Dict] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(prompts), batch_size):
            end = min(start + batch_size, len(prompts))
            batch_prompts = prompts[start:end]
            batch_idx = list(range(start, end))

            tok = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
                return_offsets_mapping=True,
            )
            offsets_b = tok.pop("offset_mapping")
            tok = {k: v.to(model.device) for k, v in tok.items()}

            out = model(**tok, output_hidden_states=True, return_dict=True)
            last_hidden_b = out.hidden_states[-1]  # (B, T, H)

            for bi in range(last_hidden_b.size(0)):
                i = batch_idx[bi]
                hidden_last = last_hidden_b[bi].detach().cpu().numpy()
                offsets = [tuple(map(int, p)) for p in offsets_b[bi].tolist()]
                entry = process_one_entry_ir_safe(
                    idx=i,
                    prompt=batch_prompts[bi],
                    hidden_last=hidden_last,
                    offsets=offsets,
                    pooling=pooling,
                    print_warn_every=print_every,
                )
                # Cast to float32 lists if needed by downstream writer
                if entry["query"]["embedding"] is not None:
                    entry["query"]["embedding"] = entry["query"]["embedding"].astype("float32").tolist()
                for c in entry["candidates"]:
                    if c["embedding"] is not None:
                        c["embedding"] = c["embedding"].astype("float32").tolist()
                results.append(entry)

            del out, last_hidden_b
            torch.cuda.empty_cache()
    return results