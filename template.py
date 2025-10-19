# template.py
from typing import List, Dict
import pandas as pd


rank_system = "You are an information retrieval assistant."


# Domain prompts used by agent-style system selection
movie_rec_prompt = (
    "You are a recommendation assistant. Select exactly 10 items from the candidate set and "
    "provide a ranked list of movie IDs. Ensure all selected items belong to the candidate set. "
    "Consider both session interactions and user history. You may include reasoning inside "
    "<thought></thought> when needed.\n"
    "## Output Format\n"
    "<output>Recommendation result: [MOVIE IDS IN ORDER, SEPARATED BY COMMA]</output>"
)

game_rec_prompt = (
    "You are a recommendation assistant. Select exactly 10 items from the candidate set and "
    "provide a ranked list of game IDs. Ensure all selected items belong to the candidate set. "
    "Consider both session interactions and user history. You may include reasoning inside "
    "<thought></thought> when needed.\n"
    "## Output Format\n"
    "<output>Recommendation result: [GAME IDS IN ORDER, SEPARATED BY COMMA]</output>"
)

# Assistant prefixes (optional)
agent_prefix = ""
direct_prefix = "<output>"
CoT_prefix = "<thought>"

# =========================
# Dataset-aware prompt builders
# =========================
ITEM_MARK = "§"

def get_agent_system(dataset_name: str = "generic", use_domain_prompt: bool = False) -> str:
    """Return a default system prompt by dataset."""
    name = (dataset_name or "").lower()
    if name in {"ml-1m", "movielens", "movie"}:
        return movie_rec_prompt
    if name in {"amazon-game", "steam", "game"}:
        return game_rec_prompt
    if name in {"rank_ir", "retrieval", "ir"}:
        return rank_system
    raise ValueError(f"Unsupported dataset: {dataset_name}")

def _row_by_id(df: pd.DataFrame, item_id: int) -> pd.Series:
    row = df.loc[df["item_id"] == item_id]
    return row.iloc[0] if not row.empty else pd.Series()

def get_item_info(item_ids: List[int], item_df: pd.DataFrame, dataset_name: str) -> str:
    """Render per-item lines for the prompt body, dataset-specific."""
    lines: List[str] = []
    if (dataset_name or "").lower() in {"ml-1m", "movielens", "movie"}:
        for mid in item_ids:
            row = _row_by_id(item_df, mid)
            title = row.get("title", "[Unknown Title]") if not row.empty else "[Unknown Title]"
            genres = row.get("genres", "[Unknown Genres]") if not row.empty else "[Unknown Genres]"
            lines.append(f"{ITEM_MARK}ID:{mid} Title:{title}; Genres:{genres}")
    elif (dataset_name or "").lower() in {"amazon-game", "steam", "game"}:
        for mid in item_ids:
            row = _row_by_id(item_df, mid)
            title = row.get("title", "[Unknown Title]") if not row.empty else "[Unknown Title]"
            lines.append(f"{ITEM_MARK}ID:{mid} Title:{title}")
    else:
        for mid in item_ids:
            lines.append(f"{ITEM_MARK}ID:{mid}")
    return "\n".join(lines)

_USER_MESSAGE_TMPL = (
    "User's watched history:\n"
    "{history_items}\n"
    "Candidate items:\n"
    "{candidate_items}\n"
)

def build_prompt(history_ids: List[int], candidate_ids: List[int],
                 item_df: pd.DataFrame, dataset_name: str) -> str:
    """Build recommendation prompt body with history and candidates sections."""
    history_info = get_item_info(history_ids, item_df, dataset_name)
    candidate_info = get_item_info(candidate_ids, item_df, dataset_name)
    return _USER_MESSAGE_TMPL.format(history_items=history_info, candidate_items=candidate_info)

def texts_to_messages(system: str, user_text: str,
                      assistant_prefix: str, assistant_message: bool = True) -> List[Dict]:
    """Wrap system/user texts (and optional assistant prefix) into chat messages."""
    msgs = [{"role": "system", "content": system},
            {"role": "user", "content": user_text}]
    if assistant_message:
        msgs.append({"role": "assistant", "content": assistant_prefix})
    return msgs

def generate_llm_inputs(samples: List[List], item_df: pd.DataFrame,
                        system: str, assistant_prefix: str,
                        dataset_name: str) -> List[List[Dict]]:
    """
    Build chat inputs for recommendation datasets.
    Each sample is expected as [history_ids, ..., candidate_ids] (history at idx 0, candidates at idx 2).
    """
    chats: List[List[Dict]] = []
    for sample in samples:
        user_prompt = build_prompt(sample[0], sample[2], item_df, dataset_name)
        chats.append(texts_to_messages(system, user_prompt, assistant_prefix))
    return chats

# =========================
# IR prompts (rank-all setting)
# =========================
IR_AGENT_PROMPT = (
    "{task}.\n\n"
    "### Inputs\n"
    "- {query}.\n\n"
    "- **Candidate Passages**:\n"
    "{candidates_block}\n\n"
    "When generating the results:\n"
    "- Ensure all IDs belong to the provided candidate set.\n"
    "- You may include reasoning inside <thought></thought> if needed.\n\n"
    "### Constraints\n"
    "- Include all passage IDs in descending order of relevance.\n"
    "- Do not invent new IDs.\n\n"
    "### Output Format\n"
    "<output>[ID1] > [ID2] > ...</output>\n"
)

def build_ir_prompt_from_parsed(record: Dict, mode: str = "agent") -> str:
    """Build IR prompt text from a pre-parsed record."""
    task_desc = record["task_desc"]
    query = record["search_query_line"]
    cblock = record["options_text"]
    if mode.lower() == "direct":
        return IR_DIRECT_PROMPT.format(task=task_desc, query=query, candidates_block=cblock)
    if mode.lower() == "cot":
        return IR_COT_PROMPT.format(task=task_desc, query=query, candidates_block=cblock)
    return IR_AGENT_PROMPT.format(task=task_desc, query=query, candidates_block=cblock)

def generate_llm_inputs_rank(records: List[Dict], system: str,
                             assistant_prefix: str, dataset_name: str) -> List[List[Dict]]:
    """Build chat inputs for IR datasets from parsed records."""
    chats: List[List[Dict]] = []
    for rec in records:
        prompt_text = build_ir_prompt_from_parsed(rec, mode=system)
        chats.append(texts_to_messages(rank_system, prompt_text, assistant_prefix))
    return chats
