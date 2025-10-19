# regression_utils.py
import os, re, json
import numpy as np
import pandas as pd
import lightgbm as lgb
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Any
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold


# ============================================================
# Basic configuration dataclasses
# ============================================================
@dataclass
class PathCfg:
    base: str
    dataset: str
    model: str
    save_dir: str
    def __post_init__(self):
        self.case_dir = os.path.join(self.base, f"{self.dataset}/{self.model}")
        self.data_path = os.path.join(self.base, self.dataset, "processed_data")
        self.saved_file_path = os.path.join(self.case_dir, "saved_data")
        self.saved_result_path = os.path.join(self.case_dir, "saved_result")
        self.save_dir = os.path.join(self.case_dir, "results")
        os.makedirs(self.saved_file_path, exist_ok=True)
        os.makedirs(self.saved_result_path, exist_ok=True)
        os.makedirs(self.save_dir, exist_ok=True)


@dataclass
class ExpCfg:
    exp_version: str = "ranking_v4"
    dataset_name: str = "ml-1m"
    k_runs: int = 5
    metric_ndcg_base: str = "ndcg10"
    metric_recall_base: str = "recall10"
    direct_prefix: str = "ranking_direct"
    cot_prefix: str = "ranking_CoT"
    genre_cols: Tuple[str, ...] = (
        "history_genre_entropy",
        "candidate_genre_entropy",
        "candidate_avg_popularity",
        "history_avg_popularity",
    )


@dataclass
class SplitCfg:
    dev_ratio: float = 0.25
    seed: int = 2025


@dataclass
class LabelCfg:
    lambda_cost: float = 0
    eps_equal: float = 3e-3
    alpha_var: float = 5.0
    use_trimmed_mean: bool = True
    trim_ratio: float = 0.1


@dataclass
class ModelCfg:
    hgb_params: Dict[str, Any] = None
    use_isotonic: bool = False
    use_isotonic_cv: bool = True
    iso_cv_folds: int = 5
    iso_cv_seed: int = 0
    use_monotone_for_delta_tokens: bool = True


@dataclass
class FeatCfg:
    include_delta_tokens: bool = True


@dataclass
class SelectCfg:
    rates_grid: np.ndarray = np.linspace(0, 1, 101)
    budget_points: int = 60
    utopia_weights: Tuple[float, float] = (1.0, 1.0)
    epsilon_target: str = "delta"
    epsilon_delta: float = -0.001


@dataclass
class OutCfg:
    save_gate_config: bool = True
    save_summary: bool = True
    save_dev_curve_csv: bool = True
    save_test_predictions: bool = True
    verbose: bool = True


# ============================================================
# Helper functions
# ============================================================
def _np_to_py(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)

def jsonify(obj, **kwargs):
    return json.dumps(obj, default=_np_to_py, **kwargs)

def _pretty(title: str):
    line = "=" * len(title)
    print(f"\n{title}\n{line}")

def coalesce_array(primary: Optional[np.ndarray], fallback: np.ndarray) -> np.ndarray:
    if primary is None:
        return fallback.copy()
    if primary.dtype.kind == "f":
        mask = np.isnan(primary)
        if mask.any():
            out = primary.copy()
            out[mask] = fallback[mask]
            return out
    return primary

def trimmed_mean(x: np.ndarray, trim=0.1, axis=1):
    x = np.sort(x, axis=axis)
    k = x.shape[axis]
    t = int(np.floor(trim * k))
    sl = [slice(None)] * x.ndim
    sl[axis] = slice(t, k - t if k - t > t else t)
    cut = x[tuple(sl)]
    return cut.mean(axis=axis) if cut.size else x.mean(axis=axis)

def _version_number(exp_version: str) -> str:
    assert isinstance(exp_version, str)
    return exp_version.split("_v")[-1]


# ============================================================
# 1、data loading (Direct/CoT metrics + tokens + features)
# ============================================================
def _find_run_cols(df: pd.DataFrame, base: str, k: int) -> List[str]:
    cols = [f"{base}_{i}" for i in range(1, k+1) if f"{base}_{i}" in df.columns]
    if len(cols) == k: return cols
    pattern = re.compile(rf"^{re.escape(base)}\s*[_\- ]?\s*(\d+)$", re.IGNORECASE)
    hits = []
    for c in df.columns:
        m = pattern.match(str(c))
        if m:
            try:
                idx = int(m.group(1))
                if 1 <= idx <= k: hits.append((idx, c))
            except ValueError:
                pass
    hits.sort(key=lambda x: x[0])
    cols = [c for _, c in hits]
    if len(cols) != k:
        raise ValueError(f"cannot find complete {base}_1..{base}_{k} columns. found: {cols}")
    return cols

def _get_col(df: pd.DataFrame, name: str):
    return df[name].to_numpy(dtype=float) if name in df.columns else None

def load_metrics_csv(csv_path: str, k_runs: int, ndcg_base: str, recall_base: str, load_recall: int = 0) -> Dict[str, np.ndarray]:
    """
    load metrics CSV, support optional to load recall related columns.
    load_recall: 1 (default) load recall related columns, 0 then not load recall related columns.
    """
    df = pd.read_csv(csv_path)
    df = df.replace(-1, 0)
    ndcg_cols = _find_run_cols(df, ndcg_base, k_runs)
    ndcg_runs = df[ndcg_cols].to_numpy(dtype=float)
    mean_ndcg = ndcg_runs.mean(axis=1)
    avg_ndcg_at_10 = coalesce_array(_get_col(df, "avg_ndcg@10"), mean_ndcg)
    index = df["index"].to_numpy() if "index" in df.columns else np.arange(len(df))
    result = dict(
        ndcg_runs=ndcg_runs,
        mean_ndcg=mean_ndcg,
        avg_ndcg_at_10=avg_ndcg_at_10,
        index=index,
    )
    if load_recall:
        recall_cols = _find_run_cols(df, recall_base, k_runs)
        recall_runs = df[recall_cols].to_numpy(dtype=float)
        mean_recall = recall_runs.mean(axis=1)
        avg_recall_at_10 = coalesce_array(_get_col(df, "avg_recall@10"), mean_recall)
        result.update(dict(
            recall_runs=recall_runs,
            mean_recall=mean_recall,
            avg_recall_at_10=avg_recall_at_10,
        ))
    return result

def load_mode_pair(direct_csv: str, cot_csv: str, exp: ExpCfg, load_recall: int = 0) -> Dict[str, np.ndarray]:
    D = load_metrics_csv(direct_csv, exp.k_runs, exp.metric_ndcg_base, exp.metric_recall_base, load_recall=load_recall)
    C = load_metrics_csv(cot_csv,    exp.k_runs, exp.metric_ndcg_base, exp.metric_recall_base, load_recall=load_recall)
    if len(D["index"]) != len(C["index"]):
        raise ValueError("Direct and CoT have different number of samples")
    result = dict(
        direct_ndcg_runs=D["ndcg_runs"],  cot_ndcg_runs=C["ndcg_runs"],
        direct_mean_ndcg=D["mean_ndcg"], cot_mean_ndcg=C["mean_ndcg"],
        index=D["index"],
    )
    if load_recall:
        result.update(dict(
            direct_recall_runs=D.get("recall_runs", None),
            cot_recall_runs=C.get("recall_runs", None),
            direct_mean_recall=D.get("mean_recall", None),
            cot_mean_recall=C.get("mean_recall", None),
        ))
    return result

def _repeat_avg_tokens_to_runs(avg_tokens: np.ndarray, k: int) -> np.ndarray:
    avg_tokens = np.asarray(avg_tokens, dtype=float).reshape(-1)
    return np.repeat(avg_tokens[:, None], k, axis=1)

def load_token_means(paths: PathCfg, exp: ExpCfg) -> Dict[str, np.ndarray]:
    num = _version_number(exp.exp_version)
    d_te = np.load(os.path.join(paths.saved_result_path, f"{exp.direct_prefix}_v{num}_test_averages.npy"))
    c_te = np.load(os.path.join(paths.saved_result_path, f"{exp.cot_prefix}_v{num}_test_averages.npy"))
    d_tr = np.load(os.path.join(paths.saved_result_path, f"{exp.direct_prefix}_v{num}_train_averages.npy"))
    c_tr = np.load(os.path.join(paths.saved_result_path, f"{exp.cot_prefix}_v{num}_train_averages.npy"))
    return dict(
        direct_train_token_mean=d_tr, cot_train_token_mean=c_tr,
        direct_test_token_mean=d_te,  cot_test_token_mean=c_te,
    )

def load_base_features(paths: PathCfg, exp: ExpCfg) -> Tuple[pd.DataFrame, pd.DataFrame]:
    num = _version_number(exp.exp_version)
    tr = os.path.join(paths.saved_file_path, f"{exp.direct_prefix}_v{num}_{exp.dataset_name}_train_features.csv")
    te = os.path.join(paths.saved_file_path, f"{exp.direct_prefix}_v{num}_{exp.dataset_name}_test_features.csv")
    tr = pd.read_csv(tr)
    te = pd.read_csv(te)
    return tr, te

def load_and_merge_genre_features(paths: PathCfg, exp: ExpCfg,
                                  base_train: pd.DataFrame, base_test: pd.DataFrame):
    num = _version_number(exp.exp_version)
    trg = os.path.join(paths.data_path, f"train_feature_df_v{num}.csv")
    teg = os.path.join(paths.data_path,  f"test_feature_df_v{num}.csv")
    genre_tr, genre_te = pd.read_csv(trg), pd.read_csv(teg)
    keep = [c for c in exp.genre_cols if c in genre_tr.columns]
    if keep:
        train = pd.concat([base_train.reset_index(drop=True), genre_tr[keep].reset_index(drop=True)], axis=1)
        test  = pd.concat([base_test.reset_index(drop=True),  genre_te[keep].reset_index(drop=True)], axis=1)
    else:
        train, test = base_train.copy(), base_test.copy()
    for df in (train, test):
        if "entry_id" in df.columns: df.drop(columns=["entry_id"], inplace=True)
    return train, test

def load_and_merge_prompt_features(paths: PathCfg, exp: ExpCfg,                        
                                   base_train: pd.DataFrame, base_test: pd.DataFrame):  
    """try to load prompt feature and merge with current feature; if file not found, return original."""                 
    num = _version_number(exp.exp_version)                                                                
                                                                        
    tr_path = os.path.join(paths.saved_file_path, f"{exp.direct_prefix}_v{num}_train_prompt_feature.csv")          
    te_path = os.path.join(paths.saved_file_path, f"{exp.direct_prefix}_v{num}_test_prompt_feature.csv")          

    print('load train and test prompt feature from:', tr_path, te_path)                                  
    if (tr_path is None) or (te_path is None):                                                                
        print("[Warn] Not found Prompt feature file; skip Prompt feature merge.")                         
        return base_train, base_test                                                    
    prompt_tr = pd.read_csv(tr_path)                                                    
    prompt_te = pd.read_csv(te_path)                                                    
    for df in (prompt_tr, prompt_te):                                                 
        if "entry_id" in df.columns: df.drop(columns=["entry_id"], inplace=True)        
    train = pd.concat([base_train.reset_index(drop=True), prompt_tr.reset_index(drop=True)], axis=1) 
    test  = pd.concat([base_test.reset_index(drop=True),  prompt_te.reset_index(drop=True)], axis=1) 
    return train, test                                                                  

def load_experiment_bundle(paths: PathCfg, exp: ExpCfg,                                 
                           use_genre: bool = True, use_prompt: bool = True, load_recall: int = 0) -> Dict[str, Dict]:
    num = _version_number(exp.exp_version)
    train_pair = load_mode_pair(
        os.path.join(paths.saved_result_path, f"{exp.direct_prefix}_v{num}_train_ndcg.csv"),
        os.path.join(paths.saved_result_path, f"{exp.cot_prefix}_v{num}_train_ndcg.csv"),
        exp,
        load_recall=load_recall
    )
    test_pair = load_mode_pair(
        os.path.join(paths.saved_result_path, f"{exp.direct_prefix}_v{num}_test_ndcg.csv"),
        os.path.join(paths.saved_result_path, f"{exp.cot_prefix}_v{num}_test_ndcg.csv"),
        exp,
        load_recall=load_recall
    )
    T = load_token_means(paths, exp)
    dtr_tok_runs = _repeat_avg_tokens_to_runs(T["direct_train_token_mean"], exp.k_runs)
    ctr_tok_runs = _repeat_avg_tokens_to_runs(T["cot_train_token_mean"],    exp.k_runs)
    dte_tok_runs = _repeat_avg_tokens_to_runs(T["direct_test_token_mean"],  exp.k_runs)
    cte_tok_runs = _repeat_avg_tokens_to_runs(T["cot_test_token_mean"],     exp.k_runs)

    base_tr, base_te = load_base_features(paths, exp)

    X_train, X_test = base_tr, base_te
    if use_genre:
        X_train, X_test = load_and_merge_genre_features(paths, exp, X_train, X_test)
        print('load train and test genre feature.')
    else:
        print("[Info] genre feature fusion is disabled.")
    if use_prompt:
        X_train, X_test = load_and_merge_prompt_features(paths, exp, X_train, X_test)
        print('load train and test prompt feature.')
    else:
        print("[Info] prompt feature fusion is disabled.")

    train_dict = {
        "direct_ndcg_runs": train_pair["direct_ndcg_runs"],
        "cot_ndcg_runs":    train_pair["cot_ndcg_runs"],
        "direct_token_runs": dtr_tok_runs,
        "cot_token_runs":    ctr_tok_runs,
        "direct_ndcg_mean": train_pair["direct_mean_ndcg"],
        "cot_ndcg_mean":    train_pair["cot_mean_ndcg"],
        "direct_token_mean": T["direct_train_token_mean"],
        "cot_token_mean":    T["cot_train_token_mean"],
        "X": X_train, "index": train_pair["index"],
    }
    test_dict = {
        "direct_ndcg_runs": test_pair["direct_ndcg_runs"],
        "cot_ndcg_runs":    test_pair["cot_ndcg_runs"],
        "direct_token_runs": dte_tok_runs,
        "cot_token_runs":    cte_tok_runs,
        "direct_ndcg_mean": test_pair["direct_mean_ndcg"],
        "cot_ndcg_mean":    test_pair["cot_mean_ndcg"],
        "direct_token_mean": T["direct_test_token_mean"],
        "cot_token_mean":    T["cot_test_token_mean"],
        "X": X_test, "index": test_pair["index"],
    }
    if load_recall:
        train_dict.update({
            "direct_recall_runs": train_pair.get("direct_recall_runs", None),
            "cot_recall_runs": train_pair.get("cot_recall_runs", None),
            "direct_mean_recall": train_pair.get("direct_mean_recall", None),
            "cot_mean_recall": train_pair.get("cot_mean_recall", None),
        })
        test_dict.update({
            "direct_recall_runs": test_pair.get("direct_recall_runs", None),
            "cot_recall_runs": test_pair.get("cot_recall_runs", None),
            "direct_mean_recall": test_pair.get("direct_mean_recall", None),
            "cot_mean_recall": test_pair.get("cot_mean_recall", None),
        })
    return {
        "train": train_dict,
        "test": test_dict,
    }


# ============================================================
# 2、Train→Dev split (3:1), avoid Test leakage
# ============================================================
def split_train_dev(train_split: Dict[str, np.ndarray], dev_ratio: float = 0.25, seed: int = 42):
    n = len(train_split["direct_ndcg_mean"])
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)
    cut = int(round(n * (1.0 - dev_ratio)))
    idx_tr, idx_dev = perm[:cut], perm[cut:]

    def sel(d, key, idx):
        v = d[key]
        return v.iloc[idx] if isinstance(v, pd.DataFrame) else v[idx]

    keys = [
        "direct_ndcg_runs","cot_ndcg_runs",
        "direct_token_runs","cot_token_runs",
        "direct_ndcg_mean","cot_ndcg_mean",
        "direct_token_mean","cot_token_mean",
        "index"
    ]
    train_new = {k: sel(train_split, k, idx_tr) for k in keys}
    dev_new   = {k: sel(train_split, k, idx_dev) for k in keys}
    train_new["X"] = train_split["X"].iloc[idx_tr].reset_index(drop=True)
    dev_new["X"]   = train_split["X"].iloc[idx_dev].reset_index(drop=True)
    return train_new, dev_new


# ============================================================
# 3、label (advantage regression) and train/calibrate
# ============================================================
def _mean_var_runs(runs: np.ndarray, use_trimmed: bool, trim_ratio: float):
    if runs.ndim == 1:
        return runs, np.zeros_like(runs)
    if use_trimmed:
        mu = trimmed_mean(runs, trim=trim_ratio, axis=1)
    else:
        mu = runs.mean(axis=1)
    var = runs.var(axis=1)
    return mu, var

def build_advantage_labels_from_split(
    split: Dict[str, np.ndarray],
    lambda_cost=0.0, eps_equal=1e-3, alpha_var=5.0,
    use_trimmed=True, trim_ratio=0.1
):
    mu_d, var_d = _mean_var_runs(split["direct_ndcg_runs"], use_trimmed, trim_ratio)
    mu_c, var_c = _mean_var_runs(split["cot_ndcg_runs"],    use_trimmed, trim_ratio)
    tok_d, _ = _mean_var_runs(split["direct_token_runs"], use_trimmed=False, trim_ratio=0.0)
    tok_c, _ = _mean_var_runs(split["cot_token_runs"],    use_trimmed=False, trim_ratio=0.0)

    U_d = mu_d - lambda_cost * (tok_d / 1000.0)
    U_c = mu_c - lambda_cost * (tok_c / 1000.0)
    A   = U_c - U_d

    # small advantage threshold: pull samples with |A| very small to 0 (more stable)
    A = np.where(np.abs(A) < eps_equal, 0.0, A)

    # higher variance, lower weight
    weight = np.exp(-alpha_var * (var_d + var_c))
    stats = dict(mu_direct=mu_d, mu_cot=mu_c, tok_direct=tok_d, tok_cot=tok_c, U_direct=U_d, U_cot=U_c)
    return A, weight, stats

def train_advantage_regressor(X: np.ndarray, A: np.ndarray, w: np.ndarray, hgb_params: dict = None):
    params = dict(learning_rate=0.05, max_bins=255, l2_regularization=1e-4,
                  early_stopping=True, random_state=0, verbose=0)
    if hgb_params: params.update(hgb_params)
    model = HistGradientBoostingRegressor(**params)
    model.fit(X, A, sample_weight=w)
    pred = model.predict(X)
    mse = float(mean_squared_error(A, pred, sample_weight=w))
    return model, mse

def fit_isotonic(A_hat_train: np.ndarray, A_true_train: np.ndarray):
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(A_hat_train, A_true_train)
    return iso

def fit_isotonic_cv(
    model: HistGradientBoostingRegressor,
    X_train: np.ndarray, y_train: np.ndarray,
    K=5, random_state=0
):
    kf = KFold(n_splits=K, shuffle=True, random_state=random_state)
    oof_pred = np.zeros(len(X_train))
    for tr, va in kf.split(X_train):
        model.fit(X_train[tr], y_train[tr])
        oof_pred[va] = model.predict(X_train[va])
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(oof_pred, y_train.astype(float))
    # fit the base model again on the full data
    model.fit(X_train, y_train)
    return model, iso

def refine_rate_local(A_hat, mu_d, mu_c, td, tc, r0, width=0.05, step=0.002, lambda_cost=0.0):
    lo, hi = max(0.0, r0-width), min(1.0, r0+width)
    grid = np.arange(lo, hi+1e-12, step)
    best = None
    order = np.argsort(-A_hat); n = len(A_hat)
    for r in grid:
        k = int(round(r*n))
        pred = np.zeros(n, dtype=int); 
        if k>0: pred[order[:k]] = 1
        res = eval_mixed_strategy(pred, mu_d, mu_c, td, tc, lambda_cost)
        res.update({"rate": float(r)})
        if best is None or res["mean_ndcg"] > best["mean_ndcg"]:
            best = res
    return best

def train_ranker_lgbm(
    X_tr, Y_tr, sample_weight=None,
    monotone_cst=None, X_dev=None, A_dev=None,
    random_state: int = 0, label_gain=None
):
    params = dict(
        objective="lambdarank",
        metric="ndcg",
        ndcg_eval_at=[10],
        learning_rate=0.05,
        num_leaves=63,
        min_data_in_leaf=100,
        max_depth=-1,
        feature_fraction=1.0,
        bagging_fraction=1.0,
        bagging_freq=0,
        reg_lambda=1e-2,
        verbosity=-1,
        seed=random_state,
    )
    if monotone_cst is not None:
        params["monotone_constraints"] = list(map(int, monotone_cst))
    if label_gain is not None:
        params["label_gain"] = list(map(float, label_gain))

    dtrain = lgb.Dataset(X_tr, label=Y_tr, weight=sample_weight, group=[len(X_tr)], free_raw_data=False)
    valid_sets, valid_names, callbacks = [dtrain], ["train"], []
    if X_dev is not None and A_dev is not None:
        dvalid = lgb.Dataset(X_dev, label=A_dev, group=[len(X_dev)], free_raw_data=False)
        valid_sets.append(dvalid); valid_names.append("dev")
        callbacks.append(lgb.early_stopping(stopping_rounds=200, verbose=False))

    booster = lgb.train(
        params, dtrain, num_boost_round=2000,
        valid_sets=valid_sets, valid_names=valid_names, callbacks=callbacks
    )
    return booster

def _make_rank_thresholds(A_train: np.ndarray, eps: float = 0.0,
                          quantiles=(0.5, 0.8, 0.95)):
    """
    take several quantiles on the positive advantage (A>eps) on the training set as the bin thresholds.
    let the number of thresholds = T, then the label range is 0..(T+1), so the length of label_gain is T+2.
    """
    A_pos = np.asarray(A_train, float)
    A_pos = A_pos[A_pos > eps]
    if A_pos.size < 50:
        thresholds = []
    else:
        qs = np.quantile(A_pos, quantiles).tolist()
        thresholds = sorted(set(float(x) for x in qs if np.isfinite(x)))

    T = len(thresholds)
    # the length must be T+2, cover the label 0..(T+1)
    # 0 level gain=0, the rest levels can use exponential/linear; here we use exponential to distinguish levels
    gains = [0.0] + [2.0**i for i in range(0, T + 1)]
    return thresholds, gains


def _assign_rank_labels(A: np.ndarray, thresholds: list, eps: float = 0.0) -> np.ndarray:
    """
    map the continuous A to the level labels {0,1,2,...,K} (int) by (eps, thresholds).
    A <= eps -> 0; beyond each threshold +1 level.
    """
    y = np.zeros_like(A, dtype=int)
    if len(thresholds) == 0:
        y = (A > eps).astype(int)
        return y
    y[A > eps] = 1
    for t in thresholds:
        y[A > t] += 1
    return y

# ============================================================
# 4、strategy evaluation and selection (Top-r / equivalence / budget)
# ============================================================
def eval_mixed_strategy(pred, mu_d, mu_c, tok_d, tok_c, lambda_cost=0.0):
    ndcg = np.where(pred==1, mu_c, mu_d)
    toks = np.where(pred==1, tok_c, tok_d)
    return dict(
        mean_ndcg=float(ndcg.mean()),
        mean_tokens=float(toks.mean()),
        total_tokens=float(toks.sum()),
        utility=float(ndcg.mean() - lambda_cost*(toks.mean()/1000.0)),
        cot_rate=float(pred.mean())
    )

def baselines(mu_d, mu_c, tok_d, tok_c, lambda_cost=0.0):
    n = len(mu_d)
    res_d = eval_mixed_strategy(np.zeros(n, dtype=int), mu_d, mu_c, tok_d, tok_c, lambda_cost)
    res_c = eval_mixed_strategy(np.ones(n,  dtype=int), mu_d, mu_c, tok_d, tok_c, lambda_cost)
    return {"Always-Direct": {"NDCG": res_d["mean_ndcg"], "Tokens_mean": res_d["mean_tokens"],
                              "Tokens_total": res_d["total_tokens"], "Utility": res_d["utility"]},
            "Always-CoT": {"NDCG": res_c["mean_ndcg"], "Tokens_mean": res_c["mean_tokens"],
                           "Tokens_total": res_c["total_tokens"], "Utility": res_c["utility"]}}

def tau_from_top_r(A_hat_ref, r): 
    return float(np.quantile(A_hat_ref, 1.0 - r))

def apply_policy(A_hat, tok_d, tok_c, policy: dict):
    name = policy["name"]
    if name in ("top_r_tau", "equivalence_tau"):
        tau = policy["tau"]; return (A_hat >= tau).astype(int)
    elif name == "budget_density":
        eta = policy["eta"]
        d_tok = tok_c - tok_d
        eps = 1e-12
        free = (d_tok <= 0) & (A_hat > 0)
        dens = A_hat / np.maximum(d_tok, eps)
        return (free | (dens >= eta)).astype(int)
    else:
        raise ValueError("Unknown policy")
    
def select_topr_by_rate(A_hat, rate: float):
    """take the top rate percentage by score from high to low; only use prediction score and sample number, no label, no leakage."""
    n = len(A_hat)
    k = int(round(rate * n))
    pred = np.zeros(n, dtype=int)
    if k > 0:
        pred[np.argsort(-A_hat)[:k]] = 1
    return pred

def sweep_rate_by_ndcg(A_hat, mu_d, mu_c, tok_d, tok_c, rates=None, lambda_cost=0.0):
    if rates is None: rates = np.linspace(0,1,41)
    order = np.argsort(-A_hat)
    n = len(A_hat)
    best, hist = None, []
    for r in rates:
        k = int(round(r*n))
        pred = np.zeros(n, dtype=int)
        if k>0: pred[order[:k]] = 1
        res = eval_mixed_strategy(pred, mu_d, mu_c, tok_d, tok_c, lambda_cost)
        res.update({"rate": float(r)})
        hist.append(res)
        if best is None or res["mean_ndcg"] > best["mean_ndcg"]:
            best = res
    return {"best": best, "all": hist}

def find_min_rate_for_target_ndcg(A_hat, mu_d, mu_c, tok_d, tok_c, target_ndcg, rates=None, lambda_cost=0.0):
    if rates is None: rates = np.linspace(0,1,101)
    order = np.argsort(-A_hat)
    n = len(A_hat)
    for r in np.sort(rates):
        k = int(round(r*n))
        pred = np.zeros(n, dtype=int)
        if k>0: pred[order[:k]] = 1
        res = eval_mixed_strategy(pred, mu_d, mu_c, tok_d, tok_c, lambda_cost)
        if res["mean_ndcg"] >= target_ndcg - 1e-12:
            res.update({"rate": float(r)}); return res
    return None

# ---- greedy under mean token budget and curve ----
def greedy_under_mean_token_budget(A_hat, mu_d, mu_c, tok_d, tok_c, target_mean_tokens, lambda_cost=0.0):
    n = len(A_hat)
    budget_total = float(target_mean_tokens) * n
    base_total = float(tok_d.sum())
    pred = np.zeros(n, dtype=int)
    d_tok = (tok_c - tok_d).astype(float)
    gain  = A_hat.astype(float)
    free_idx = np.where((d_tok <= 0) & (gain > 0))[0]
    pred[free_idx] = 1
    total_tokens = base_total + float(d_tok[free_idx].sum())
    # normal candidates: greedy by density
    mask = (d_tok > 0) & (gain > 0)
    cand = np.where(mask)[0]
    if cand.size>0:
        ratio = gain[cand] / d_tok[cand]
        order = cand[np.argsort(-ratio)]
        for i in order:
            need = float(d_tok[i])
            if total_tokens + need <= budget_total + 1e-9:
                pred[i] = 1; total_tokens += need
    res = eval_mixed_strategy(pred, mu_d, mu_c, tok_d, tok_c, lambda_cost)
    res.update({"budget_mean_tokens": float(target_mean_tokens),
                "selected": int(pred.sum()),
                "feasible": bool(res["mean_tokens"] <= target_mean_tokens + 1e-9)})
    return res

def sweep_mean_token_budgets(A_hat, mu_d, mu_c, tok_d, tok_c, mean_token_targets, lambda_cost=0.0):
    return [greedy_under_mean_token_budget(A_hat, mu_d, mu_c, tok_d, tok_c, t, lambda_cost) 
            for t in mean_token_targets]

def eta_for_mean_tokens(A_hat_ref, tok_d_ref, tok_c_ref, target_mean_tokens):
    n = len(A_hat_ref)
    d_tok = tok_c_ref - tok_d_ref
    density = np.where(d_tok > 0, A_hat_ref / d_tok, np.inf)
    order = np.argsort(-density)
    budget_total = float(target_mean_tokens) * n
    total_tokens = float(tok_d_ref.sum())
    eta = np.inf
    for idx in order:
        need = float(max(d_tok[idx], 0.0))
        if total_tokens + need <= budget_total + 1e-9:
            total_tokens += need
            eta = float(density[idx])
        else:
            break
    return eta

def rebuild_test_budget_policy(policy_dev: dict, A_hat_te, tok_d_te, tok_c_te):
    """
    recalculate η for Test using the same target_mean_tokens as Dev, to avoid using η from Dev directly on Test.
    """
    if policy_dev is None:
        return None
    T = float(policy_dev["target_mean_tokens"])
    eta = eta_for_mean_tokens(A_hat_te, tok_d_te, tok_c_te, target_mean_tokens=T)
    return {
        "name": "budget_density",
        "target_mean_tokens": T,
        "eta": float(eta),
        "free_rule": "use CoT if (Δtokens<=0 and A_hat>0)"
    }


# ---- Pareto and point selection ----
def _pareto_front(curve):
    pts = sorted(curve, key=lambda x: x["mean_tokens"])
    front, best_ndcg = [], -1.0
    for p in pts:
        if p["mean_ndcg"] > best_ndcg + 1e-12:
            front.append(p); best_ndcg = p["mean_ndcg"]
    return front

def _normalize_front(front):
    toks = np.array([p["mean_tokens"] for p in front], float)
    ndcg = np.array([p["mean_ndcg"] for p in front], float)
    t_min, t_max = toks.min(), toks.max()
    n_min, n_max = ndcg.min(), ndcg.max()
    t_norm = (toks - t_min) / max(t_max - t_min, 1e-12)
    n_norm = (ndcg - n_min) / max(n_max - n_min, 1e-12)
    t_good = 1.0 - t_norm
    return t_good, n_norm

def select_knee(front):
    x, y = _normalize_front(front)
    dist = np.abs(y - x) / np.sqrt(2.0)
    i = int(np.argmax(dist))
    return {**front[i], "selector": "knee", "idx": i}

def select_utopia(front, w_ndcg=1.0, w_tokens=1.0):
    x, y = _normalize_front(front)
    dist = np.sqrt(w_tokens*(1-x)**2 + w_ndcg*(1-y)**2)
    i = int(np.argmin(dist))
    return {**front[i], "selector": "utopia", "idx": i, "w_ndcg": w_ndcg, "w_tokens": w_tokens}

def select_epsilon(front, target_ndcg):
    cand = [p for p in front if p["mean_ndcg"] >= target_ndcg - 1e-12]
    if not cand: return None
    i = int(np.argmin([p["mean_tokens"] for p in cand]))
    return {**cand[i], "selector": "epsilon", "target_ndcg": float(target_ndcg)}


# ============================================================
# 5、print and save summary
# ============================================================
def report_and_save(
    save_dir: str,
    cfg_snapshot: Dict[str, Any],
    train_mse: float,
    base_dev: Dict, base_test: Dict,
    dev_params: Dict,              # selected r*/tau*/eta* & points
    dev_reports: Dict,             # reports of each strategy on Dev
    test_reports: Dict,            # reports of each strategy on Test
    curve_dev_df: Optional[pd.DataFrame] = None,
    verbose: bool = True,
    selected_cols: List[str] = None,
):
    if verbose:
        _pretty("CONFIG SNAPSHOT")
        print(jsonify(cfg_snapshot, indent=2))
        _pretty("TRAIN / FIT")
        print(f"train_mse: {train_mse:.6f}")

        _pretty("DEV / BASELINES")
        print(jsonify(base_dev, indent=2))

        _pretty("DEV / SELECTED PARAMS")
        print(jsonify(dev_params, indent=2))

        _pretty("DEV / POLICY REPORTS")
        print(jsonify(dev_reports, indent=2))

        _pretty("TEST / BASELINES")
        print(jsonify(base_test, indent=2))

        _pretty("TEST / FINAL REPORTS")
        print(jsonify(test_reports, indent=2))

    # save summary
    summary = {
        "config": cfg_snapshot,
        "selected_cols": selected_cols,
        "train_mse": train_mse,
        "dev": {"baselines": base_dev, "selected_params": dev_params, "reports": dev_reports},
        "test": {"baselines": base_test, "reports": test_reports},
    }
    summary_path = os.path.join(save_dir, "results_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=_np_to_py)
    print(f"[Saved] {summary_path}")

    if curve_dev_df is not None:
        curve_path = os.path.join(save_dir, "dev_budget_curve.csv")
        curve_dev_df.to_csv(curve_path, index=False)
        print(f"[Saved] {curve_path}")


def _load_selected_cols(default_json: str, override_path: Optional[str], all_cols: List[str]) -> List[str]:
    """
    path use override_path first, otherwise use default_json.
    JSON example: {"selected_cols": ["feat_a", "feat_b", ...]}
    only keep columns that exist in the current data; if empty, fallback to all_cols.
    """
    path = override_path or default_json
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        cols = [c for c in obj.get("selected_cols", []) if c in all_cols]
        if cols:
            print(f"[Info] loaded selected features ({len(cols)} cols) from {path}")
            return cols
        print(f"[Warn] columns in {path} are not available in the current data, fallback to all features.")
    else:
        print(f"[Warn] selected features json not found: {path}, fallback to all features.")
    return list(all_cols)