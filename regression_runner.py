# regression_runner.py
import os, json
import numpy as np
import pandas as pd
import argparse
import lightgbm as lgb
from dataclasses import asdict
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import mean_squared_error

# Unified import
from regression_utils import (
    PathCfg, ExpCfg, SplitCfg, LabelCfg, ModelCfg, FeatCfg, SelectCfg, OutCfg,
    load_experiment_bundle, split_train_dev, build_advantage_labels_from_split,
    train_ranker_lgbm, fit_isotonic,
    baselines, eval_mixed_strategy, select_knee, select_utopia, select_epsilon,
    tau_from_top_r, _make_rank_thresholds, _assign_rank_labels, _pareto_front,
    _np_to_py, _load_selected_cols
)


# ============================================================
# Evaluate averaged test results across policies
# ============================================================
def evaluate_test_result(paths, exp, bundle, selected_metric, num=4):
    direct_result = pd.read_csv(
        os.path.join(paths.saved_result_path, f"{exp.direct_prefix}_v{num}_test_ndcg.csv")
    )
    cot_result = pd.read_csv(
        os.path.join(paths.saved_result_path, f"{exp.cot_prefix}_v{num}_test_ndcg.csv")
    )
    pred_test = pd.read_csv(os.path.join(paths.save_dir, "predictions_test.csv"))

    direct_result = direct_result[selected_metric]
    cot_result = cot_result[selected_metric]
    direct_result["token_mean"] = bundle["test"]["direct_token_mean"]
    cot_result["token_mean"] = bundle["test"]["cot_token_mean"]

    mean_rows = []
    pred_cols = [col for col in pred_test.columns if col.startswith("pred")]
    for policy in pred_cols:
        preds = pred_test[policy].to_numpy()
        rows = [
            (cot_result.iloc[i] if p == 1 else direct_result.iloc[i])
            for i, p in enumerate(preds)
        ]
        mean_row = pd.DataFrame(rows).mean(axis=0).to_frame().T
        mean_row.index = [policy]
        mean_rows.append(mean_row)
    return pd.concat(mean_rows)


# ============================================================
# CLI
# ============================================================
def build_parser():
    p = argparse.ArgumentParser(description="Adaptive CoT regression and policy selection")
    p.add_argument("--dataset", type=str, default="ml-1m")
    p.add_argument("--model", type=str, default="Qwen3-4B")
    p.add_argument("--data_root", type=str, default="./data")
    p.add_argument("--out_root", type=str, default="./outputs")
    p.add_argument("--exp_version", type=str, default="ranking_v4")
    p.add_argument("--selected_json", type=str, default=None)

    # Label config
    p.add_argument("--lambda_cost", type=float, default=0.1)
    p.add_argument("--eps_equal", type=float, default=3e-3)
    p.add_argument("--alpha_var", type=float, default=5.0)
    p.add_argument("--use_trimmed_mean", type=int, default=1)
    p.add_argument("--trim_ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=2025)

    # Model config
    p.add_argument("--learning_rate", type=float, default=0.05)
    p.add_argument("--max_bins", type=int, default=255)
    p.add_argument("--l2_regularization", type=float, default=1e-4)
    p.add_argument("--max_iter", type=int, default=800)
    p.add_argument("--max_leaf_nodes", type=int, default=63)
    p.add_argument("--min_samples_leaf", type=int, default=50)
    p.add_argument("--use_isotonic", type=int, default=0)
    p.add_argument("--use_monotone_for_delta_tokens", type=int, default=1)

    # Feature switches
    p.add_argument("--use_genre", type=int, default=0)
    p.add_argument("--use_prompt", type=int, default=1)
    p.add_argument("--include_delta_tokens", type=int, default=1)

    # Selection strategy
    p.add_argument("--epsilon_target", type=str, default="always_best")
    p.add_argument("--epsilon_delta", type=float, default=0.0)
    return p


# ============================================================
# Helper utilities
# ============================================================
def base_cot_vs_direct(mu_d, mu_c, td, tc, lambda_cost=0.0):
    r_d = eval_mixed_strategy(np.zeros_like(mu_d, int), mu_d, mu_c, td, tc, lambda_cost)
    r_c = eval_mixed_strategy(np.ones_like(mu_d, int), mu_d, mu_c, td, tc, lambda_cost)
    return r_c["mean_ndcg"], r_d["mean_ndcg"]


def sweep_by_topr_budget(A_hat_pos, mu_pos, mu_neg, t_pos, t_neg, r_grid, lambda_cost=0.0):
    rows, n = [], len(A_hat_pos)
    order = np.argsort(-A_hat_pos)
    for r in np.asarray(r_grid):
        k = int(round(r * n))
        pred = np.zeros(n, int)
        if k > 0:
            pred[order[:k]] = 1
        rep = eval_mixed_strategy(pred, mu_neg, mu_pos, t_neg, t_pos, lambda_cost)
        rows.append({"rate": float(r), "mean_tokens": rep["mean_tokens"], "mean_ndcg": rep["mean_ndcg"]})
    return rows


def best_topr_policy(A_hat_pos, mu_pos, mu_neg, t_pos, t_neg, rates_grid, lambda_cost=0.0):
    curve = sweep_by_topr_budget(A_hat_pos, mu_pos, mu_neg, t_pos, t_neg, rates_grid, lambda_cost)
    best = max(curve, key=lambda d: d["mean_ndcg"])
    r_star, tau = best["rate"], tau_from_top_r(A_hat_pos, best["rate"])
    n = len(A_hat_pos)
    pred = np.zeros(n, int)
    if r_star > 0:
        pred[np.argsort(-A_hat_pos)[:int(round(r_star * n))]] = 1
    rep = eval_mixed_strategy(pred, mu_neg, mu_pos, t_neg, t_pos, lambda_cost)
    return {"r": float(r_star), "tau": float(tau), "report": rep, "curve": curve}


def apply_policy_symmetric(A_hat, td, tc, pol):
    mode = pol.get("positive_mode", "cot")
    if pol["name"] == "top_r_tau":
        score = A_hat if mode == "cot" else -A_hat
        sel = (score >= pol["tau"]).astype(int)
        return sel if mode == "cot" else 1 - sel
    elif pol["name"] in ("budget_topr_point", "utility_max_point"):
        score = A_hat if mode == "cot" else -A_hat
        n = len(score)
        k = int(round(pol["rate"] * n))
        pred = np.zeros(n, int)
        if k > 0:
            pred[np.argsort(-score)[:k]] = 1
        return pred if mode == "cot" else 1 - pred
    return np.zeros_like(A_hat, int)


# ============================================================
# Main
# ============================================================
def main(args):
    PATHS = PathCfg(base=args.out_root, dataset=args.dataset, model=args.model, save_dir="")
    EXP = ExpCfg(exp_version=args.exp_version, dataset_name=args.dataset)
    SPLIT = SplitCfg(seed=args.seed)
    LABEL = LabelCfg(
        lambda_cost=args.lambda_cost, eps_equal=args.eps_equal, alpha_var=args.alpha_var,
        use_trimmed_mean=bool(args.use_trimmed_mean), trim_ratio=args.trim_ratio,
    )
    MODEL = ModelCfg(
        hgb_params=dict(
            learning_rate=args.learning_rate,
            max_bins=args.max_bins,
            l2_regularization=args.l2_regularization,
            early_stopping=True,
            random_state=args.seed,
            verbose=0,
            max_iter=args.max_iter,
            max_leaf_nodes=args.max_leaf_nodes,
            min_samples_leaf=args.min_samples_leaf,
        ),
        use_isotonic=bool(args.use_isotonic),
        use_isotonic_cv=not args.use_isotonic,
        iso_cv_folds=5,
        iso_cv_seed=0,
        use_monotone_for_delta_tokens=bool(args.use_monotone_for_delta_tokens),
    )
    FEAT = FeatCfg(include_delta_tokens=bool(args.include_delta_tokens))
    SELECT = SelectCfg(
        rates_grid=np.linspace(0, 1, 101),
        utopia_weights=(1.0, 1.0),
        epsilon_target=args.epsilon_target,
        epsilon_delta=args.epsilon_delta,
    )
    OUT = OutCfg(verbose=True)

    print(f"[INFO] use_genre={args.use_genre}, use_prompt={args.use_prompt}")
    bundle = load_experiment_bundle(PATHS, EXP, use_genre=args.use_genre, use_prompt=args.use_prompt)

    train_split, dev_split = split_train_dev(bundle["train"], dev_ratio=SPLIT.dev_ratio, seed=SPLIT.seed)
    test_split = bundle["test"]

    if FEAT.include_delta_tokens:
        for s in (train_split, dev_split, test_split):
            s["X"]["delta_tokens_mean"] = s["cot_token_mean"] - s["direct_token_mean"]

    default_sel_json = os.path.join(PATHS.save_dir, "selected_features.json")
    selected_cols = _load_selected_cols(default_sel_json, args.selected_json, list(train_split["X"].columns))
    if FEAT.include_delta_tokens and "delta_tokens_mean" not in selected_cols:
        selected_cols.append("delta_tokens_mean")

    for s in (train_split, dev_split, test_split):
        s["X"] = s["X"][selected_cols]

    # Advantage label building
    A_tr, w_tr, _ = build_advantage_labels_from_split(train_split, **asdict(LABEL))
    A_dev, _, _ = build_advantage_labels_from_split(dev_split, **asdict(LABEL))

    X_tr = train_split["X"].to_numpy(float)
    X_dev = dev_split["X"].to_numpy(float)

    thresholds, label_gain = _make_rank_thresholds(A_tr, eps=LABEL.eps_equal)
    y_tr = _assign_rank_labels(A_tr, thresholds, eps=LABEL.eps_equal)
    y_dev = _assign_rank_labels(A_dev, thresholds, eps=LABEL.eps_equal)

    mono = None
    if MODEL.use_monotone_for_delta_tokens and "delta_tokens_mean" in train_split["X"].columns:
        mono = np.zeros(X_tr.shape[1], int)
        mono[list(train_split["X"].columns).index("delta_tokens_mean")] = -1

    ranker = train_ranker_lgbm(
        X_tr, y_tr, sample_weight=w_tr, monotone_cst=mono,
        X_dev=X_dev, A_dev=y_dev, random_state=SPLIT.seed, label_gain=label_gain,
    )

    # Isotonic calibration
    train_pred = ranker.predict(X_tr)
    calibrator = IsotonicRegression(out_of_bounds="clip") if MODEL.use_isotonic_cv else None
    if calibrator:
        calibrator.fit(train_pred, A_tr)
    def predict_A(X_df): 
        A = ranker.predict(X_df.to_numpy(float))
        return calibrator.predict(A) if calibrator else A

    A_hat_dev = predict_A(dev_split["X"])
    A_hat_te = predict_A(test_split["X"])

    mu_d_dev, mu_c_dev = dev_split["direct_ndcg_mean"], dev_split["cot_ndcg_mean"]
    td_dev, tc_dev = dev_split["direct_token_mean"], dev_split["cot_token_mean"]
    mu_d_te, mu_c_te = test_split["direct_ndcg_mean"], test_split["cot_ndcg_mean"]
    td_te, tc_te = test_split["direct_token_mean"], test_split["cot_token_mean"]

    base_dev = baselines(mu_d_dev, mu_c_dev, td_dev, tc_dev, lambda_cost=LABEL.lambda_cost)
    cot_ndcg, direct_ndcg = base_cot_vs_direct(mu_d_dev, mu_c_dev, td_dev, tc_dev, LABEL.lambda_cost)
    cot_is_better = cot_ndcg >= direct_ndcg
    pos_mode = "cot" if cot_is_better else "direct"

    A_pos = A_hat_dev if cot_is_better else -A_hat_dev
    mu_pos, mu_neg = (mu_c_dev, mu_d_dev) if cot_is_better else (mu_d_dev, mu_c_dev)
    t_pos, t_neg = (tc_dev, td_dev) if cot_is_better else (td_dev, tc_dev)

    topr = best_topr_policy(A_pos, mu_pos, mu_neg, t_pos, t_neg, SELECT.rates_grid, LABEL.lambda_cost)
    best_r, tau_topr = topr["r"], topr["tau"]
    curve_dev = topr["curve"]

    policy_topr = {"name": "top_r_tau", "r": best_r, "tau": tau_topr, "positive_mode": pos_mode}
    front = _pareto_front(curve_dev)
    knee_point = select_knee(front)
    utopia_point = select_utopia(front, *SELECT.utopia_weights)

    best_base = max(base_dev["Always-CoT"]["NDCG"], base_dev["Always-Direct"]["NDCG"])
    eps_target = best_base + SELECT.epsilon_delta if SELECT.epsilon_target == "delta" else best_base
    eps_point = select_epsilon(front, eps_target)

    def policy_from_point(pt, name="budget_topr_point"):
        return {"name": name, "rate": pt["rate"], "target_mean_tokens": pt["mean_tokens"], "positive_mode": pos_mode}

    policy_knee = policy_from_point(knee_point)
    policy_utopia = policy_from_point(utopia_point)
    policy_eps = policy_from_point(eps_point) if eps_point else None

    # Evaluate on dev/test
    dev_reports, test_reports = {}, {}
    for label, pol in {
        "Top-r": policy_topr,
        "Knee": policy_knee,
        "Utopia": policy_utopia,
        **({"Epsilon": policy_eps} if policy_eps else {}),
    }.items():
        dev_pred = apply_policy_symmetric(A_hat_dev, td_dev, tc_dev, pol)
        test_pred = apply_policy_symmetric(A_hat_te, td_te, tc_te, pol)
        dev_reports[label] = eval_mixed_strategy(dev_pred, mu_d_dev, mu_c_dev, td_dev, tc_dev, LABEL.lambda_cost)
        test_reports[label] = eval_mixed_strategy(test_pred, mu_d_te, mu_c_te, td_te, tc_te, LABEL.lambda_cost)

    summary = {
        "config": asdict(EXP),
        "dev": {"baselines": base_dev, "reports": dev_reports},
        "test": {"reports": test_reports},
        "policies": {"topr": policy_topr, "knee": policy_knee, "utopia": policy_utopia, "epsilon": policy_eps},
    }

    os.makedirs(PATHS.save_dir, exist_ok=True)
    out_summary = os.path.join(PATHS.save_dir, "regression_summary.json")
    with open(out_summary, "w") as f:
        json.dump(summary, f, indent=2, default=_np_to_py)
    print(f"[Saved] {out_summary}")

    exp_num = args.exp_version.split("_v")[-1]
    if args.dataset == "rank_ir":
        selected_metric = ["avg_top1", "avg_pairacc", "avg_ndcg@5", "avg_ndcg@10"]
    else:
        selected_metric = ["avg_recall@5", "avg_ndcg@5", "avg_recall@10", "avg_ndcg@10"]
    all_mean = evaluate_test_result(PATHS, EXP, bundle, selected_metric, num=exp_num)
    all_mean.to_csv(os.path.join(PATHS.save_dir, "selected_result_test_mean.csv"), index=True)
    print("[Done]")


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    main(args)
