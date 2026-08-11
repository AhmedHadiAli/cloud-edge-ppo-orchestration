def summarize_results(raw: pd.DataFrame) -> pd.DataFrame:
    metrics = ["latency_ms", "device_energy_j", "accuracy", "violation_rate", "utilization", "depletion"]
    grouped = raw.groupby(["scenario", "policy"])[metrics].agg(["mean", "std"]).reset_index()
    grouped.columns = ["_".join([x for x in col if x]).rstrip("_") for col in grouped.columns.to_flat_index()]
    return grouped


def paired_statistics(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metrics = ["latency_ms", "device_energy_j", "violation_rate"]
    for scenario in raw["scenario"].unique():
        ppo = raw[(raw.scenario == scenario) & (raw.policy == "PPO")].sort_values("seed")
        for baseline in ["Local-only", "Edge-only", "Cloud-only", "Static", "Heuristic"]:
            base = raw[(raw.scenario == scenario) & (raw.policy == baseline)].sort_values("seed")
            for metric in metrics:
                try:
                    statistic, p_value = wilcoxon(ppo[metric].to_numpy(), base[metric].to_numpy(), zero_method="zsplit")
                except ValueError:
                    statistic, p_value = np.nan, 1.0
                rows.append({"scenario": scenario, "baseline": baseline, "metric": metric, "statistic": statistic, "p_value": p_value})
    result = pd.DataFrame(rows)
    # Holm correction is applied within each baseline across the four scenarios
    # and three reported metrics (12 paired tests per comparator).
    adjusted_parts = []
    for baseline, group in result.groupby("baseline", sort=False):
        group = group.copy()
        pvals = group["p_value"].to_numpy(dtype=float)
        order = np.argsort(pvals)
        adjusted = np.empty_like(pvals)
        running = 0.0
        m = len(pvals)
        for rank, idx in enumerate(order):
            running = max(running, (m - rank) * pvals[idx])
            adjusted[idx] = min(running, 1.0)
        group["p_holm"] = adjusted
        adjusted_parts.append(group)
    return pd.concat(adjusted_parts, ignore_index=True)


def run_ablation(params: SystemParams, seeds: Sequence[int], episodes: int = 90) -> Tuple[pd.DataFrame, Dict[str, PPOAgent]]:
    variants = {
        "Full PPO": (None, None),
        "Fixed weights": ("no_adaptive", None),
        "No partitioning": ("no_partition", None),
        "No quantization": ("no_quant", None),
    }
    rows = []
    agents: Dict[str, PPOAgent] = {}
    for index, (name, (mode, fixed)) in enumerate(variants.items()):
        agent, _ = train_ppo(params, "normal", episodes=episodes, seed=2000 + index * 200, ablation_mode=mode, fixed_weights=fixed)
        agents[name] = agent
        for seed in seeds:
            metrics, _ = evaluate_policy(params, "normal", seed, "PPO", agent, ablation_mode=mode, fixed_weights=fixed)
            rows.append({"variant": name, "seed": seed, **metrics})
    return pd.DataFrame(rows), agents


