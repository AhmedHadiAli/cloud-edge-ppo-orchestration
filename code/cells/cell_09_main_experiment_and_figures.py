def save_figures(raw: pd.DataFrame, summary: pd.DataFrame, agents: Dict[str, PPOAgent], params: SystemParams, output_dir: Path, ablation_raw: pd.DataFrame) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    scenarios = ["normal", "burst", "fluctuation", "energy_constrained"]
    policies = ["Local-only", "Edge-only", "Cloud-only", "Static", "Heuristic", "PPO"]

    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(scenarios))
    width = 0.13
    for i, policy in enumerate(policies):
        means = []
        stds = []
        for scenario in scenarios:
            subset = raw[(raw.scenario == scenario) & (raw.policy == policy)]["latency_ms"]
            means.append(subset.mean())
            stds.append(subset.std())
        ax.bar(x + (i - 2.5) * width, means, width, yerr=stds, capsize=2, label=policy)
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("_", " ").title() for s in scenarios])
    ax.set_ylabel("Average end-to-end latency (ms)")
    ax.set_title("Latency across operating scenarios (mean ± SD, 10 seeds)")
    ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "figure_1_latency.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 6))
    normal = raw[raw.scenario == "normal"].groupby("policy").mean(numeric_only=True)
    label_offsets = {
        "Local-only": (4, 5), "Edge-only": (-5, -14), "Cloud-only": (5, -13),
        "Static": (4, 5), "Heuristic": (-5, 6), "PPO": (5, 8),
    }
    for policy in policies:
        ax.scatter(normal.loc[policy, "device_energy_j"], normal.loc[policy, "latency_ms"], s=80)
        ax.annotate(policy, (normal.loc[policy, "device_energy_j"], normal.loc[policy, "latency_ms"]), xytext=label_offsets[policy], textcoords="offset points", fontsize=8)
    ax.set_xlabel("Device-side energy per task (J)")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Latency-energy trade-off in the normal scenario")
    fig.tight_layout()
    fig.savefig(output_dir / "figure_2_tradeoff.png", dpi=220)
    plt.close(fig)

    trace_params = replace(params, max_tasks=500)
    _, trace = evaluate_policy(trace_params, "fluctuation", 777, "PPO", agents["fluctuation"], return_trace=True)
    assert trace is not None
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()
    ax1.plot(trace["arrival_time_s"], trace["partition_layer"], linewidth=1.3, label="Partition layer")
    ax2.plot(trace["arrival_time_s"], trace["bandwidth_mbps"], linestyle="--", linewidth=1.2, label="Selected-link bandwidth")
    ax1.set_xlabel("Arrival time (s)")
    ax1.set_ylabel("Partition layer")
    ax2.set_ylabel("Bandwidth (Mbps)")
    ax1.set_title("PPO decisions under deterministic network fluctuation")
    fig.tight_layout()
    fig.savefig(output_dir / "figure_3_partition.png", dpi=220)
    plt.close(fig)

    ablation_summary = ablation_raw.groupby("variant")[["latency_ms", "device_energy_j", "accuracy", "violation_rate"]].mean()
    normalized = ablation_summary.copy()
    normalized["latency_ms"] = ablation_summary["latency_ms"].min() / ablation_summary["latency_ms"]
    normalized["device_energy_j"] = ablation_summary["device_energy_j"].min() / ablation_summary["device_energy_j"]
    normalized["accuracy"] = ablation_summary["accuracy"] / ablation_summary["accuracy"].max()
    normalized["violation_rate"] = (1.0 - ablation_summary["violation_rate"]) / max(1.0 - ablation_summary["violation_rate"].min(), 1e-8)
    fig, ax = plt.subplots(figsize=(10, 6))
    normalized.plot(kind="bar", ax=ax)
    ax.set_ylabel("Normalized score (higher is better)")
    ax.set_title("Ablation analysis in the normal scenario")
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(output_dir / "figure_4_ablation.png", dpi=220)
    plt.close(fig)


def run_complete_experiment(output_dir: str | Path, quick: bool = False) -> Dict[str, str]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    params = SystemParams(max_tasks=90 if quick else 120)
    scenarios = ["normal", "burst", "fluctuation", "energy_constrained"]
    training_episodes = 35 if quick else 100
    ablation_episodes = 20 if quick else 30
    evaluation_seeds = list(range(1000, 1005 if quick else 1010))
    training_seed_bases = {scenario: 100 + index * 200 for index, scenario in enumerate(scenarios)}
    ablation_training_seed_bases = {
        "Full PPO": 2000,
        "Fixed weights": 2200,
        "No partitioning": 2400,
        "No quantization": 2600,
    }

    agents: Dict[str, PPOAgent] = {}
    histories = []
    for index, scenario in enumerate(scenarios):
        agent, history = train_ppo(params, scenario, episodes=training_episodes, seed=training_seed_bases[scenario])
        agents[scenario] = agent
        history.insert(0, "scenario", scenario)
        histories.append(history)
        torch.save(agent.policy.state_dict(), output / f"ppo_{scenario}.pt")

    training_history = pd.concat(histories, ignore_index=True)
    raw = run_multi_seed_evaluation(params, agents, scenarios, evaluation_seeds)
    summary = summarize_results(raw)
    stats = paired_statistics(raw)
    ablation_raw, _ = run_ablation(params, evaluation_seeds, episodes=ablation_episodes)
    ablation_summary = ablation_raw.groupby("variant")[["latency_ms", "device_energy_j", "accuracy", "violation_rate", "utilization", "depletion"]].agg(["mean", "std"])

    training_history.to_csv(output / "training_history.csv", index=False)
    raw.to_csv(output / "results_raw.csv", index=False)
    summary.to_csv(output / "results_summary.csv", index=False)
    stats.to_csv(output / "paired_statistics.csv", index=False)
    ablation_raw.to_csv(output / "ablation_raw.csv", index=False)
    ablation_summary.to_csv(output / "ablation_summary.csv")
    save_figures(raw, summary, agents, params, output, ablation_raw)
    with open(output / "experiment_config.json", "w", encoding="utf-8") as handle:
        json.dump({
            "params": asdict(params),
            "training_episodes": training_episodes,
            "ablation_episodes": ablation_episodes,
            "training_seed_bases": training_seed_bases,
            "ablation_training_seed_bases": ablation_training_seed_bases,
            "validation_seed_offsets": [10000, 10001],
            "evaluation_seeds": evaluation_seeds,
        }, handle, indent=2)
    return {
        "training_history": str(output / "training_history.csv"),
        "raw": str(output / "results_raw.csv"),
        "summary": str(output / "results_summary.csv"),
        "statistics": str(output / "paired_statistics.csv"),
        "ablation": str(output / "ablation_summary.csv"),
    }


if __name__ == "__main__":
    paths = run_complete_experiment(Path(__file__).parent / "outputs", quick=False)
    print(json.dumps(paths, indent=2))
