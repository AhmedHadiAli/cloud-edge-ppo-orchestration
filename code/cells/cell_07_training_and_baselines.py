def train_ppo(
    params: SystemParams,
    scenario: str,
    episodes: int = 120,
    seed: int = 42,
    ablation_mode: Optional[str] = None,
    fixed_weights: Optional[Tuple[float, float, float]] = None,
) -> Tuple[PPOAgent, pd.DataFrame]:
    probe = CloudEdgeEnv(params, scenario, seed, ablation_mode, fixed_weights)
    agent = PPOAgent(probe.state_dim, probe.action_dim, seed=seed)
    history = []
    best_validation_reward = -float("inf")
    best_state = copy.deepcopy(agent.policy.state_dict())
    validation_seeds = (seed + 10000, seed + 10001)
    for episode in range(episodes):
        env = CloudEdgeEnv(params, scenario, seed + episode, ablation_mode, fixed_weights)
        state = env.reset(seed + episode)
        rollout = []
        episode_return = 0.0
        while True:
            action, log_prob, value = agent.select_action(state, deterministic=False)
            next_state, reward, done, _ = env.step(action)
            rollout.append((state, action, reward, done, log_prob, value))
            episode_return += reward
            state = next_state
            if done:
                break
        update_info = agent.update(rollout)
        row = {"episode": episode + 1, "return": episode_return, **env.summary(), **update_info}
        history.append(row)
        if (episode + 1) % 5 == 0 or episode == episodes - 1:
            validation_rewards = []
            for validation_seed in validation_seeds:
                validation_env = CloudEdgeEnv(params, scenario, validation_seed, ablation_mode, fixed_weights)
                validation_state = validation_env.reset(validation_seed)
                while True:
                    validation_action, _, _ = agent.select_action(validation_state, deterministic=True)
                    validation_state, _, validation_done, _ = validation_env.step(validation_action)
                    if validation_done:
                        break
                validation_rewards.append(validation_env.summary()["mean_reward"])
            validation_score = float(np.mean(validation_rewards))
            history[-1]["validation_reward"] = validation_score
            if validation_score > best_validation_reward:
                best_validation_reward = validation_score
                best_state = copy.deepcopy(agent.policy.state_dict())
    agent.policy.load_state_dict(best_state)
    return agent, pd.DataFrame(history)


def choose_baseline_action(env: CloudEdgeEnv, policy_name: str) -> int:
    names = [a.name for a in env.actions]
    if policy_name == "Local-only":
        return names.index("local-fp32")
    if policy_name == "Cloud-only":
        return names.index("cloud-full-fp32")
    if policy_name == "Edge-only":
        candidates = [i for i, a in enumerate(env.actions) if a.target_type == "edge" and a.partition_fraction == 0.0 and a.quant_bits == 32]
        return min(candidates, key=lambda i: env.estimate_action(i)["latency_s"])
    if policy_name == "Static":
        assert env.current_task is not None
        _, bw_factor, _ = env._scenario_factors(env.current_task["arrival_time"])
        if env.current_task["total_gflops"] > 3.4 and bw_factor >= 0.8:
            return names.index("cloud-full-fp32")
        return names.index("local-fp32")
    if policy_name == "Heuristic":
        weights = (0.40, 0.30, 0.30)
        scores = []
        for i in range(env.action_dim):
            estimate = env.estimate_action(i)
            assert env.current_task is not None
            # This baseline uses fixed offline service-cost estimates and does not
            # react to the current queue backlog. The distinction makes it a valid
            # non-adaptive comparator rather than an oracle with full state access.
            service_latency = max(estimate["latency_s"] - estimate["queue_wait_s"], 0.0)
            latency_ratio = min(service_latency / env.current_task["deadline_s"], 4.0)
            energy_norm = min(estimate["device_energy_j"] / 2.0, 4.0)
            quality_loss = (1.0 - estimate["accuracy"]) * 10.0
            score = weights[0] * latency_ratio + weights[1] * energy_norm + weights[2] * quality_loss + 0.75 * estimate["violation"] + 1.25 * estimate["infeasible_energy"]
            scores.append(score)
        return int(np.argmin(scores))
    raise ValueError(f"Unknown baseline policy: {policy_name}")


def evaluate_policy(
    params: SystemParams,
    scenario: str,
    seed: int,
    policy_name: str,
    agent: Optional[PPOAgent] = None,
    ablation_mode: Optional[str] = None,
    fixed_weights: Optional[Tuple[float, float, float]] = None,
    return_trace: bool = False,
) -> Tuple[Dict[str, float], Optional[pd.DataFrame]]:
    env = CloudEdgeEnv(params, scenario, seed, ablation_mode, fixed_weights)
    state = env.reset(seed)
    while True:
        if policy_name == "PPO":
            if agent is None:
                raise ValueError("agent is required for PPO evaluation")
            action, _, _ = agent.select_action(state, deterministic=True)
        else:
            action = choose_baseline_action(env, policy_name)
        state, _, done, _ = env.step(action)
        if done:
            break
    trace = pd.DataFrame(env.metrics) if return_trace else None
    return env.summary(), trace


def run_multi_seed_evaluation(
    params: SystemParams,
    agents: Dict[str, PPOAgent],
    scenarios: Sequence[str],
    seeds: Sequence[int],
) -> pd.DataFrame:
    policies = ["Local-only", "Edge-only", "Cloud-only", "Static", "Heuristic", "PPO"]
    rows = []
    for scenario in scenarios:
        for seed in seeds:
            for policy in policies:
                metrics, _ = evaluate_policy(params, scenario, seed, policy, agents.get(scenario))
                rows.append({"scenario": scenario, "seed": seed, "policy": policy, **metrics})
    return pd.DataFrame(rows)


