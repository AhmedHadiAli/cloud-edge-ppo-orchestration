"""Reviewer-driven revision of the cloud-edge orchestration experiment.

Key corrections relative to the originally submitted artifact:
- one unified PPO policy is trained across mixed operating conditions;
- the environment runs for 45 s, covering a complete 40 s fluctuation cycle and a
  complete 30 s burst/recovery cycle;
- resource capacities are re-provisioned so aggregate tier capacities are within
  roughly a factor of two;
- transmit energy is charged only during serialization, never propagation;
- the undocumented reduced-precision compute speed-up is removed;
- the nine-action space is retained but reduced precision is used only for
  partitioned FP16 activation transfer;
- a validation-selected exhaustive constant-action baseline is included;
- a DQN baseline is trained under the same mixed-scenario RL interaction budget;
- the PPO actor is warm-started by behavior cloning from the transparent
  one-step greedy policy and then PPO-fine-tuned; the retained checkpoint must
  occur after PPO updates;
- policy entropy, deterministic/stochastic PPO evaluation, action diversity,
  scenario coverage, sensitivity analysis, effect sizes and confidence
  intervals are exported;
- all generated task/network randomness is seed-controlled and policy-independent.

The study remains a simulation study. A separate hardware_microbenchmark.py file
executes real PyTorch ResNet-50 and MobileNetV2 inference on the available CPU as
an external timing sanity check; it is not treated as deployment validation or as
an energy measurement.
"""
from __future__ import annotations

import copy
import json
import math
import platform
import random
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import rankdata, wilcoxon
from torch.distributions import Categorical

torch.set_num_threads(1)


# ------------------------------- Reproducibility -------------------------------

def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ------------------------------- Model profile --------------------------------

@dataclass(frozen=True)
class ModelProfile:
    name: str
    base_gflops: float
    base_retention: float
    layer_fractions: Tuple[float, ...]
    activation_mb: Tuple[float, ...]

    @property
    def num_layers(self) -> int:
        return len(self.layer_fractions)


RESNET50_PROFILE = ModelProfile(
    name="ResNet-50 analytical 12-stage profile",
    base_gflops=4.10,
    base_retention=0.935,
    layer_fractions=(0.04, 0.05, 0.07, 0.08, 0.10, 0.10, 0.11, 0.11, 0.10, 0.09, 0.08, 0.07),
    activation_mb=(2.50, 1.80, 1.20, 0.85, 0.55, 0.40, 0.30, 0.22, 0.16, 0.12, 0.08, 0.02),
)


# ---------------------------------- System -------------------------------------

@dataclass
class SystemParams:
    num_devices: int = 4
    num_edge_servers: int = 3
    device_caps_gflops: Tuple[float, ...] = (20.0, 24.0, 28.0, 32.0)
    edge_caps_gflops: Tuple[float, ...] = (36.0, 48.0, 60.0)
    cloud_cap_gflops: float = 120.0
    device_power_active_w: float = 3.5
    device_tx_power_w: float = 1.4
    nominal_battery_j: float = 120.0
    energy_scenario_budgets_j: Tuple[float, ...] = (24.0, 28.0, 32.0, 36.0)
    bw_device_edge_mbps: float = 90.0
    bw_device_cloud_mbps: float = 75.0
    latency_device_edge_ms: float = 12.0
    latency_device_cloud_ms: float = 42.0
    arrival_rate_tasks_s: float = 16.0
    horizon_s: float = 45.0
    input_size_mean_mb: float = 0.75
    input_size_std_mb: float = 0.20
    complexity_mean: float = 0.90
    complexity_std: float = 0.18
    deadline_mean_ms: float = 250.0
    deadline_std_ms: float = 65.0
    network_bw_jitter_sd: float = 0.08
    network_latency_jitter_sd: float = 0.05
    model_name: str = "resnet50"

    def validate(self) -> None:
        if self.num_devices != 4 or self.num_edge_servers != 3:
            raise ValueError("This released action/state definition expects 4 devices and 3 edge servers.")
        if self.horizon_s < 40.0:
            raise ValueError("horizon_s must be at least 40 s to cover all fluctuation phases.")


@dataclass(frozen=True)
class ActionSpec:
    name: str
    target: str
    target_index: Optional[int]
    partition_fraction: float
    quant_bits: int


ACTIONS: Tuple[ActionSpec, ...] = (
    ActionSpec("local-fp32", "local", None, 1.00, 32),
    ActionSpec("edge-1-full-fp32", "edge", 0, 0.00, 32),
    ActionSpec("edge-1-split-fp16", "edge", 0, 0.45, 16),
    ActionSpec("edge-2-full-fp32", "edge", 1, 0.00, 32),
    ActionSpec("edge-2-split-fp16", "edge", 1, 0.45, 16),
    ActionSpec("edge-3-full-fp32", "edge", 2, 0.00, 32),
    ActionSpec("edge-3-split-fp16", "edge", 2, 0.45, 16),
    ActionSpec("cloud-full-fp32", "cloud", None, 0.00, 32),
    ActionSpec("cloud-split-fp16", "cloud", None, 0.65, 16),
)

SCENARIOS = ("normal", "burst", "fluctuation", "energy")


class CloudEdgeEnv:
    """Event-based finite-capacity device-edge-cloud simulator."""

    state_dim: int = 17
    action_dim: int = 9

    def __init__(
        self,
        params: SystemParams,
        scenario: str,
        seed: int,
        domain_randomization: bool = False,
        parameter_overrides: Optional[Dict[str, float]] = None,
    ) -> None:
        params.validate()
        if scenario not in SCENARIOS:
            raise ValueError(f"Unknown scenario: {scenario}")
        self.params = copy.deepcopy(params)
        self.scenario = scenario
        self.seed = seed
        self.domain_randomization = domain_randomization
        self.parameter_overrides = parameter_overrides or {}
        self.rng = np.random.default_rng(seed)
        self._sample_episode_scales()
        self.reset(seed)

    @property
    def profile(self) -> ModelProfile:
        return RESNET50_PROFILE

    @property
    def actions(self) -> Tuple[ActionSpec, ...]:
        return ACTIONS

    def _sample_episode_scales(self) -> None:
        if self.domain_randomization:
            self.device_scale = float(self.rng.uniform(0.90, 1.10))
            self.edge_scale = float(self.rng.uniform(0.88, 1.12))
            self.cloud_scale = float(self.rng.uniform(0.88, 1.12))
            self.bandwidth_scale = float(self.rng.uniform(0.85, 1.15))
            self.latency_scale = float(self.rng.uniform(0.90, 1.10))
            self.arrival_scale = float(self.rng.uniform(0.90, 1.10))
        else:
            self.device_scale = self.edge_scale = self.cloud_scale = 1.0
            self.bandwidth_scale = self.latency_scale = self.arrival_scale = 1.0
        self.bandwidth_scale *= float(self.parameter_overrides.get("bandwidth_scale", 1.0))
        self.device_scale *= float(self.parameter_overrides.get("capacity_scale", 1.0))
        self.edge_scale *= float(self.parameter_overrides.get("capacity_scale", 1.0))
        self.cloud_scale *= float(self.parameter_overrides.get("capacity_scale", 1.0))
        self.arrival_scale *= float(self.parameter_overrides.get("arrival_scale", 1.0))

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        if seed is not None:
            self.seed = int(seed)
            self.rng = np.random.default_rng(self.seed)
            self._sample_episode_scales()
        self.time_s = 0.0
        self.task_count = 0
        self.device_busy_until = np.zeros(self.params.num_devices, dtype=float)
        self.edge_busy_until = np.zeros(self.params.num_edge_servers, dtype=float)
        self.cloud_busy_until = 0.0
        self.recent_violations: List[int] = []
        if self.scenario == "energy":
            budgets = np.asarray(self.params.energy_scenario_budgets_j, dtype=float)
            # Random permutation varies which device receives which budget while preserving
            # the same set of budgets for every policy under a matched seed.
            self.initial_battery = self.rng.permutation(budgets).astype(float)
        else:
            self.initial_battery = np.repeat(self.params.nominal_battery_j, self.params.num_devices).astype(float)
        self.battery = self.initial_battery.copy()
        self.metrics: Dict[str, List[float]] = {
            "latency_ms": [], "device_energy_j": [], "retention": [],
            "violation": [], "drop": [], "reward": [], "action": [],
            "arrival_time_s": [], "edge_bandwidth_mbps": [], "cloud_bandwidth_mbps": [],
            "fluctuation_phase": [], "burst_window": [],
        }
        self.current_task = self._generate_task(first=True)
        return self._get_state() if self.current_task is not None else np.zeros(self.state_dim, np.float32)

    def _scenario_factors(self, time_s: float) -> Tuple[float, float, float, float, float, int, int]:
        arrival_factor = 1.0
        edge_bw, edge_lat, cloud_bw, cloud_lat = 1.0, 1.0, 1.0, 1.0
        phase = -1
        burst_window = 0
        if self.scenario == "burst":
            burst_window = int((time_s % 30.0) < 10.0)
            arrival_factor = 2.5 if burst_window else 0.75
        elif self.scenario == "fluctuation":
            phase = int(time_s // 10.0) % 4
            edge_bw = (0.55, 0.85, 1.25, 1.60)[phase]
            edge_lat = (1.80, 1.30, 0.85, 0.65)[phase]
            cloud_bw = (0.45, 0.75, 1.35, 1.70)[phase]
            cloud_lat = (1.65, 1.25, 0.80, 0.60)[phase]
        return arrival_factor, edge_bw, edge_lat, cloud_bw, cloud_lat, phase, burst_window

    def _generate_task(self, first: bool = False) -> Optional[Dict[str, float]]:
        arrival_factor, edge_bw, edge_lat, cloud_bw, cloud_lat, phase, burst_window = self._scenario_factors(self.time_s)
        rate = self.params.arrival_rate_tasks_s * self.arrival_scale * arrival_factor
        self.time_s += 0.0 if first else float(self.rng.exponential(1.0 / rate))
        if self.time_s > self.params.horizon_s:
            return None
        # Recompute deterministic phase factors at the actual arrival time.
        arrival_factor, edge_bw, edge_lat, cloud_bw, cloud_lat, phase, burst_window = self._scenario_factors(self.time_s)
        source = self.task_count % self.params.num_devices
        self.task_count += 1
        input_size = float(np.clip(self.rng.normal(self.params.input_size_mean_mb, self.params.input_size_std_mb), 0.20, 1.60))
        complexity = float(np.clip(self.rng.normal(self.params.complexity_mean, self.params.complexity_std), 0.50, 1.35))
        total_gflops = self.profile.base_gflops * complexity
        deadline_s = float(np.clip(self.rng.normal(self.params.deadline_mean_ms, self.params.deadline_std_ms), 120.0, 600.0)) / 1000.0
        # Seed-controlled per-task link jitter ensures evaluation seeds vary both task streams
        # and network conditions, while all policies see the same realization.
        e_bw_j = float(np.clip(self.rng.lognormal(mean=0.0, sigma=self.params.network_bw_jitter_sd), 0.75, 1.30))
        c_bw_j = float(np.clip(self.rng.lognormal(mean=0.0, sigma=self.params.network_bw_jitter_sd), 0.75, 1.30))
        e_lat_j = float(np.clip(self.rng.normal(1.0, self.params.network_latency_jitter_sd), 0.85, 1.20))
        c_lat_j = float(np.clip(self.rng.normal(1.0, self.params.network_latency_jitter_sd), 0.85, 1.20))
        return {
            "arrival": self.time_s, "source": float(source), "input_mb": input_size,
            "complexity": complexity, "gflops": total_gflops, "deadline_s": deadline_s,
            "edge_bw_factor": edge_bw * e_bw_j, "edge_latency_factor": edge_lat * e_lat_j,
            "cloud_bw_factor": cloud_bw * c_bw_j, "cloud_latency_factor": cloud_lat * c_lat_j,
            "phase": float(phase), "burst_window": float(burst_window),
        }

    def _get_state(self) -> np.ndarray:
        if self.current_task is None:
            return np.zeros(self.state_dim, dtype=np.float32)
        q = self.current_task
        arrival = q["arrival"]
        source = int(q["source"])
        source_wait = max(0.0, self.device_busy_until[source] - arrival)
        edge_waits = [max(0.0, x - arrival) for x in self.edge_busy_until]
        cloud_wait = max(0.0, self.cloud_busy_until - arrival)
        source_battery = self.battery[source] / max(self.initial_battery[source], 1e-9)
        mean_battery = float(np.mean(self.battery / np.maximum(self.initial_battery, 1e-9)))
        recent_violation = float(np.mean(self.recent_violations[-30:])) if self.recent_violations else 0.0
        phase = 2.0 * math.pi * ((arrival % 40.0) / 40.0)
        values = [
            q["input_mb"] / 1.60,
            q["gflops"] / (self.profile.base_gflops * 1.35),
            q["deadline_s"] / 0.60,
            min(source_wait / 1.0, 4.0),
            *[min(x / 1.0, 4.0) for x in edge_waits],
            min(cloud_wait / 1.0, 4.0),
            q["edge_bw_factor"] / 1.80,
            q["edge_latency_factor"] / 2.00,
            q["cloud_bw_factor"] / 1.80,
            q["cloud_latency_factor"] / 1.80,
            source_battery,
            mean_battery,
            recent_violation,
            math.sin(phase),
            math.cos(phase),
        ]
        if len(values) != self.state_dim:
            raise AssertionError(f"Expected {self.state_dim} state elements, got {len(values)}")
        return np.asarray(values, dtype=np.float32)

    def estimate_action(self, action_index: int) -> Dict[str, float]:
        if self.current_task is None:
            raise RuntimeError("No current task")
        a = self.actions[action_index]
        q = self.current_task
        arrival = q["arrival"]
        source = int(q["source"])
        n = self.profile.num_layers
        partition = int(np.clip(round(a.partition_fraction * n), 0, n))
        layer_gflops = np.asarray(self.profile.layer_fractions) * q["gflops"]
        local_gflops = float(layer_gflops[:partition].sum())
        remote_gflops = float(layer_gflops[partition:].sum())
        device_cap = self.params.device_caps_gflops[source] * self.device_scale
        local_start = max(arrival, self.device_busy_until[source]) if local_gflops > 0 else arrival
        local_exec = local_gflops / device_cap if local_gflops > 0 else 0.0
        local_finish = local_start + local_exec
        serialization = propagation = remote_exec = 0.0
        finish = local_finish
        tx_size = 0.0
        if a.target != "local":
            if partition == 0:
                tx_size = q["input_mb"]
            else:
                tx_size = self.profile.activation_mb[partition - 1] * q["complexity"] * (a.quant_bits / 32.0)
            if a.target == "edge":
                bw = self.params.bw_device_edge_mbps * self.bandwidth_scale * q["edge_bw_factor"]
                propagation = self.params.latency_device_edge_ms * self.latency_scale * q["edge_latency_factor"] / 1000.0
                remote_cap = self.params.edge_caps_gflops[a.target_index or 0] * self.edge_scale
                busy_until = self.edge_busy_until[a.target_index or 0]
            else:
                bw = self.params.bw_device_cloud_mbps * self.bandwidth_scale * q["cloud_bw_factor"]
                propagation = self.params.latency_device_cloud_ms * self.latency_scale * q["cloud_latency_factor"] / 1000.0
                remote_cap = self.params.cloud_cap_gflops * self.cloud_scale
                busy_until = self.cloud_busy_until
            serialization = tx_size * 8.0 / max(bw, 0.1)
            remote_exec = remote_gflops / remote_cap
            remote_start = max(local_finish + serialization + propagation, busy_until)
            finish = remote_start + remote_exec
        latency = finish - arrival
        # Reviewer correction M14: propagation does not consume transmit energy.
        energy = self.params.device_power_active_w * local_exec + self.params.device_tx_power_w * serialization
        retention = self.profile.base_retention - (0.004 if a.quant_bits == 16 else 0.0)
        infeasible = energy > self.battery[source] + 1e-12
        if infeasible:
            latency = max(latency, q["deadline_s"] * 1.5)
        violation = float(latency > q["deadline_s"] or infeasible)
        drop = float(infeasible)
        latency_term = min(latency / max(q["deadline_s"], 1e-8), 4.0)
        battery_ratio = self.battery[source] / max(self.initial_battery[source], 1e-9)
        # Explicit state-dependent energy pressure: same equation in all scenarios;
        # it becomes relevant only as the source battery is depleted.
        energy_pressure = 1.0 + 6.0 * (1.0 - battery_ratio) ** 2
        energy_term = min(energy / 0.80, 4.0) * energy_pressure
        quality_term = min((self.profile.base_retention - retention) / 0.02, 1.0)
        reward = -(
            0.55 * latency_term + 0.25 * energy_term + 0.20 * quality_term
            + 1.00 * violation + 3.00 * drop
        )
        return {
            "partition": float(partition), "latency_s": latency,
            "device_energy_j": energy, "retention": retention,
            "violation": violation, "drop": drop, "reward": reward,
            "local_exec_s": local_exec, "local_finish_s": local_finish,
            "remote_exec_s": remote_exec, "finish_s": finish,
            "serialization_s": serialization, "propagation_s": propagation,
            "edge_bandwidth_mbps": self.params.bw_device_edge_mbps * self.bandwidth_scale * q["edge_bw_factor"],
            "cloud_bandwidth_mbps": self.params.bw_device_cloud_mbps * self.bandwidth_scale * q["cloud_bw_factor"],
        }

    def step(self, action_index: int) -> Tuple[np.ndarray, float, bool, Dict[str, float]]:
        if self.current_task is None:
            return np.zeros(self.state_dim, np.float32), 0.0, True, {}
        q = self.current_task
        a = self.actions[action_index]
        e = self.estimate_action(action_index)
        source = int(q["source"])
        available_before = self.battery[source]
        if not e["drop"]:
            if e["local_exec_s"] > 0:
                self.device_busy_until[source] = e["local_finish_s"]
            if a.target == "edge":
                self.edge_busy_until[a.target_index or 0] = e["finish_s"]
            elif a.target == "cloud":
                self.cloud_busy_until = e["finish_s"]
            self.battery[source] = max(0.0, available_before - e["device_energy_j"])
            realized_energy = e["device_energy_j"]
            realized_retention = e["retention"]
        else:
            # The task is dropped because the requested action is not feasible under
            # the remaining device-side energy budget.
            realized_energy = available_before
            self.battery[source] = 0.0
            realized_retention = np.nan
        self.recent_violations.append(int(e["violation"]))
        self.metrics["latency_ms"].append(e["latency_s"] * 1000.0)
        self.metrics["device_energy_j"].append(realized_energy)
        self.metrics["retention"].append(realized_retention)
        self.metrics["violation"].append(e["violation"])
        self.metrics["drop"].append(e["drop"])
        self.metrics["reward"].append(e["reward"])
        self.metrics["action"].append(float(action_index))
        self.metrics["arrival_time_s"].append(q["arrival"])
        self.metrics["edge_bandwidth_mbps"].append(e["edge_bandwidth_mbps"])
        self.metrics["cloud_bandwidth_mbps"].append(e["cloud_bandwidth_mbps"])
        self.metrics["fluctuation_phase"].append(q["phase"])
        self.metrics["burst_window"].append(q["burst_window"])
        self.current_task = self._generate_task()
        done = self.current_task is None
        return (np.zeros(self.state_dim, np.float32) if done else self._get_state(), float(e["reward"]), done, e)

    def summary(self) -> Dict[str, float]:
        m = self.metrics
        n = len(m["reward"])
        retention = np.asarray(m["retention"], dtype=float)
        return {
            "tasks": float(n),
            "latency_ms": float(np.mean(m["latency_ms"])) if n else np.nan,
            "device_energy_j": float(np.mean(m["device_energy_j"])) if n else np.nan,
            "retention": float(np.nanmean(retention)) if np.any(~np.isnan(retention)) else np.nan,
            "violation_rate": float(np.mean(m["violation"])) if n else np.nan,
            "drop_rate": float(np.mean(m["drop"])) if n else np.nan,
            "mean_reward": float(np.mean(m["reward"])) if n else np.nan,
            "depletion": float(1.0 - np.mean(self.battery / np.maximum(self.initial_battery, 1e-9))),
        }


# ----------------------------------- PPO ---------------------------------------

class ActorCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, 96), nn.Tanh(),
            nn.Linear(96, 96), nn.Tanh(),
        )
        self.actor = nn.Linear(96, action_dim)
        self.critic = nn.Linear(96, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(x)
        return self.actor(h), self.critic(h).squeeze(-1)


class PPOAgent:
    def __init__(self, state_dim: int, action_dim: int, seed: int) -> None:
        set_global_seed(seed)
        self.policy = ActorCritic(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=8e-5)
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.clip_ratio = 0.20
        self.entropy_coef = 2e-4
        self.value_coef = 0.50
        self.update_epochs = 4

    @torch.no_grad()
    def select_action(self, state: np.ndarray, deterministic: bool) -> Tuple[int, float, float, float]:
        x = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
        logits, value = self.policy(x)
        dist = Categorical(logits=logits)
        action = torch.argmax(logits, dim=-1) if deterministic else dist.sample()
        return int(action.item()), float(dist.log_prob(action).item()), float(value.item()), float(dist.entropy().item())

    def update(self, rollout: Sequence[Tuple[np.ndarray, int, float, bool, float, float]]) -> Dict[str, float]:
        states = torch.as_tensor(np.asarray([x[0] for x in rollout]), dtype=torch.float32)
        actions = torch.as_tensor([x[1] for x in rollout], dtype=torch.long)
        rewards = np.asarray([x[2] for x in rollout], dtype=np.float32)
        dones = np.asarray([x[3] for x in rollout], dtype=np.float32)
        old_log_probs = torch.as_tensor([x[4] for x in rollout], dtype=torch.float32)
        values_np = np.asarray([x[5] for x in rollout], dtype=np.float32)
        advantages = np.zeros_like(rewards)
        gae, next_value = 0.0, 0.0
        for t in range(len(rewards) - 1, -1, -1):
            non_terminal = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * next_value * non_terminal - values_np[t]
            gae = delta + self.gamma * self.gae_lambda * non_terminal * gae
            advantages[t] = gae
            next_value = values_np[t]
        returns = advantages + values_np
        adv = torch.as_tensor(advantages, dtype=torch.float32)
        adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
        ret = torch.as_tensor(returns, dtype=torch.float32)
        indices = np.arange(len(rollout))
        last_loss = last_entropy = 0.0
        for _ in range(self.update_epochs):
            np.random.shuffle(indices)
            for start in range(0, len(indices), 256):
                idx = indices[start:start + 256]
                logits, values = self.policy(states[idx])
                dist = Categorical(logits=logits)
                new_log_probs = dist.log_prob(actions[idx])
                ratio = torch.exp(new_log_probs - old_log_probs[idx])
                unclipped = ratio * adv[idx]
                clipped = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * adv[idx]
                policy_loss = -torch.min(unclipped, clipped).mean()
                value_loss = ((values - ret[idx]) ** 2).mean()
                entropy = dist.entropy().mean()
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.optimizer.step()
                last_loss = float(loss.detach().item())
                last_entropy = float(entropy.detach().item())
        return {"loss": last_loss, "update_entropy": last_entropy}


def greedy_myopic_action(env: CloudEdgeEnv) -> int:
    return int(np.argmax([env.estimate_action(i)["reward"] for i in range(env.action_dim)]))


def behavior_clone_warm_start(
    agent: PPOAgent,
    params: SystemParams,
    seed: int,
    n_samples: int = 20000,
    epochs: int = 20,
) -> pd.DataFrame:
    """Warm-start the actor from the transparent one-step greedy policy.

    A mixture of greedy and random behavior is used to expose the classifier to
    queue states not reached by the greedy policy alone. The critic is present
    but only the actor classification loss is optimized during this phase.
    """
    rng = np.random.default_rng(seed)
    states: List[np.ndarray] = []
    labels: List[int] = []
    while len(states) < n_samples:
        scenario = SCENARIOS[int(rng.integers(0, len(SCENARIOS)))]
        env_seed = int(rng.integers(100000, 900000))
        env = CloudEdgeEnv(params, scenario, env_seed, domain_randomization=True)
        state = env.reset(env_seed)
        while env.current_task is not None and len(states) < n_samples:
            best = greedy_myopic_action(env)
            states.append(state.copy())
            labels.append(best)
            behavior = best if rng.random() < 0.55 else int(rng.integers(0, env.action_dim))
            state, _, done, _ = env.step(behavior)
            if done:
                break
    x = torch.as_tensor(np.asarray(states), dtype=torch.float32)
    y = torch.as_tensor(labels, dtype=torch.long)
    optimizer = optim.Adam(agent.policy.parameters(), lr=1e-3)
    order = np.arange(len(y))
    rows = []
    for epoch in range(epochs):
        rng.shuffle(order)
        losses, correct, total = [], 0, 0
        for start in range(0, len(order), 256):
            idx = order[start:start + 256]
            logits, _ = agent.policy(x[idx])
            loss = nn.functional.cross_entropy(logits, y[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().item()))
            correct += int((logits.argmax(dim=-1) == y[idx]).sum().item())
            total += len(idx)
        rows.append({"epoch": epoch + 1, "loss": float(np.mean(losses)), "accuracy": correct / max(total, 1)})
    return pd.DataFrame(rows)


def evaluate_ppo_validation(agent: PPOAgent, params: SystemParams, base_seed: int) -> Tuple[float, float]:
    rewards, entropies = [], []
    for j, scenario in enumerate(SCENARIOS):
        env = CloudEdgeEnv(params, scenario, base_seed + j)
        state = env.reset(base_seed + j)
        local_ent = []
        while True:
            action, _, _, entropy = agent.select_action(state, deterministic=True)
            local_ent.append(entropy)
            state, _, done, _ = env.step(action)
            if done:
                break
        rewards.append(env.summary()["mean_reward"])
        entropies.append(float(np.mean(local_ent)))
    return float(np.mean(rewards)), float(np.mean(entropies))


def train_unified_ppo(
    params: SystemParams,
    seed: int,
    episodes: int = 48,
) -> Tuple[PPOAgent, pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    probe = CloudEdgeEnv(params, "normal", seed)
    agent = PPOAgent(probe.state_dim, probe.action_dim, seed)
    bc_history = behavior_clone_warm_start(agent, params, seed + 900)
    bc_score, bc_entropy = evaluate_ppo_validation(agent, params, 8000 + seed)
    history: List[Dict[str, float]] = []
    best_score = -float("inf")
    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_episode = -1
    set_global_seed(seed + 17)
    for episode in range(episodes):
        # The first eight episodes provide deterministic coverage of all scenarios;
        # thereafter scenario order is randomized.
        scenario = SCENARIOS[episode % 4] if episode < 8 else random.choice(SCENARIOS)
        env_seed = seed + episode
        env = CloudEdgeEnv(params, scenario, env_seed, domain_randomization=True)
        state = env.reset(env_seed)
        rollout = []
        rollout_entropies = []
        while True:
            action, log_prob, value, entropy = agent.select_action(state, deterministic=False)
            next_state, reward, done, _ = env.step(action)
            rollout.append((state, action, reward, done, log_prob, value))
            rollout_entropies.append(entropy)
            state = next_state
            if done:
                break
        update_info = agent.update(rollout)
        row = {
            "episode": episode + 1, "scenario": scenario,
            "rollout_entropy": float(np.mean(rollout_entropies)),
            **env.summary(), **update_info,
        }
        if (episode + 1) % 4 == 0:
            score, val_entropy = evaluate_ppo_validation(agent, params, 8000 + seed)
            row["validation_reward"] = score
            row["validation_entropy"] = val_entropy
            # Important: the initial behavior-cloned network is never eligible for
            # retention; all retained checkpoints have undergone PPO updates.
            if score > best_score:
                best_score = score
                best_state = copy.deepcopy(agent.policy.state_dict())
                best_episode = episode + 1
        history.append(row)
    if best_state is None:
        raise RuntimeError("No PPO-fine-tuned checkpoint was retained")
    agent.policy.load_state_dict(best_state)
    info = {
        "behavior_clone_validation_reward": bc_score,
        "behavior_clone_validation_entropy": bc_entropy,
        "retained_ppo_episode": float(best_episode),
        "retained_validation_reward": best_score,
    }
    return agent, pd.DataFrame(history), bc_history, info


# ----------------------------------- DQN ---------------------------------------

class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 96), nn.ReLU(),
            nn.Linear(96, 96), nn.ReLU(),
            nn.Linear(96, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DQNAgent:
    def __init__(self, state_dim: int, action_dim: int, seed: int) -> None:
        set_global_seed(seed)
        self.q = QNetwork(state_dim, action_dim)
        self.target = QNetwork(state_dim, action_dim)
        self.target.load_state_dict(self.q.state_dict())
        self.optimizer = optim.Adam(self.q.parameters(), lr=5e-4)
        self.gamma = 0.99
        self.action_dim = action_dim

    @torch.no_grad()
    def select_action(self, state: np.ndarray, epsilon: float = 0.0) -> int:
        if random.random() < epsilon:
            return random.randrange(self.action_dim)
        q = self.q(torch.as_tensor(state, dtype=torch.float32).unsqueeze(0))
        return int(q.argmax(dim=-1).item())

    def update(self, batch: Tuple[torch.Tensor, ...]) -> float:
        states, actions, rewards, next_states, dones = batch
        q = self.q(states).gather(1, actions[:, None]).squeeze(1)
        with torch.no_grad():
            next_q = self.target(next_states).max(dim=1).values
            target = rewards + self.gamma * next_q * (1.0 - dones)
        loss = nn.functional.smooth_l1_loss(q, target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), 1.0)
        self.optimizer.step()
        return float(loss.detach().item())


def evaluate_dqn_validation(agent: DQNAgent, params: SystemParams, base_seed: int) -> float:
    rewards = []
    for j, scenario in enumerate(SCENARIOS):
        env = CloudEdgeEnv(params, scenario, base_seed + j)
        state = env.reset(base_seed + j)
        while True:
            action = agent.select_action(state, epsilon=0.0)
            state, _, done, _ = env.step(action)
            if done:
                break
        rewards.append(env.summary()["mean_reward"])
    return float(np.mean(rewards))


def train_unified_dqn(params: SystemParams, seed: int, episodes: int = 48) -> Tuple[DQNAgent, pd.DataFrame, Dict[str, float]]:
    probe = CloudEdgeEnv(params, "normal", seed)
    agent = DQNAgent(probe.state_dim, probe.action_dim, seed)
    replay: deque = deque(maxlen=50000)
    rows = []
    best_score = -float("inf")
    best_state = None
    best_episode = -1
    steps = 0
    set_global_seed(seed + 23)
    for episode in range(episodes):
        scenario = SCENARIOS[episode % 4] if episode < 8 else random.choice(SCENARIOS)
        env_seed = seed + episode
        env = CloudEdgeEnv(params, scenario, env_seed, domain_randomization=True)
        state = env.reset(env_seed)
        losses = []
        while True:
            epsilon = max(0.05, 1.0 - steps / 25000.0)
            action = agent.select_action(state, epsilon)
            next_state, reward, done, _ = env.step(action)
            replay.append((state, action, reward, next_state, float(done)))
            state = next_state
            steps += 1
            if len(replay) >= 1000 and steps % 4 == 0:
                ids = np.random.randint(0, len(replay), 128)
                sample = [replay[i] for i in ids]
                batch = (
                    torch.as_tensor(np.asarray([x[0] for x in sample]), dtype=torch.float32),
                    torch.as_tensor([x[1] for x in sample], dtype=torch.long),
                    torch.as_tensor([x[2] for x in sample], dtype=torch.float32),
                    torch.as_tensor(np.asarray([x[3] for x in sample]), dtype=torch.float32),
                    torch.as_tensor([x[4] for x in sample], dtype=torch.float32),
                )
                losses.append(agent.update(batch))
            if steps % 1000 == 0:
                agent.target.load_state_dict(agent.q.state_dict())
            if done:
                break
        row = {"episode": episode + 1, "scenario": scenario, "loss": float(np.mean(losses)) if losses else np.nan, **env.summary()}
        if (episode + 1) % 4 == 0:
            score = evaluate_dqn_validation(agent, params, 9000 + seed)
            row["validation_reward"] = score
            if score > best_score:
                best_score = score
                best_state = copy.deepcopy(agent.q.state_dict())
                best_episode = episode + 1
        rows.append(row)
    if best_state is None:
        raise RuntimeError("No DQN checkpoint was retained")
    agent.q.load_state_dict(best_state)
    return agent, pd.DataFrame(rows), {"retained_dqn_episode": float(best_episode), "retained_validation_reward": best_score}


# ------------------------------- Policy evaluation -----------------------------

def static_rule_action(env: CloudEdgeEnv) -> int:
    """Transparent non-learning rule; thresholds are fixed a priori and disclosed."""
    q = env.current_task
    assert q is not None
    source = int(q["source"])
    battery_ratio = env.battery[source] / max(env.initial_battery[source], 1e-9)
    if battery_ratio < 0.25:
        return 5  # Edge 3 full FP32: low device-side serialization energy.
    if q["cloud_bw_factor"] >= 1.15 and q["gflops"] >= 3.8:
        return 7  # Cloud full FP32.
    if q["edge_bw_factor"] < 0.70:
        return 0  # Local FP32 during severe edge-link degradation.
    return 6      # Edge 3 split FP16 otherwise.


def edge_only_action(env: CloudEdgeEnv) -> int:
    candidates = (1, 3, 5)
    return min(candidates, key=lambda i: env.estimate_action(i)["latency_s"])


def evaluate_named_policy(
    params: SystemParams,
    scenario: str,
    seed: int,
    policy_name: str,
    ppo_agent: Optional[PPOAgent] = None,
    dqn_agent: Optional[DQNAgent] = None,
    constant_action: Optional[int] = None,
    stochastic_seed: Optional[int] = None,
    parameter_overrides: Optional[Dict[str, float]] = None,
    return_trace: bool = False,
) -> Tuple[Dict[str, float], Optional[pd.DataFrame]]:
    env = CloudEdgeEnv(params, scenario, seed, parameter_overrides=parameter_overrides)
    state = env.reset(seed)
    entropies = []
    if stochastic_seed is not None:
        set_global_seed(stochastic_seed)
    while True:
        if policy_name == "PPO":
            if ppo_agent is None:
                raise ValueError("ppo_agent required")
            action, _, _, entropy = ppo_agent.select_action(state, deterministic=True)
            entropies.append(entropy)
        elif policy_name == "PPO-stochastic":
            if ppo_agent is None:
                raise ValueError("ppo_agent required")
            action, _, _, entropy = ppo_agent.select_action(state, deterministic=False)
            entropies.append(entropy)
        elif policy_name == "DQN":
            if dqn_agent is None:
                raise ValueError("dqn_agent required")
            action = dqn_agent.select_action(state, epsilon=0.0)
        elif policy_name == "Local-only":
            action = 0
        elif policy_name == "Cloud-only":
            action = 7
        elif policy_name == "Edge-only":
            action = edge_only_action(env)
        elif policy_name == "Static-rule":
            action = static_rule_action(env)
        elif policy_name == "Greedy-myopic":
            action = greedy_myopic_action(env)
        elif policy_name == "Best-constant":
            if constant_action is None:
                raise ValueError("constant_action required")
            action = int(constant_action)
        else:
            raise ValueError(policy_name)
        state, _, done, _ = env.step(action)
        if done:
            break
    summary = env.summary()
    summary["policy_entropy"] = float(np.mean(entropies)) if entropies else np.nan
    summary["unique_actions"] = float(len(set(map(int, env.metrics["action"]))))
    trace = pd.DataFrame(env.metrics) if return_trace else None
    return summary, trace


def select_validation_best_constants(params: SystemParams, validation_seeds: Sequence[int]) -> pd.DataFrame:
    rows = []
    for scenario in SCENARIOS:
        candidate_rows = []
        for action_index, action in enumerate(ACTIONS):
            rewards = []
            for seed in validation_seeds:
                metrics, _ = evaluate_named_policy(params, scenario, seed, "Best-constant", constant_action=action_index)
                rewards.append(metrics["mean_reward"])
            candidate_rows.append((action_index, action.name, float(np.mean(rewards))))
        best = max(candidate_rows, key=lambda x: x[2])
        for action_index, name, reward in candidate_rows:
            rows.append({
                "scenario": scenario, "action_index": action_index, "action_name": name,
                "validation_mean_reward": reward, "selected": action_index == best[0],
            })
    return pd.DataFrame(rows)


def aggregate_learned_replicates(raw: pd.DataFrame) -> pd.DataFrame:
    learned = raw[raw["policy"].isin(["PPO", "DQN", "PPO-stochastic"])]
    nonlearned = raw[~raw["policy"].isin(["PPO", "DQN", "PPO-stochastic"])]
    metrics = ["latency_ms", "device_energy_j", "retention", "violation_rate", "drop_rate", "mean_reward", "depletion", "policy_entropy", "unique_actions"]
    learned_agg = learned.groupby(["scenario", "eval_seed", "policy"], as_index=False)[metrics].mean()
    nonlearned_agg = nonlearned.groupby(["scenario", "eval_seed", "policy"], as_index=False)[metrics].mean()
    return pd.concat([learned_agg, nonlearned_agg], ignore_index=True)


def summarize_for_manuscript(aggregated: pd.DataFrame) -> pd.DataFrame:
    metrics = ["latency_ms", "device_energy_j", "retention", "violation_rate", "drop_rate", "mean_reward"]
    out = aggregated.groupby(["scenario", "policy"])[metrics].agg(["mean", "std"]).reset_index()
    out.columns = ["_".join([x for x in c if x]).rstrip("_") for c in out.columns.to_flat_index()]
    return out


def holm_adjust(pvalues: np.ndarray) -> np.ndarray:
    pvalues = np.asarray(pvalues, dtype=float)
    order = np.argsort(pvalues)
    adjusted = np.full(len(pvalues), np.nan)
    running = 0.0
    m = len(pvalues)
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * pvalues[idx])
        adjusted[idx] = min(running, 1.0)
    return adjusted


def paired_rank_biserial(differences: np.ndarray) -> float:
    d = np.asarray(differences, dtype=float)
    d = d[np.abs(d) > 1e-12]
    if len(d) == 0:
        return np.nan
    ranks = rankdata(np.abs(d))
    r_pos = ranks[d > 0].sum()
    r_neg = ranks[d < 0].sum()
    return float((r_pos - r_neg) / (r_pos + r_neg))


def bootstrap_mean_ci(differences: np.ndarray, seed: int = 321, draws: int = 4000) -> Tuple[float, float]:
    d = np.asarray(differences, dtype=float)
    rng = np.random.default_rng(seed)
    means = np.empty(draws, dtype=float)
    for i in range(draws):
        means[i] = np.mean(rng.choice(d, size=len(d), replace=True))
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(lo), float(hi)


def paired_statistics(aggregated: pd.DataFrame) -> pd.DataFrame:
    """All PPO-vs-comparator tests in one Holm family.

    Retention is descriptive only because the configured retention is a
    deterministic function of the selected action, not a measured stochastic
    classification outcome.
    """
    comparators = ["Local-only", "Edge-only", "Cloud-only", "Static-rule", "Greedy-myopic", "Best-constant", "DQN"]
    metrics = ["latency_ms", "device_energy_j", "violation_rate"]
    rows = []
    for scenario in SCENARIOS:
        ppo = aggregated[(aggregated.scenario == scenario) & (aggregated.policy == "PPO")].sort_values("eval_seed")
        for comparator in comparators:
            base = aggregated[(aggregated.scenario == scenario) & (aggregated.policy == comparator)].sort_values("eval_seed")
            for metric in metrics:
                a = ppo[metric].to_numpy(dtype=float)
                b = base[metric].to_numpy(dtype=float)
                if len(a) != len(b) or not np.array_equal(ppo.eval_seed.to_numpy(), base.eval_seed.to_numpy()):
                    raise RuntimeError("Paired evaluation seeds do not align")
                diff = a - b
                testable = not np.allclose(diff, 0.0, atol=1e-12)
                if testable:
                    try:
                        statistic, pvalue = wilcoxon(diff, zero_method="wilcox", alternative="two-sided", method="auto")
                    except ValueError:
                        statistic, pvalue, testable = np.nan, np.nan, False
                else:
                    statistic, pvalue = np.nan, np.nan
                ci_lo, ci_hi = bootstrap_mean_ci(diff, seed=100 + len(rows))
                rows.append({
                    "scenario": scenario, "comparator": comparator, "metric": metric,
                    "n_pairs": len(diff), "mean_difference_ppo_minus_comparator": float(np.mean(diff)),
                    "ci95_low": ci_lo, "ci95_high": ci_hi,
                    "rank_biserial": paired_rank_biserial(diff),
                    "statistic": statistic, "p_value": pvalue, "testable": testable,
                })
    df = pd.DataFrame(rows)
    mask = df["testable"] & df["p_value"].notna()
    df["p_holm_global"] = np.nan
    if mask.any():
        df.loc[mask, "p_holm_global"] = holm_adjust(df.loc[mask, "p_value"].to_numpy())
    return df


# ------------------------------ Diagnostics ------------------------------------

def state_dependence_probe(agent: PPOAgent, params: SystemParams, seed: int, target_states: int = 20000) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    states = []
    while len(states) < target_states:
        scenario = SCENARIOS[int(rng.integers(0, 4))]
        env_seed = int(rng.integers(100000, 900000))
        env = CloudEdgeEnv(params, scenario, env_seed, domain_randomization=True)
        state = env.reset(env_seed)
        while env.current_task is not None and len(states) < target_states:
            states.append(state.copy())
            # Random behavior deliberately explores queue configurations.
            action = int(rng.integers(0, env.action_dim))
            state, _, done, _ = env.step(action)
            if done:
                break
    x = torch.as_tensor(np.asarray(states), dtype=torch.float32)
    with torch.no_grad():
        logits, _ = agent.policy(x)
        probs = torch.softmax(logits, dim=-1)
        actions = logits.argmax(dim=-1).cpu().numpy()
        entropy = (-probs * torch.log(probs + 1e-12)).sum(dim=-1).cpu().numpy()
    counts = np.bincount(actions, minlength=len(ACTIONS))
    top_share = counts.max() / counts.sum()
    out: Dict[str, float] = {
        "n_states": float(len(states)), "distinct_argmax_actions": float(np.count_nonzero(counts)),
        "top_action_share": float(top_share), "mean_entropy": float(np.mean(entropy)),
        "max_entropy": float(math.log(len(ACTIONS))),
    }
    for i, action in enumerate(ACTIONS):
        out[f"share_{action.name}"] = float(counts[i] / counts.sum())
    return out


def scenario_coverage(params: SystemParams, seeds: Sequence[int]) -> pd.DataFrame:
    rows = []
    for scenario in ("burst", "fluctuation"):
        for seed in seeds:
            env = CloudEdgeEnv(params, scenario, seed)
            env.reset(seed)
            while env.current_task is not None:
                # Action is irrelevant to arrival-phase coverage; use local execution.
                _, _, done, _ = env.step(0)
                if done:
                    break
            trace = pd.DataFrame(env.metrics)
            if scenario == "burst":
                for value, count in trace["burst_window"].value_counts().sort_index().items():
                    rows.append({"scenario": scenario, "seed": seed, "phase": "burst" if int(value) == 1 else "recovery", "tasks": int(count)})
            else:
                for value, count in trace["fluctuation_phase"].value_counts().sort_index().items():
                    rows.append({"scenario": scenario, "seed": seed, "phase": f"phase_{int(value)}", "tasks": int(count)})
    return pd.DataFrame(rows)


def sensitivity_analysis(
    params: SystemParams,
    ppo_agents: Sequence[PPOAgent],
    dqn_agents: Sequence[DQNAgent],
    seeds: Sequence[int],
) -> pd.DataFrame:
    settings = [
        ("arrival_scale", 0.75), ("arrival_scale", 1.00), ("arrival_scale", 1.25),
        ("bandwidth_scale", 0.70), ("bandwidth_scale", 1.00), ("bandwidth_scale", 1.30),
        ("capacity_scale", 0.80), ("capacity_scale", 1.00), ("capacity_scale", 1.20),
    ]
    rows = []
    for parameter, value in settings:
        overrides = {parameter: value}
        for policy in ("PPO", "DQN"):
            agents = ppo_agents if policy == "PPO" else dqn_agents
            for training_rep, agent in enumerate(agents):
                for seed in seeds:
                    kwargs = {"ppo_agent": agent} if policy == "PPO" else {"dqn_agent": agent}
                    metrics, _ = evaluate_named_policy(params, "normal", seed, policy, parameter_overrides=overrides, **kwargs)
                    rows.append({"parameter": parameter, "value": value, "policy": policy, "training_rep": training_rep, "eval_seed": seed, **metrics})
    return pd.DataFrame(rows)


# ---------------------------------- Figures ------------------------------------

def configure_plot_fonts() -> None:
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
        "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
        "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    })


def save_figure_both(fig: plt.Figure, path_without_suffix: Path) -> None:
    fig.savefig(path_without_suffix.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(path_without_suffix.with_suffix(".png"), dpi=350, bbox_inches="tight")
    plt.close(fig)


def make_architecture_figure(params: SystemParams, out: Path) -> None:
    configure_plot_fonts()
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.axis("off")
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    def box(x, y, w, h, text, lw=1.0):
        p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02", fill=False, linewidth=lw)
        ax.add_patch(p); ax.text(x+w/2, y+h/2, text, ha="center", va="center")
    box(0.34, 0.80, 0.32, 0.12, f"Cloud node\n{params.cloud_cap_gflops:.0f} GFLOPS; FIFO queue")
    box(0.36, 0.61, 0.28, 0.10, "Unified PPO / DQN orchestrator\nstate -> one of nine actions")
    for i, cap in enumerate(params.edge_caps_gflops):
        box(0.08 + i*0.30, 0.38, 0.24, 0.11, f"Edge server {i+1}\n{cap:.0f} GFLOPS")
    for i, cap in enumerate(params.device_caps_gflops):
        box(0.02 + i*0.245, 0.10, 0.20, 0.11, f"Source device {i+1}\n{cap:.0f} GFLOPS + battery")
    ax.add_patch(FancyArrowPatch((0.50,0.80),(0.50,0.71),arrowstyle="<->",mutation_scale=12))
    ax.add_patch(FancyArrowPatch((0.50,0.61),(0.50,0.50),arrowstyle="->",mutation_scale=12))
    for i in range(3): ax.add_patch(FancyArrowPatch((0.50,0.61),(0.20+i*0.30,0.49),arrowstyle="->",mutation_scale=10))
    for i in range(4): ax.add_patch(FancyArrowPatch((0.12+i*0.245,0.21),(0.20+min(i,2)*0.30,0.38),arrowstyle="<->",mutation_scale=9))
    ax.text(0.70,0.74,"device-cloud: 75 Mbps, 42 ms nominal",ha="left",va="center",fontsize=8)
    ax.text(0.69,0.31,"device-edge: 90 Mbps, 12 ms nominal",ha="left",va="center",fontsize=8)
    ax.set_xlim(0,1); ax.set_ylim(0,1)
    fig.tight_layout()
    save_figure_both(fig, out / "figure_1_architecture")


def make_ppo_workflow_figure(out: Path) -> None:
    configure_plot_fonts()
    fig, ax = plt.subplots(figsize=(7.2, 5.2)); ax.axis("off")
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    nodes = [
        (0.32,0.86,0.36,0.08,"Mixed-scenario environment\n45 s event-based episode"),
        (0.32,0.72,0.36,0.08,"State observation (17 variables)"),
        (0.32,0.58,0.36,0.08,"Actor samples one of 9 actions"),
        (0.32,0.44,0.36,0.08,"Environment step + reward"),
        (0.32,0.30,0.36,0.08,"On-policy rollout buffer"),
        (0.32,0.16,0.36,0.08,"GAE + clipped PPO objective\nactor-critic update"),
    ]
    for x,y,w,h,t in nodes:
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.02",fill=False,linewidth=1.0)); ax.text(x+w/2,y+h/2,t,ha="center",va="center")
    for j in range(len(nodes)-1):
        x=0.50; y1=nodes[j][1]; y2=nodes[j+1][1]+nodes[j+1][3]
        ax.add_patch(FancyArrowPatch((x,y1),(x,y2),arrowstyle="->",mutation_scale=12))
    ax.add_patch(FancyArrowPatch((0.32,0.20),(0.12,0.20),connectionstyle="arc3,rad=0.0",arrowstyle="->",mutation_scale=12))
    ax.add_patch(FancyArrowPatch((0.12,0.20),(0.12,0.76),connectionstyle="arc3,rad=0.0",arrowstyle="->",mutation_scale=12))
    ax.add_patch(FancyArrowPatch((0.12,0.76),(0.32,0.76),connectionstyle="arc3,rad=0.0",arrowstyle="->",mutation_scale=12))
    ax.text(0.08,0.49,"updated policy",rotation=90,ha="center",va="center",fontsize=8)
    ax.text(0.72,0.19,"Behavior-cloned actor initialization\nprecedes PPO fine-tuning",ha="left",va="center",fontsize=8)
    ax.set_xlim(0,1); ax.set_ylim(0.08,0.98); fig.tight_layout()
    save_figure_both(fig, out / "figure_2_ppo_workflow")


def make_result_figures(aggregated: pd.DataFrame, raw: pd.DataFrame, ppo_agent: PPOAgent, params: SystemParams, out: Path) -> None:
    configure_plot_fonts()
    policies = ["Local-only", "Edge-only", "Cloud-only", "Static-rule", "Greedy-myopic", "Best-constant", "DQN", "PPO"]
    scenario_labels = {"normal":"Normal", "burst":"Burst/recovery", "fluctuation":"Network fluctuation", "energy":"Energy constrained"}
    # Figure 3: log-scale latency with hatching to remain interpretable in grayscale.
    fig, ax = plt.subplots(figsize=(8.0, 4.8)); x=np.arange(4); width=.10
    hatches=["//","\\\\","..","xx","++","--","oo","**"]
    for i,policy in enumerate(policies):
        means=[]; stds=[]
        for sc in SCENARIOS:
            v=aggregated[(aggregated.scenario==sc)&(aggregated.policy==policy)]["latency_ms"]
            means.append(v.mean()); stds.append(v.std())
        ax.bar(x+(i-3.5)*width,means,width,yerr=stds,capsize=1.5,label=policy,fill=False,hatch=hatches[i],linewidth=.8)
    ax.set_yscale("log"); ax.set_ylabel("Mean end-to-end latency (ms; log scale)"); ax.set_xticks(x); ax.set_xticklabels([scenario_labels[s] for s in SCENARIOS]); ax.legend(ncol=4,loc="upper center",bbox_to_anchor=(.5,1.23)); fig.tight_layout()
    save_figure_both(fig,out/"figure_3_latency")
    # Figure 4: latency-energy trade-off, unique marker for every policy.
    fig,ax=plt.subplots(figsize=(6.8,4.8)); markers=["s","^","v","D","P","X","o","*"]
    normal=aggregated[aggregated.scenario=="normal"].groupby("policy").mean(numeric_only=True)
    sizes={"Cloud-only":120,"Best-constant":55,"PPO":110}
    for i,p in enumerate(policies):
        ax.scatter(normal.loc[p,"device_energy_j"],normal.loc[p,"latency_ms"],marker=markers[i],s=sizes.get(p,65),label=p,zorder=4 if p=="Best-constant" else 3)
    x_overlap=normal.loc["Cloud-only","device_energy_j"]; y_overlap=normal.loc["Cloud-only","latency_ms"]
    ax.annotate("Cloud-only = Best-constant",xy=(x_overlap,y_overlap),xytext=(x_overlap+0.04,y_overlap-3.5),arrowprops=dict(arrowstyle="->"),fontsize=8)
    ax.set_xlabel("Device-side energy per task (J)"); ax.set_ylabel("Mean latency (ms)"); ax.legend(ncol=2); fig.tight_layout()
    save_figure_both(fig,out/"figure_4_tradeoff")
    # Figure 5: one evaluation-seed trace generated under the exact 45-s protocol.
    _,trace=evaluate_named_policy(params,"fluctuation",1000,"PPO",ppo_agent=ppo_agent,return_trace=True)
    assert trace is not None
    fig,ax1=plt.subplots(figsize=(7.6,4.5)); ax2=ax1.twinx()
    ax1.step(trace["arrival_time_s"],trace["action"],where="post",linewidth=.9,label="PPO action index")
    ax2.plot(trace["arrival_time_s"],trace["cloud_bandwidth_mbps"],linestyle="--",linewidth=1.0,label="Cloud bandwidth")
    ax1.set_xlabel("Arrival time (s)"); ax1.set_ylabel("PPO action index"); ax1.set_yticks(range(9)); ax1.set_ylim(-.5,8.5); ax2.set_ylabel("Cloud bandwidth (Mbps)")
    lines=ax1.get_lines()+ax2.get_lines(); ax1.legend(lines,[l.get_label() for l in lines],loc="upper right"); fig.tight_layout()
    save_figure_both(fig,out/"figure_5_dynamic_trace")
    # Figure 6: policy entropy during PPO training (mean across training seeds).
    ph=raw.attrs.get("ppo_history")
    if isinstance(ph,pd.DataFrame):
        fig,ax=plt.subplots(figsize=(6.8,4.2)); g=ph.groupby("episode")["rollout_entropy"].agg(["mean","std"]); ax.plot(g.index,g["mean"],marker="o",markersize=2.5,linewidth=1.0); ax.fill_between(g.index,(g["mean"]-g["std"]).to_numpy(),(g["mean"]+g["std"]).to_numpy(),alpha=.15); ax.axhline(math.log(9),linestyle="--",linewidth=.9,label="Uniform-policy maximum ln(9)"); ax.set_xlabel("PPO fine-tuning episode"); ax.set_ylabel("Policy entropy (nats)"); ax.legend(); fig.tight_layout(); save_figure_both(fig,out/"figure_6_entropy")


# ------------------------------- Full experiment -------------------------------

def run_complete_experiment(output_root: str | Path) -> Dict[str, object]:
    root = Path(output_root)
    results_dir = root / "results"; checkpoints_dir = root / "checkpoints"; figures_dir = root / "figures"; config_dir=root/"config"
    for d in (results_dir, checkpoints_dir, figures_dir, config_dir): d.mkdir(parents=True, exist_ok=True)
    params = SystemParams()
    ppo_training_seeds = [42, 142, 242]
    dqn_training_seeds = [123, 223, 323]
    evaluation_seeds = list(range(1000, 1020))
    validation_constant_seeds = list(range(7000, 7005))
    sensitivity_seeds = list(range(1100, 1105))

    ppo_agents: List[PPOAgent] = []
    dqn_agents: List[DQNAgent] = []
    ppo_histories=[]; bc_histories=[]; ppo_info=[]; dqn_histories=[]; dqn_info=[]
    for rep,seed in enumerate(ppo_training_seeds):
        agent,hist,bc,info=train_unified_ppo(params,seed,episodes=32)
        ppo_agents.append(agent); hist.insert(0,"training_rep",rep); hist.insert(1,"training_seed",seed); ppo_histories.append(hist); bc.insert(0,"training_rep",rep); bc.insert(1,"training_seed",seed); bc_histories.append(bc); ppo_info.append({"training_rep":rep,"training_seed":seed,**info})
        torch.save(agent.policy.state_dict(),checkpoints_dir/f"ppo_unified_seed_{seed}.pt")
    for rep,seed in enumerate(dqn_training_seeds):
        agent,hist,info=train_unified_dqn(params,seed,episodes=32)
        dqn_agents.append(agent); hist.insert(0,"training_rep",rep); hist.insert(1,"training_seed",seed); dqn_histories.append(hist); dqn_info.append({"training_rep":rep,"training_seed":seed,**info})
        torch.save(agent.q.state_dict(),checkpoints_dir/f"dqn_unified_seed_{seed}.pt")

    constant_grid=select_validation_best_constants(params,validation_constant_seeds)
    selected_constant={sc:int(constant_grid[(constant_grid.scenario==sc)&(constant_grid.selected)].iloc[0].action_index) for sc in SCENARIOS}

    rows=[]
    nonlearned=["Local-only","Edge-only","Cloud-only","Static-rule","Greedy-myopic","Best-constant"]
    for scenario in SCENARIOS:
        for eval_seed in evaluation_seeds:
            for policy in nonlearned:
                metrics,_=evaluate_named_policy(params,scenario,eval_seed,policy,constant_action=selected_constant.get(scenario))
                rows.append({"scenario":scenario,"eval_seed":eval_seed,"training_rep":-1,"training_seed":-1,"policy":policy,**metrics})
            for rep,agent in enumerate(ppo_agents):
                metrics,_=evaluate_named_policy(params,scenario,eval_seed,"PPO",ppo_agent=agent)
                rows.append({"scenario":scenario,"eval_seed":eval_seed,"training_rep":rep,"training_seed":ppo_training_seeds[rep],"policy":"PPO",**metrics})
                stochastic,_=evaluate_named_policy(params,scenario,eval_seed,"PPO-stochastic",ppo_agent=agent,stochastic_seed=500000+rep*10000+eval_seed)
                rows.append({"scenario":scenario,"eval_seed":eval_seed,"training_rep":rep,"training_seed":ppo_training_seeds[rep],"policy":"PPO-stochastic",**stochastic})
            for rep,agent in enumerate(dqn_agents):
                metrics,_=evaluate_named_policy(params,scenario,eval_seed,"DQN",dqn_agent=agent)
                rows.append({"scenario":scenario,"eval_seed":eval_seed,"training_rep":rep,"training_seed":dqn_training_seeds[rep],"policy":"DQN",**metrics})
    raw=pd.DataFrame(rows)
    aggregated=aggregate_learned_replicates(raw)
    summary=summarize_for_manuscript(aggregated)
    stats=paired_statistics(aggregated)
    coverage=scenario_coverage(params,evaluation_seeds)
    state_probe=pd.DataFrame([{"training_rep":i,"training_seed":ppo_training_seeds[i],**state_dependence_probe(agent,params,6000+i)} for i,agent in enumerate(ppo_agents)])
    sensitivity=sensitivity_analysis(params,ppo_agents,dqn_agents,sensitivity_seeds)
    sens_summary=sensitivity.groupby(["parameter","value","policy"])[["latency_ms","device_energy_j","violation_rate","mean_reward"]].agg(["mean","std"]).reset_index()
    sens_summary.columns=["_".join([x for x in c if x]).rstrip("_") for c in sens_summary.columns.to_flat_index()]

    ppo_history=pd.concat(ppo_histories,ignore_index=True); bc_history=pd.concat(bc_histories,ignore_index=True); dqn_history=pd.concat(dqn_histories,ignore_index=True)
    raw.attrs["ppo_history"]=ppo_history
    ppo_history.to_csv(results_dir/"ppo_training_history.csv",index=False)
    bc_history.to_csv(results_dir/"ppo_behavior_clone_history.csv",index=False)
    dqn_history.to_csv(results_dir/"dqn_training_history.csv",index=False)
    pd.DataFrame(ppo_info).to_csv(results_dir/"ppo_checkpoint_selection.csv",index=False)
    pd.DataFrame(dqn_info).to_csv(results_dir/"dqn_checkpoint_selection.csv",index=False)
    constant_grid.to_csv(results_dir/"constant_action_validation.csv",index=False)
    raw.to_csv(results_dir/"results_raw_all_replicates.csv",index=False)
    aggregated.to_csv(results_dir/"results_paired_aggregated.csv",index=False)
    summary.to_csv(results_dir/"results_summary.csv",index=False)
    stats.to_csv(results_dir/"paired_statistics_global_holm.csv",index=False)
    coverage.to_csv(results_dir/"scenario_coverage.csv",index=False)
    state_probe.to_csv(results_dir/"state_dependence_probe.csv",index=False)
    sensitivity.to_csv(results_dir/"sensitivity_raw.csv",index=False)
    sens_summary.to_csv(results_dir/"sensitivity_summary.csv",index=False)

    make_architecture_figure(params,figures_dir)
    make_ppo_workflow_figure(figures_dir)
    make_result_figures(aggregated,raw,ppo_agents[0],params,figures_dir)

    config={
        "params":asdict(params),
        "scenarios":list(SCENARIOS),
        "actions":[asdict(a) for a in ACTIONS],
        "ppo_training_seeds":ppo_training_seeds,
        "dqn_training_seeds":dqn_training_seeds,
        "evaluation_seeds":evaluation_seeds,
        "constant_validation_seeds":validation_constant_seeds,
        "sensitivity_seeds":sensitivity_seeds,
        "ppo_fine_tuning_episodes":32,
        "dqn_training_episodes":32,
        "ppo_behavior_clone_samples":20000,
        "ppo_behavior_clone_epochs":20,
        "reward":{
            "latency_weight":0.55,"energy_weight":0.25,"quality_weight":0.20,
            "deadline_violation_penalty":1.0,"infeasible_energy_penalty":3.0,
            "latency_normalizer":"task deadline","energy_normalizer_j":0.80,
            "quality_degradation_normalizer":0.02,
            "energy_pressure":"1 + 6(1-battery_ratio)^2",
        },
        "software":{"python":platform.python_version(),"torch":torch.__version__,"numpy":np.__version__,"pandas":pd.__version__},
    }
    with open(config_dir/"experiment_config.json","w",encoding="utf-8") as f: json.dump(config,f,indent=2)
    return {"summary":summary,"stats":stats,"coverage":coverage,"state_probe":state_probe,"sensitivity_summary":sens_summary,"selected_constant":selected_constant,"ppo_info":ppo_info,"dqn_info":dqn_info}


if __name__ == "__main__":
    root=Path(__file__).resolve().parents[1]
    outputs=run_complete_experiment(root)
    print("Completed reviewer-driven experiment revision.")
    print("Selected best-constant actions:",outputs["selected_constant"])
    print(outputs["summary"].to_string(index=False))
