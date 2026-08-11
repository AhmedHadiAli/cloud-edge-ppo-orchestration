"""Reproducible cloud-edge orchestration simulation.

This module replaces the original proof-of-concept code with an event-based,
seed-controlled simulator, valid heuristic baselines, deterministic evaluation,
and multi-run statistical reporting. It deliberately describes model quality
as an accuracy-retention profile rather than claiming that an image dataset is
executed when no dataset is present.
"""
from __future__ import annotations

import copy
import json
import math
import random
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
torch.set_num_threads(1)
import torch.nn as nn
import torch.optim as optim
from scipy.stats import wilcoxon
from torch.distributions import Categorical


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@dataclass(frozen=True)
class ModelProfile:
    name: str
    base_gflops: float
    base_accuracy: float
    layer_fractions: Tuple[float, ...]
    activation_mb: Tuple[float, ...]

    @property
    def num_layers(self) -> int:
        return len(self.layer_fractions)


MODEL_PROFILES: Dict[str, ModelProfile] = {
    "resnet50": ModelProfile(
        name="ResNet-50 analytical profile",
        base_gflops=4.10,
        base_accuracy=0.935,
        layer_fractions=(0.04, 0.05, 0.07, 0.08, 0.10, 0.10, 0.11, 0.11, 0.10, 0.09, 0.08, 0.07),
        activation_mb=(3.20, 2.80, 2.40, 1.80, 1.40, 1.10, 0.90, 0.65, 0.48, 0.30, 0.18, 0.02),
    ),
    "mobilenetv2": ModelProfile(
        name="MobileNetV2 analytical profile",
        base_gflops=0.32,
        base_accuracy=0.900,
        layer_fractions=(0.07, 0.09, 0.10, 0.11, 0.12, 0.12, 0.12, 0.11, 0.09, 0.07),
        activation_mb=(2.40, 2.00, 1.60, 1.20, 0.90, 0.65, 0.44, 0.28, 0.14, 0.02),
    ),
}


def get_model_profile(model_name: str) -> ModelProfile:
    try:
        return MODEL_PROFILES[model_name.lower()]
    except KeyError as exc:
        raise ValueError(f"Unknown model profile: {model_name}") from exc


def accuracy_retention(base_accuracy: float, quant_bits: int, split_compression: bool) -> float:
    """Return a monotonic accuracy profile.

    Partitioning alone does not reduce accuracy. A small calibrated penalty is
    applied only when reduced-precision execution or activation transmission is
    selected. The values are simulation assumptions and are reported as such.
    """
    quant_penalty = {32: 0.000, 16: 0.003, 8: 0.012}[quant_bits]
    activation_penalty = 0.002 if split_compression and quant_bits < 32 else 0.0
    return float(np.clip(base_accuracy - quant_penalty - activation_penalty, 0.0, 1.0))


def is_pareto_efficient(costs: np.ndarray) -> np.ndarray:
    """Return a mask for non-dominated rows in a minimization problem."""
    if costs.ndim != 2:
        raise ValueError("costs must be a two-dimensional array")
    efficient = np.ones(costs.shape[0], dtype=bool)
    for i, candidate in enumerate(costs):
        if not efficient[i]:
            continue
        dominated = np.all(costs >= candidate, axis=1) & np.any(costs > candidate, axis=1)
        efficient[dominated] = False
    return efficient


@dataclass
class SystemParams:
    num_devices: int = 2
    num_edge_servers: int = 2
    device_caps_gflops: Tuple[float, ...] = (10.0, 14.0, 18.0, 22.0)
    edge_caps_gflops: Tuple[float, ...] = (25.0, 40.0, 70.0, 100.0)
    cloud_cap_gflops: float = 420.0
    device_power_active_w: float = 3.5
    device_power_idle_w: float = 0.6
    device_tx_power_w: float = 1.6
    edge_power_active_w: float = 65.0
    cloud_power_active_w: float = 220.0
    device_battery_j: float = 320.0
    bw_device_edge_mbps: float = 60.0
    bw_device_cloud_mbps: float = 45.0
    latency_device_edge_ms: float = 8.0
    latency_device_cloud_ms: float = 55.0
    arrival_rate_tasks_s: float = 12.0
    input_size_mean_mb: float = 0.75
    input_size_std_mb: float = 0.20
    complexity_mean: float = 0.90
    complexity_std: float = 0.18
    deadline_mean_ms: float = 650.0
    deadline_std_ms: float = 130.0
    model_name: str = "resnet50"
    max_tasks: int = 160

    def validate(self) -> None:
        if not (1 <= self.num_devices <= len(self.device_caps_gflops)):
            raise ValueError("num_devices exceeds configured capacities")
        if not (1 <= self.num_edge_servers <= len(self.edge_caps_gflops)):
            raise ValueError("num_edge_servers exceeds configured capacities")
        if self.max_tasks <= 0:
            raise ValueError("max_tasks must be positive")


@dataclass
class ResourceNode:
    name: str
    cap_gflops: float
    power_active_w: float
    battery_j: Optional[float] = None
    busy_until: float = 0.0
    total_busy_time: float = 0.0

    def execution_time(self, gflops: float, precision_speed: float = 1.0) -> float:
        effective_capacity = self.cap_gflops / max(precision_speed, 1e-8)
        return gflops / effective_capacity

    @property
    def initial_battery_j(self) -> Optional[float]:
        return getattr(self, "_initial_battery_j", self.battery_j)

    def set_initial_battery(self, value: Optional[float]) -> None:
        self._initial_battery_j = value
        self.battery_j = value

    def battery_ratio(self) -> float:
        if self.initial_battery_j is None:
            return 1.0
        return float(np.clip((self.battery_j or 0.0) / self.initial_battery_j, 0.0, 1.0))


@dataclass
class NetworkLink:
    bandwidth_mbps: float
    propagation_ms: float

    def transmission_time(self, data_mb: float, bw_factor: float = 1.0, latency_factor: float = 1.0) -> float:
        bandwidth = max(self.bandwidth_mbps * bw_factor, 0.1)
        return (data_mb * 8.0 / bandwidth) + (self.propagation_ms * latency_factor / 1000.0)


@dataclass(frozen=True)
class ActionSpec:
    name: str
    target_type: str
    target_index: Optional[int]
    partition_fraction: float
    quant_bits: int


class CloudEdgeEnv:
    """Event-based task simulator with queueing represented by resource busy times."""

    def __init__(
        self,
        params: SystemParams,
        scenario: str = "normal",
        seed: int = 42,
        ablation_mode: Optional[str] = None,
        fixed_weights: Optional[Tuple[float, float, float]] = None,
    ) -> None:
        params.validate()
        if scenario not in {"normal", "burst", "fluctuation", "energy_constrained"}:
            raise ValueError(f"Unknown scenario: {scenario}")
        if ablation_mode not in {None, "no_adaptive", "no_partition", "no_quant"}:
            raise ValueError(f"Unknown ablation mode: {ablation_mode}")
        self.params = params
        self.scenario = scenario
        self.seed = seed
        self.ablation_mode = ablation_mode
        self.fixed_weights = fixed_weights
        self.profile = get_model_profile(params.model_name)
        self.rng = np.random.default_rng(seed)
        self.devices: List[ResourceNode] = []
        self.edges: List[ResourceNode] = []
        self.cloud: ResourceNode
        self.link_device_edge = NetworkLink(params.bw_device_edge_mbps, params.latency_device_edge_ms)
        self.link_device_cloud = NetworkLink(params.bw_device_cloud_mbps, params.latency_device_cloud_ms)
        self.actions = self._build_actions()
        self.metrics: Dict[str, List[float]] = {}
        self.current_task: Optional[Dict[str, float]] = None
        self.current_time = 0.0
        self.last_finish_time = 0.0
        self.task_index = 0
        self.recent_violations: List[int] = []
        self.reset(seed)

    def _build_actions(self) -> List[ActionSpec]:
        actions = [
            ActionSpec("local-fp32", "local", None, 1.0, 32),
            ActionSpec("local-int8", "local", None, 1.0, 8),
        ]
        for i in range(self.params.num_edge_servers):
            actions.append(ActionSpec(f"edge-{i + 1}-full-fp32", "edge", i, 0.0, 32))
            actions.append(ActionSpec(f"edge-{i + 1}-split-fp16", "edge", i, 0.50, 16))
        actions.extend(
            [
                ActionSpec("cloud-full-fp32", "cloud", None, 0.0, 32),
                ActionSpec("cloud-split-fp16", "cloud", None, 0.67, 16),
                ActionSpec("cloud-full-int8", "cloud", None, 0.0, 8),
            ]
        )
        if self.ablation_mode == "no_partition":
            actions = [a for a in actions if a.partition_fraction in {0.0, 1.0}]
        if self.ablation_mode == "no_quant":
            actions = [ActionSpec(a.name.replace("fp16", "fp32").replace("int8", "fp32"), a.target_type, a.target_index, a.partition_fraction, 32) for a in actions]
        # Remove duplicates introduced by no_quant while preserving order.
        unique: List[ActionSpec] = []
        seen = set()
        for action in actions:
            key = (action.target_type, action.target_index, action.partition_fraction, action.quant_bits)
            if key not in seen:
                unique.append(action)
                seen.add(key)
        return unique

    @property
    def action_dim(self) -> int:
        return len(self.actions)

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        if seed is not None:
            self.seed = seed
            self.rng = np.random.default_rng(seed)
        self.devices = []
        for i in range(self.params.num_devices):
            node = ResourceNode(
                name=f"device-{i + 1}",
                cap_gflops=self.params.device_caps_gflops[i],
                power_active_w=self.params.device_power_active_w,
            )
            ratio = 1.0
            if self.scenario == "energy_constrained":
                ratio = float(self.rng.choice([0.30, 0.50, 0.70]))
            node.set_initial_battery(self.params.device_battery_j * ratio)
            self.devices.append(node)
        self.edges = [
            ResourceNode(f"edge-{i + 1}", self.params.edge_caps_gflops[i], self.params.edge_power_active_w)
            for i in range(self.params.num_edge_servers)
        ]
        self.cloud = ResourceNode("cloud", self.params.cloud_cap_gflops, self.params.cloud_power_active_w)
        self.metrics = {
            "latency_ms": [],
            "device_energy_j": [],
            "accuracy": [],
            "violation": [],
            "partition_layer": [],
            "quant_bits": [],
            "action": [],
            "arrival_time_s": [],
            "bandwidth_mbps": [],
            "reward": [],
        }
        self.current_time = 0.0
        self.last_finish_time = 0.0
        self.task_index = 0
        self.recent_violations = []
        self.current_task = self._generate_next_task(first=True)
        return self._get_state()

    def _scenario_factors(self, time_s: float) -> Tuple[float, float, float]:
        arrival_factor = 1.0
        bandwidth_factor = 1.0
        latency_factor = 1.0
        if self.scenario == "burst":
            arrival_factor = 3.0 if (time_s % 30.0) < 10.0 else 1.0
        elif self.scenario == "fluctuation":
            phase = int(time_s // 10.0) % 4
            bandwidth_factor = (0.45, 0.75, 1.30, 1.80)[phase]
            latency_factor = (1.90, 1.35, 0.85, 0.60)[phase]
        return arrival_factor, bandwidth_factor, latency_factor

    def _generate_next_task(self, first: bool = False) -> Dict[str, float]:
        arrival_factor, _, _ = self._scenario_factors(self.current_time)
        rate = self.params.arrival_rate_tasks_s * arrival_factor
        interarrival = 0.0 if first else float(self.rng.exponential(1.0 / rate))
        self.current_time += interarrival
        source_index = self.task_index % self.params.num_devices
        input_size = float(np.clip(self.rng.normal(self.params.input_size_mean_mb, self.params.input_size_std_mb), 0.20, 1.60))
        complexity = float(np.clip(self.rng.normal(self.params.complexity_mean, self.params.complexity_std), 0.50, 1.35))
        total_gflops = self.profile.base_gflops * complexity
        deadline_ms = float(np.clip(self.rng.normal(self.params.deadline_mean_ms, self.params.deadline_std_ms), 350.0, 1300.0))
        task = {
            "id": float(self.task_index),
            "arrival_time": self.current_time,
            "source_index": float(source_index),
            "input_size_mb": input_size,
            "complexity": complexity,
            "total_gflops": total_gflops,
            "deadline_s": deadline_ms / 1000.0,
        }
        self.task_index += 1
        return task

    def _weights(self) -> Tuple[float, float, float]:
        if self.fixed_weights is not None:
            return self.fixed_weights
        if self.ablation_mode == "no_adaptive":
            return (0.40, 0.30, 0.30)
        battery = float(np.mean([d.battery_ratio() for d in self.devices]))
        recent_rate = float(np.mean(self.recent_violations[-20:])) if self.recent_violations else 0.0
        if self.scenario == "energy_constrained" or battery < 0.30:
            return (0.20, 0.70, 0.10)
        if recent_rate > 0.20:
            return (0.60, 0.15, 0.25)
        return (0.40, 0.30, 0.30)

    @staticmethod
    def _precision_speed_factor(bits: int) -> float:
        return {32: 1.0, 16: 0.78, 8: 0.58}[bits]

    def estimate_action(self, action_index: int) -> Dict[str, float]:
        if self.current_task is None:
            raise RuntimeError("No active task")
        if not (0 <= action_index < self.action_dim):
            raise IndexError("Invalid action index")
        action = self.actions[action_index]
        task = self.current_task
        source = self.devices[int(task["source_index"])]
        arrival = task["arrival_time"]
        _, bw_factor, latency_factor = self._scenario_factors(arrival)
        layer_gflops = np.asarray(self.profile.layer_fractions) * task["total_gflops"]
        num_layers = self.profile.num_layers
        partition = int(round(action.partition_fraction * num_layers))
        partition = int(np.clip(partition, 0, num_layers))
        local_gflops = float(layer_gflops[:partition].sum())
        remote_gflops = float(layer_gflops[partition:].sum())
        speed_factor = self._precision_speed_factor(action.quant_bits)

        local_start = max(arrival, source.busy_until) if local_gflops > 0 else arrival
        local_exec = source.execution_time(local_gflops, speed_factor) if local_gflops > 0 else 0.0
        local_finish = local_start + local_exec

        if action.target_type == "local":
            remote = None
            tx_time = 0.0
            remote_exec = 0.0
            finish = local_finish
            tx_size = 0.0
        else:
            if partition == 0:
                tx_size = task["input_size_mb"]
            else:
                activation_index = min(max(partition - 1, 0), num_layers - 1)
                tx_size = self.profile.activation_mb[activation_index] * task["complexity"]
            # Model precision does not change the raw input size during full
            # offloading. Reduced precision affects only transmitted intermediate
            # activations when the network is partitioned.
            if partition > 0:
                tx_size *= action.quant_bits / 32.0
            link = self.link_device_edge if action.target_type == "edge" else self.link_device_cloud
            tx_time = link.transmission_time(tx_size, bw_factor, latency_factor)
            remote = self.edges[action.target_index or 0] if action.target_type == "edge" else self.cloud
            remote_exec = remote.execution_time(remote_gflops, speed_factor)
            remote_start = max(local_finish + tx_time, remote.busy_until)
            finish = remote_start + remote_exec

        latency = finish - arrival
        device_energy = source.power_active_w * local_exec + self.params.device_tx_power_w * tx_time
        split_compression = 0 < partition < num_layers
        accuracy = accuracy_retention(self.profile.base_accuracy, action.quant_bits, split_compression)
        violation = float(latency > task["deadline_s"])
        infeasible_energy = float(source.battery_j is not None and device_energy > source.battery_j)
        return {
            "partition": float(partition),
            "latency_s": latency,
            "device_energy_j": device_energy,
            "accuracy": accuracy,
            "violation": violation,
            "infeasible_energy": infeasible_energy,
            "local_exec_s": local_exec,
            "local_finish_s": local_finish,
            "remote_exec_s": remote_exec,
            "queue_wait_s": max(0.0, latency - local_exec - tx_time - remote_exec),
            "finish_s": finish,
            "tx_time_s": tx_time,
            "tx_size_mb": tx_size,
            "bandwidth_mbps": (self.params.bw_device_edge_mbps if action.target_type == "edge" else self.params.bw_device_cloud_mbps) * bw_factor if action.target_type != "local" else 0.0,
        }

    def _commit_action(self, action_index: int, estimate: Dict[str, float]) -> None:
        assert self.current_task is not None
        action = self.actions[action_index]
        source = self.devices[int(self.current_task["source_index"])]
        local_exec = estimate["local_exec_s"]
        if local_exec > 0:
            source.busy_until = estimate["local_finish_s"]
            source.total_busy_time += local_exec
        if action.target_type != "local":
            remote = self.edges[action.target_index or 0] if action.target_type == "edge" else self.cloud
            remote.busy_until = estimate["finish_s"]
            remote.total_busy_time += estimate["remote_exec_s"]
        if source.battery_j is not None:
            source.battery_j = max(0.0, source.battery_j - estimate["device_energy_j"])
        self.last_finish_time = max(self.last_finish_time, estimate["finish_s"])

    def _get_state(self) -> np.ndarray:
        if self.current_task is None:
            return np.zeros(self.state_dim, dtype=np.float32)
        task = self.current_task
        arrival = task["arrival_time"]
        _, bw_factor, latency_factor = self._scenario_factors(arrival)
        source = self.devices[int(task["source_index"])]
        device_wait = max(0.0, source.busy_until - arrival)
        edge_waits = [max(0.0, edge.busy_until - arrival) for edge in self.edges]
        cloud_wait = max(0.0, self.cloud.busy_until - arrival)
        batteries = [d.battery_ratio() for d in self.devices]
        recent_rate = float(np.mean(self.recent_violations[-20:])) if self.recent_violations else 0.0
        values = [
            task["input_size_mb"] / 1.6,
            task["total_gflops"] / max(self.profile.base_gflops * 1.35, 1e-8),
            task["deadline_s"] / 1.3,
            device_wait / 2.0,
            *[wait / 2.0 for wait in edge_waits],
            cloud_wait / 2.0,
            bw_factor / 2.0,
            latency_factor / 2.0,
            *batteries,
            recent_rate,
            arrival / 60.0,
        ]
        return np.asarray(values, dtype=np.float32)

    @property
    def state_dim(self) -> int:
        return 3 + 1 + self.params.num_edge_servers + 1 + 2 + self.params.num_devices + 2

    def step(self, action_index: int) -> Tuple[np.ndarray, float, bool, Dict[str, float]]:
        if self.current_task is None:
            return np.zeros(self.state_dim, dtype=np.float32), 0.0, True, {}
        estimate = self.estimate_action(action_index)
        task = dict(self.current_task)
        weights = self._weights()
        latency_ratio = min(estimate["latency_s"] / max(task["deadline_s"], 1e-6), 4.0)
        energy_norm = min(estimate["device_energy_j"] / 2.0, 4.0)
        quality_loss = (1.0 - estimate["accuracy"]) * 10.0
        reward = -(
            weights[0] * latency_ratio
            + weights[1] * energy_norm
            + weights[2] * quality_loss
            + 0.75 * estimate["violation"]
            + 1.25 * estimate["infeasible_energy"]
        )
        self._commit_action(action_index, estimate)
        self.recent_violations.append(int(estimate["violation"]))
        action = self.actions[action_index]
        self.metrics["latency_ms"].append(estimate["latency_s"] * 1000.0)
        self.metrics["device_energy_j"].append(estimate["device_energy_j"])
        self.metrics["accuracy"].append(estimate["accuracy"])
        self.metrics["violation"].append(estimate["violation"])
        self.metrics["partition_layer"].append(estimate["partition"])
        self.metrics["quant_bits"].append(float(action.quant_bits))
        self.metrics["action"].append(float(action_index))
        self.metrics["arrival_time_s"].append(task["arrival_time"])
        self.metrics["bandwidth_mbps"].append(estimate["bandwidth_mbps"])
        self.metrics["reward"].append(reward)

        exhausted = all((d.battery_j or 0.0) <= 0.0 for d in self.devices)
        done = self.task_index >= self.params.max_tasks or exhausted
        self.current_task = None if done else self._generate_next_task()
        next_state = np.zeros(self.state_dim, dtype=np.float32) if done else self._get_state()
        info = {**estimate, "weights_latency": weights[0], "weights_energy": weights[1], "weights_accuracy": weights[2]}
        return next_state, float(reward), done, info

    def summary(self) -> Dict[str, float]:
        count = len(self.metrics["latency_ms"])
        horizon = max(self.last_finish_time, self.current_time, 1e-6)
        compute_nodes = [*self.devices, *self.edges, self.cloud]
        utilization = float(np.mean([min(n.total_busy_time / horizon, 1.0) for n in compute_nodes]))
        depletion = 1.0 - float(np.mean([d.battery_ratio() for d in self.devices]))
        return {
            "tasks": float(count),
            "latency_ms": float(np.mean(self.metrics["latency_ms"])) if count else np.nan,
            "device_energy_j": float(np.mean(self.metrics["device_energy_j"])) if count else np.nan,
            "accuracy": float(np.mean(self.metrics["accuracy"])) if count else np.nan,
            "violation_rate": float(np.mean(self.metrics["violation"])) if count else np.nan,
            "utilization": utilization,
            "depletion": depletion,
            "mean_reward": float(np.mean(self.metrics["reward"])) if count else np.nan,
        }


class ActorCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
        )
        self.actor = nn.Linear(128, action_dim)
        self.critic = nn.Linear(128, 1)

    def forward(self, states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(states)
        return self.actor(features), self.critic(features).squeeze(-1)


class PPOAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_ratio: float = 0.20,
        update_epochs: int = 6,
        entropy_coef: float = 0.01,
        value_coef: float = 0.50,
        seed: int = 42,
    ) -> None:
        set_global_seed(seed)
        self.policy = ActorCritic(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.update_epochs = update_epochs
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef

    @torch.no_grad()
    def select_action(self, state: np.ndarray, deterministic: bool = False) -> Tuple[int, float, float]:
        tensor = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
        logits, value = self.policy(tensor)
        distribution = Categorical(logits=logits)
        action = torch.argmax(logits, dim=-1) if deterministic else distribution.sample()
        log_prob = distribution.log_prob(action)
        return int(action.item()), float(log_prob.item()), float(value.item())

    def update(self, rollout: Sequence[Tuple[np.ndarray, int, float, bool, float, float]]) -> Dict[str, float]:
        if not rollout:
            return {"loss": 0.0}
        states = torch.as_tensor(np.asarray([x[0] for x in rollout]), dtype=torch.float32)
        actions = torch.as_tensor([x[1] for x in rollout], dtype=torch.long)
        rewards = np.asarray([x[2] for x in rollout], dtype=np.float32)
        dones = np.asarray([x[3] for x in rollout], dtype=np.float32)
        old_log_probs = torch.as_tensor([x[4] for x in rollout], dtype=torch.float32)
        values_np = np.asarray([x[5] for x in rollout], dtype=np.float32)

        advantages = np.zeros_like(rewards)
        gae = 0.0
        next_value = 0.0
        for t in reversed(range(len(rewards))):
            non_terminal = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * next_value * non_terminal - values_np[t]
            gae = delta + self.gamma * self.gae_lambda * non_terminal * gae
            advantages[t] = gae
            next_value = values_np[t]
        returns = advantages + values_np
        advantages_t = torch.as_tensor(advantages, dtype=torch.float32)
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std(unbiased=False) + 1e-8)
        returns_t = torch.as_tensor(returns, dtype=torch.float32)

        last_loss = 0.0
        for _ in range(self.update_epochs):
            logits, values = self.policy(states)
            distribution = Categorical(logits=logits)
            new_log_probs = distribution.log_prob(actions)
            entropy = distribution.entropy().mean()
            ratio = torch.exp(new_log_probs - old_log_probs)
            unclipped = ratio * advantages_t
            clipped = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * advantages_t
            policy_loss = -torch.min(unclipped, clipped).mean()
            value_loss = torch.mean((values - returns_t) ** 2)
            loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()
            last_loss = float(loss.item())
        return {"loss": last_loss}


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
