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


