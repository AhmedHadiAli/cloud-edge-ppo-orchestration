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


