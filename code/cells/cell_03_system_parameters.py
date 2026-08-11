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


