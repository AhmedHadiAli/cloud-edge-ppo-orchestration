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


