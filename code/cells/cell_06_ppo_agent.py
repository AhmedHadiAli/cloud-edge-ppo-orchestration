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


