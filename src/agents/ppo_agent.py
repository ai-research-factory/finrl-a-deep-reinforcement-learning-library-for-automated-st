"""PPO agent wrapper using Stable-Baselines3."""

from pathlib import Path

from stable_baselines3 import PPO


class PPOAgent:
    """Wrapper around SB3's PPO for stock trading."""

    def __init__(self, env, **kwargs):
        """Initialize PPO agent.

        Args:
            env: Gymnasium environment instance.
            **kwargs: Additional arguments passed to SB3 PPO.
        """
        default_kwargs = {
            "policy": "MlpPolicy",
            "learning_rate": 3e-4,
            "n_steps": 2048,
            "batch_size": 64,
            "n_epochs": 10,
            "gamma": 0.99,
            "verbose": 1,
        }
        default_kwargs.update(kwargs)
        self.model = PPO(env=env, **default_kwargs)

    def train(self, total_timesteps: int = 100_000):
        """Train the agent."""
        self.model.learn(total_timesteps=total_timesteps)

    def predict(self, observation, deterministic: bool = True):
        """Predict action given observation."""
        action, _states = self.model.predict(observation, deterministic=deterministic)
        return action

    def save(self, path: str):
        """Save the trained model."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.model.save(path)

    def load(self, path: str, env=None):
        """Load a trained model."""
        self.model = PPO.load(path, env=env)
