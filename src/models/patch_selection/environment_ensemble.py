import numpy as np
import torch

import gymnasium as gym
from gymnasium import spaces

from src.utils.device import resolve_device


class WaferMapGridEnsemble(gym.Env):
    """Inference environment for an ensemble of KV-cached causal Transformers.

    Same interface as ``WaferMapGrid``, but ``classifier`` is a list of classifiers. Every member receives
    the same acquired patch. ``step`` returns the members' averaged sigmoid probabilities (not logits)
    and a list with one observation per member, so that each member's actor gets its own state vector.
    The reward is the members' average sparse reward (exact match minus ``step_penalty``).
    """

    metadata = {"render_modes": []}

    def __init__(self, classifier, train_loader=None, test_loader=None, start_policy=0, reward_fun="sparse",
                 step_penalty=0.005, device=None):
        super().__init__()

        if reward_fun != "sparse":
            raise ValueError("WaferMapGridEnsemble only supports reward_fun='sparse'.")

        self.device = resolve_device(device)

        self.classifier = classifier
        self.num_models = len(classifier)
        self.train_loader = train_loader
        self.test_loader = test_loader

        self.patches = None        # [B, S, input_dim]
        self.targets = None        # [B, C] multi-hot labels
        self.state = None          # list of key/value caches, one per member

        self.curr_time = 0
        self.start_policy = start_policy
        self.reward_fun = reward_fun
        self.step_penalty = step_penalty

        self.action_space = spaces.Discrete(classifier[0].seq_len)
        self.action_mask = None    # [B, S], 1 = not yet acquired

        obs_dim = getattr(classifier[0], "embed_dim", classifier[0].hidden_dim)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

    def reward_func(self, y_pred):
        """Sparse reward [B] from one member's logits [B, C]."""
        pred = (torch.sigmoid(y_pred) >= 0.5).to(self.targets.dtype)
        return (pred == self.targets).all(dim=-1).float() - self.step_penalty

    def step(self, actions, compute_reward=True, mode="train"):
        """Acquire patch ``actions[b]`` for every wafer b with every member.

        Returns
        -------
        averaged probabilities [B, C], list of observations [B, embed_dim], averaged reward [B] (or None),
        done, None
        """
        if self.patches is None:
            raise RuntimeError("Call reset(patches=..., y=...) before step().")

        if not torch.is_tensor(actions):
            actions = torch.tensor(actions, device=self.device, dtype=torch.long)
        else:
            actions = actions.to(self.device).long()

        B, S, D = self.patches.shape
        idx = actions.view(B, 1, 1).expand(B, 1, D)
        inputs = self.patches.gather(dim=1, index=idx).squeeze(1)   # [B, D]

        self.action_mask.scatter_(1, actions.view(B, 1), 0.0)

        probs_avg, reward_avg, feats = 0, 0, []
        for i in range(self.num_models):
            with torch.no_grad():
                logits, feat, _, new_state, _ = self.classifier[i].make_step(
                    inputs, patch_id=actions, state=self.state[i], detach_state=False)

            if compute_reward:
                reward_avg += self.reward_func(logits)

            self.state[i] = new_state
            feats.append(feat)
            probs_avg += torch.sigmoid(logits)

        probs_avg /= self.num_models
        reward = reward_avg / self.num_models if compute_reward else None

        self.curr_time += 1
        done = self.curr_time == self.classifier[0].seq_len

        return probs_avg, feats, reward, done, None

    def reset(self, patches=None, y=None, seed=0, options=None):
        """Start an episode; returns (None, zero observations [num_models, B, embed_dim])."""
        super().reset(seed=seed)

        self.patches = patches.to(self.device)
        self.targets = y.to(self.device)

        B, S = self.patches.size(0), self.patches.size(1)
        self.state = [self.classifier[i].init_state(batch_size=B, device=self.device)
                      for i in range(self.num_models)]
        self.action_mask = torch.ones((B, S), device=self.device)
        self.curr_time = 0

        obs_dim = getattr(self.classifier[0], "embed_dim", self.classifier[0].hidden_dim)
        return None, torch.zeros(self.num_models, B, obs_dim, device=self.device)

    def render(self):
        pass
