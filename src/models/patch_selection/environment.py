import numpy as np
import torch
import torch.nn.functional as F

import gymnasium as gym
from gymnasium import spaces

from src.utils.device import resolve_device


class WaferMapGrid(gym.Env):
    """Batched environment for sequential patch acquisition with a KV-cached causal Transformer.

    A batch of wafer maps is given to ``reset`` as patch sequences [B, S, input_dim]. Each call to ``step``
    acquires one patch per wafer (the action is the patch id), feeds it to the classifier and returns the
    new prediction, the classifier state vector (the observation for the policy) and the reward.
    An episode always lasts ``classifier.seq_len`` steps.

    Rewards (multi-label targets):
      ``"sparse"``     1 - step_penalty if all labels are predicted correctly, else -step_penalty
      ``"delta_bce"``  decrease of the binary cross-entropy since the previous step
      ``"combined"``   sum of both
    """

    metadata = {"render_modes": []}

    def __init__(self, classifier, train_loader=None, test_loader=None, start_policy=0, reward_fun="sparse",
                 step_penalty=0.005, device=None):
        super().__init__()

        self.device = resolve_device(device)

        self.classifier = classifier
        self.train_loader = train_loader
        self.test_loader = test_loader

        self.patches = None        # [B, S, input_dim]
        self.targets = None        # [B, C] multi-hot labels
        self.state = None          # key/value caches of the classifier
        self.prev_score = None

        self.curr_time = 0
        self.start_policy = start_policy   # transitions are used for learning only after this many steps
        self.reward_fun = reward_fun
        self.step_penalty = step_penalty

        self.action_space = spaces.Discrete(classifier.seq_len)
        self.action_mask = None    # [B, S], 1 = not yet acquired

        obs_dim = getattr(classifier, "embed_dim", classifier.hidden_dim)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

    def reward_func(self, y_pred):
        """Reward [B] from the classifier logits [B, C]."""
        reward = 0

        if self.reward_fun in ("sparse", "combined"):
            pred = (torch.sigmoid(y_pred) >= 0.5).to(self.targets.dtype)
            reward += (pred == self.targets).all(dim=-1).float() - self.step_penalty

        if self.reward_fun in ("delta_bce", "combined"):
            p = torch.sigmoid(y_pred)
            score = -F.binary_cross_entropy(p, self.targets.float(), reduction="none").sum(dim=1)
            if self.prev_score is None:
                reward = torch.zeros_like(score)
            else:
                reward += score - self.prev_score
            self.prev_score = score.detach()

        return reward

    def step(self, actions, compute_reward=True, mode="train"):
        """Acquire patch ``actions[b]`` for every wafer b.

        Returns
        -------
        logits [B, C], observation [B, embed_dim], reward [B] (or None), done, None
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

        with torch.no_grad():
            y_pred, feat, _, new_state, _ = self.classifier.make_step(
                inputs, patch_id=actions, state=self.state, detach_state=False)

        reward = self.reward_func(y_pred) if compute_reward else None

        self.state = new_state
        self.curr_time += 1
        done = self.curr_time == self.classifier.seq_len

        return y_pred, feat, reward, done, None

    def reset(self, patches=None, y=None, seed=0, options=None):
        """Start an episode for a batch of patch sequences [B, S, input_dim] with labels [B, C].

        Returns (None, initial observation): zeros [B, embed_dim], since no patch has been acquired.
        """
        super().reset(seed=seed)

        self.patches = patches.to(self.device)
        self.targets = y.to(self.device)
        self.prev_score = None

        B, S = self.patches.size(0), self.patches.size(1)
        self.state = self.classifier.init_state(batch_size=B, device=self.device)
        self.action_mask = torch.ones((B, S), device=self.device)
        self.curr_time = 0

        obs_dim = getattr(self.classifier, "embed_dim", self.classifier.hidden_dim)
        return None, torch.zeros(B, obs_dim, device=self.device)

    def render(self):
        pass
