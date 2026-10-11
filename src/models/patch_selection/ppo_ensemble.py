"""Inference-only ensemble of PPO patch selectors, one per seed (each trained with ``PPOAgent``)."""
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.device import resolve_device
from src.utils.masked_loss import MaskedCategorical


class ActorNetwork(nn.Module):
    """Policy network of one member: (state [B, D], action mask [B, A]) -> logits [B, A].

    Same architecture and parameter names as ``ppo.ActorNetwork``, so the members' checkpoints load directly.
    """

    def __init__(self, n_actions, obs_dim, fc1_dims=400, fc2_dims=400, chkpt_dir="models",
                 model_name="transformer", params_str="", seed=0, device=None):
        super().__init__()
        self.n_actions = n_actions
        self.obs_dim = obs_dim
        self.checkpoint_file = os.path.join(chkpt_dir, f'actor-{model_name}' + params_str + f"_seed{seed}.pt")

        self.actor = nn.Sequential(
            nn.Linear(obs_dim + n_actions, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, n_actions),
        )

        self.device = resolve_device(device)
        self.to(self.device)

    def forward(self, state, action_mask):
        return self.actor(torch.cat((state, action_mask), dim=1))

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file, map_location=self.device))


class PPOAgentEnsemble:
    """Averages the action distributions of several trained PPO actors.

    Member ``i`` receives the observation of classifier ``i`` (``observation[i]`` from
    ``WaferMapGridEnsemble``). The actors are loaded from
    ``{chkpt_dir}/actor-{model_name}_lr{alpha}_gamma{gamma}_gae{gae_lambda}_seed{seed}.pt``, the files written
    by ``PPOAgent``. ``policy_clip``, ``batch_size`` and ``n_epochs`` are accepted for interface compatibility
    with ``PPOAgent`` and are not used.
    """

    def __init__(self, n_actions, obs_dim, gamma=0.99, alpha=3e-4, gae_lambda=0.95, policy_clip=0.2,
                 batch_size=64, n_epochs=10, seeds=(0, 1, 2, 3, 4), chkpt_dir="models", model_name="transformer",
                 device=None):
        self.gamma = gamma
        self.n_actions = n_actions
        self.num_models = len(seeds)

        params_str = f"_lr{alpha}_gamma{gamma}_gae{gae_lambda}"
        self.actors = [
            ActorNetwork(n_actions=n_actions, obs_dim=obs_dim, chkpt_dir=chkpt_dir, model_name=model_name,
                         params_str=params_str, seed=seed, device=device)
            for seed in seeds
        ]
        self.device = self.actors[0].device

    def load_models(self):
        print("... loading models ...")
        for actor in self.actors:
            actor.load_checkpoint()

    @torch.no_grad()
    def choose_action(self, observation, action_mask, determenistic=False):
        """Select one patch per wafer from the members' averaged probabilities.

        observation: [num_models, B, obs_dim]; action_mask: [B, n_actions] with 1 = available.
        Returns (actions, renormalised probabilities) if ``determenistic`` (argmax), otherwise
        (sampled actions, their log-probabilities).
        """
        probs = 0
        for i in range(self.num_models):
            obs = observation[i].to(self.device)
            mask = action_mask.to(self.device)
            probs += F.softmax(self.actors[i](obs, mask), dim=-1)
        probs /= self.num_models

        if determenistic:
            probs = (probs * action_mask) / (probs * action_mask).sum(dim=-1, keepdim=True)
            actions = probs.argmax(dim=-1)
        else:
            dists = MaskedCategorical(probs=probs, mask=action_mask.to(torch.bool))
            actions = dists.sample()
            probs = dists.log_prob(actions)

        return actions, probs
