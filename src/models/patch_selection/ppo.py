"""PPO patch selector (Schulman et al., 2017) with masked actions over the unacquired patches.

Structure based on https://github.com/philtabor/Youtube-Code-Repository/blob/master/ReinforcementLearning/PolicyGradient/PPO/torch/ppo_torch.py
"""
import os
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from src.utils.device import resolve_device
from src.utils.masked_loss import MaskedCategorical


class PPOMemory:
    """Transitions of one batch of episodes; each entry holds one time step for all B wafers."""

    def __init__(self):
        self.states = []
        self.action_masks = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []

    def generate_batches(self, batch_size=1):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        batches = [indices[i:i + batch_size] for i in batch_start]

        return (self.states,
                self.action_masks,
                self.actions,
                self.probs,
                self.vals,
                self.rewards,
                self.dones,
                batches)

    def store_memory(self, state, action_mask, action, probs, vals, reward, done):
        # Stored on CPU to keep GPU memory small
        self.states.append(state.detach().cpu())            # [B,D]
        self.action_masks.append(action_mask.detach().cpu())# [B,A]
        self.actions.append(action.detach().cpu())          # [B]
        self.probs.append(probs.detach().cpu())             # [B]
        self.vals.append(vals.detach().cpu())               # [B]
        self.rewards.append(reward.detach().cpu())          # [B]
        self.dones.append(done)                             # bool

    def clear_memory(self):
        self.states = []
        self.action_masks = []
        self.probs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.vals = []


class ActorNetwork(nn.Module):
    """Policy: (state [B, D], action mask [B, A]) -> masked categorical distribution over the A patches."""
    def __init__(
        self,
        n_actions: int,
        obs_dim: int,
        lr: float,
        fc1_dims=400,
        fc2_dims=400,
        chkpt_dir="tmp/ppo",
        model_name="transformer",
        params_str="",
        device=None,
    ):
        super().__init__()
        self.n_actions = n_actions
        self.obs_dim = obs_dim

        self.checkpoint_file = os.path.join(chkpt_dir, f'actor-{model_name}' + params_str + f".pt")

        self.actor = nn.Sequential(
            nn.Linear(obs_dim + n_actions, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, n_actions),
        )

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = resolve_device(device)
        self.to(self.device)

    def forward(self, state, action_mask):
        x = torch.cat((state, action_mask), dim=1)
        logits = self.actor(x)  # [B,A]
        dists = MaskedCategorical(logits=logits, mask=action_mask.to(torch.bool))
        return dists

    def save_checkpoint(self):
        os.makedirs(os.path.dirname(self.checkpoint_file), exist_ok=True)
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file, map_location=self.device))


class CriticNetwork(nn.Module):
    """Value function: (state [B, D], action mask [B, A]) -> value [B, 1]."""
    def __init__(
        self,
        n_actions: int,
        obs_dim: int,
        lr: float,
        fc1_dims=400,
        fc2_dims=400,
        chkpt_dir="tmp/ppo",
        model_name="transformer",
        params_str="",
        device=None,
    ):
        super().__init__()
        self.n_actions = n_actions
        self.obs_dim = obs_dim

        self.checkpoint_file = os.path.join(chkpt_dir, f'critic-{model_name}' + params_str + f".pt")

        self.critic = nn.Sequential(
            nn.Linear(obs_dim + n_actions, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, 1),
        )

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = resolve_device(device)
        self.to(self.device)

    def forward(self, state, action_mask):
        x = torch.cat((state, action_mask), dim=1)
        value = self.critic(x)  # [B,1]
        return value

    def save_checkpoint(self):
        os.makedirs(os.path.dirname(self.checkpoint_file), exist_ok=True)
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file, map_location=self.device))


class PPOAgent:
    """PPO agent acting on batches of wafers.

    Observations are the classifier state vectors [B, obs_dim] (obs_dim = classifier.embed_dim); actions are
    patch ids among the unacquired patches. Checkpoints are written to
    ``{chkpt_dir}/{actor,critic}-{model_name}_lr{alpha}_gamma{gamma}_gae{gae_lambda}_seed{seed}.pt``.
    ``batch_size`` and ``n_epochs`` are stored but not used by ``learn``.
    """
    def __init__(
        self,
        n_actions: int,
        obs_dim: int,
        gamma=0.99,
        alpha=3e-4,
        gae_lambda=0.95,
        policy_clip=0.2,
        batch_size=64,
        n_epochs=10,
        seed=0,
        chkpt_dir="tmp/ppo",
        model_name="transformer",
        device=None,
    ):
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda
        self.n_actions = n_actions
        self.batch_size = batch_size

        params_str = f"_lr{alpha}_gamma{gamma}_gae{gae_lambda}_seed{seed}"

        self.actor = ActorNetwork(
            n_actions=n_actions, obs_dim=obs_dim, lr=alpha,
            chkpt_dir=chkpt_dir, model_name=model_name, params_str=params_str, device=device
        )
        self.critic = CriticNetwork(
            n_actions=n_actions, obs_dim=obs_dim, lr=alpha,
            chkpt_dir=chkpt_dir, model_name=model_name, params_str=params_str, device=device
        )

        self.memory = PPOMemory()
        self.device = self.actor.device

    def remember(self, state, action_mask, action, probs, vals, reward, done):
        self.memory.store_memory(state, action_mask, action, probs, vals, reward, done)

    def save_models(self):
        print("... saving models ...")
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()

    def load_models(self, load_critic=True):
        print("... loading models ...")
        self.actor.load_checkpoint()
        if load_critic:
            self.critic.load_checkpoint()

    @torch.no_grad()
    def choose_action(self, observation, action_mask, determenistic=False):
        """Select one patch per wafer.

        observation [B, obs_dim]; action_mask [B, n_actions] with 1 = available.
        Returns (actions, log-probs, values), or (actions, log-probs) if ``determenistic`` (argmax policy).
        """
        state = observation.to(self.device)
        mask = action_mask.to(self.device)

        dists = self.actor(state, mask)
        values = self.critic(state, mask).squeeze(-1)  # [B]

        if determenistic:
            actions = dists.mode
        else:
            actions = dists.sample()  # [B]

        log_probs = dists.log_prob(actions)  # [B]

        if determenistic:
            return actions, log_probs

        return actions, log_probs, values

    def learn(self):
        """One PPO update on the stored episode batch; returns the summed loss.

        Advantages use generalised advantage estimation over the episode (the last step gets zero advantage
        and is not used). Time steps are visited in random order and gradients are accumulated over
        micro-batches of 16 steps (each step covers all B wafers) before an optimiser step.
        Loss = clipped policy loss + 0.5 * value loss - 0.01 * entropy.
        """
        (state_arr, action_mask_arr, action_arr, old_prob_arr, vals_arr,
         reward_arr, dones_arr, batches) = self.memory.generate_batches()

        # Stack the time dimension: [T, B]
        values = torch.stack(vals_arr, dim=0).to(self.device)                    # [T,B]
        rewards = torch.stack(reward_arr, dim=0).to(self.device).float()         # [T,B]

        T, B = rewards.shape

        advantage = torch.zeros((T, B), dtype=torch.float, device=self.device)

        # Generalised advantage estimation
        for t in range(T - 1):
            discount = 1.0
            a_t = 0.0
            for k in range(t, T - 1):
                done_k = float(dones_arr[k])
                a_t = a_t + discount * (rewards[k] + self.gamma * values[k + 1] * (1.0 - done_k) - values[k])
                discount *= self.gamma * self.gae_lambda

            advantage[t] = a_t

        # The last time step has no bootstrapped advantage and is skipped
        num_timesteps = min(len(state_arr), self.n_actions - 1)

        bs = 16  # time steps per gradient update
        total_loss = 0.0

        batch_inds = np.arange(num_timesteps)
        np.random.shuffle(batch_inds)

        for ind, t in enumerate(batch_inds):
            if ind % bs == 0:
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()

            states = state_arr[t].to(self.device)          # [B,obs_dim]
            masks = action_mask_arr[t].to(self.device)     # [B,n_actions]
            old_logp = old_prob_arr[t].to(self.device)     # [B]
            actions = action_arr[t].to(self.device)        # [B]

            dist = self.actor(states, masks)
            critic_value = self.critic(states, masks).squeeze(-1)  # [B]

            new_logp = dist.log_prob(actions)              # [B]
            prob_ratio = (new_logp - old_logp).exp()       # [B]

            # Clipped PPO objective
            weighted_probs = advantage[t] * prob_ratio
            weighted_clipped_probs = torch.clamp(
                prob_ratio, 1 - self.policy_clip, 1 + self.policy_clip
            ) * advantage[t]

            actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()

            # Value target
            value_target = advantage[t] + values[t]        # [B]
            critic_loss = (value_target - critic_value).pow(2).mean()

            entropy_bonus = dist.entropy().mean()

            loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy_bonus
            loss.backward()

            total_loss += loss.item()

            if ((ind + 1) % bs == 0) or ((ind + 1) == num_timesteps):
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)
                self.actor.optimizer.step()
                self.critic.optimizer.step()

        return total_loss
