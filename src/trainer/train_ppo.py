from datetime import datetime

import torch

from src.utils.eval import eval_policy_transformer
from src.utils.utils import img_to_patch


def train_one_epoch_ppo_transformer(agent, env):
    """One pass over ``env.train_loader``: one full episode per batch of wafers, then one PPO update.

    Returns the agent, the mean loss per wafer and the mean discounted reward per wafer.
    """
    agent.actor.train()
    agent.critic.train()

    total_loss = 0.0
    total_reward = 0.0
    train_loader = env.train_loader

    for X, y in train_loader:
        X = X.to(env.device)
        y = y.to(env.device)

        patches = img_to_patch(X, patch_size=env.classifier.patch_size)   # [B, S, input_dim]
        _, observation = env.reset(patches=patches, y=y)

        done = False
        discount = 1.0
        while not done:
            with torch.no_grad():
                action_mask = env.action_mask.clone()
                action, prob, val = agent.choose_action(observation, action_mask)
                _, observation_, reward, done, _ = env.step(action)

                # Transitions are stored only after the first ``start_policy`` steps
                if env.curr_time > env.start_policy:
                    total_reward += discount * reward.sum().float()
                    agent.remember(observation, action_mask, action, prob, val, reward, done)
                    discount *= agent.gamma

                observation = observation_

        loss = agent.learn()
        agent.memory.clear_memory()
        total_loss += loss * X.size(0)

    total_loss /= len(train_loader.dataset)
    total_reward /= len(train_loader.dataset)
    return agent, total_loss, total_reward


def training_loop_ppo_transformer(agent, env, n_epochs=100):
    """Train for ``n_epochs`` and save the checkpoint with the highest evaluation reward.

    The evaluation reward is computed on ``env.test_loader`` with the deterministic policy.
    """
    best_score = -1e10

    for epoch in range(n_epochs):
        agent, train_loss, train_score = train_one_epoch_ppo_transformer(agent, env)
        eval_acc, _, eval_score, _, _ = eval_policy_transformer(agent, env, train=False)

        if eval_score > best_score:
            best_score = eval_score
            agent.save_models()

        print(
            f'{datetime.now().time().replace(microsecond=0)} --- '
            f'Epoch: {epoch}\t'
            f'Train loss: {train_loss:.4f}\t'
            f'Train reward: {train_score:.2f}\t'
            f'Eval accuracy (all patches): {eval_acc:.2f}\t'
            f'Eval reward: {eval_score:.2f}'
        )

    return agent
