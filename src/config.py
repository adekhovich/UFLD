"""Shared model configurations and checkpoint naming used by all notebooks."""

# Transformer configurations (Table 1): embedding_dim, number of Transformer blocks
MODEL_CONFIGS = {
    'small': (128, 8),
    'medium': (256, 4),
    'large': (256, 8),
}


def classifier_checkpoint(dataset_name, seed, num_layers, hidden_dim, num_heads=4, num_output_heads=64,
                          c=0.1, classifier_name='transformer', models_dir='models'):
    """Path of a pretrained causal Transformer checkpoint (written by 0-pretrain.ipynb)."""
    return (f'{models_dir}/{dataset_name}_c{c}_supcon{classifier_name}_sch_{num_output_heads}outs_'
            f'{num_heads}heads_{num_layers}layers{hidden_dim}_clip0.5_seed{seed}.pth')


def ppo_model_name(dataset_name, num_layers, hidden_dim, num_heads=4, num_output_heads=64, c=0.1,
                   reward_fun='sparse', classifier_name='transformer'):
    """Base name of the PPO actor/critic checkpoints (written by 1-rl.ipynb).

    The agent appends ``_lr{lr}_gamma{gamma}_gae{gae_lambda}_seed{seed}.pt`` to this name.
    """
    return (f'ppo_bs16_{dataset_name}_c{c}_supcon{classifier_name}_sch_r{reward_fun}0.005_'
            f'{num_output_heads}outs_{num_heads}heads_{num_layers}layers{hidden_dim}_clip0.5')
