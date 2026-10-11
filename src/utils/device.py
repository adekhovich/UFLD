import torch


def resolve_device(device=None):
    """Return ``device`` as a torch.device; default to the first GPU if available, else CPU."""
    if device is None:
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(device)
