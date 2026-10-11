import random
from datetime import datetime

import torch

from src.utils.utils import accuracy_transformer


class TwoCropTransform:
    """Return two independently augmented views of the same image (for the contrastive loss)."""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x):
        return [self.transforms(x), self.transforms(x)]


def train_causal_transformer(model, train_loader, criterion, optimizer, num_output_heads, device,
                             supcon_loss=None, supcon_weight=0.1):
    """One training epoch of the causal Transformer on random patch orders.

    For every batch, ``num_output_heads`` random prefix lengths are sampled and the loss is averaged over
    them: binary cross-entropy (summed over labels) plus ``supcon_weight`` times the supervised contrastive
    loss between the two augmented views. With ``num_output_heads = -1`` only the full sequence is used.
    """
    model.train()
    running_loss = 0.0

    for X, y_true in train_loader:
        optimizer.zero_grad()

        if supcon_loss is None:
            X = X.to(device)
        else:
            X = torch.cat([X[0], X[1]], dim=0).to(device)   # two views: [2B, ...]
        y_true = y_true.to(device)

        if num_output_heads != -1:
            # Prediction positions (0-based index of the last acquired patch)
            train_heads = list(random.sample(range(0, model.seq_len), num_output_heads))
            loss_vec = torch.zeros(num_output_heads, dtype=torch.float64, device=device)

            if supcon_loss is None:
                y_hat_list, _, _ = model(X, prediction_index=train_heads)
                for i in range(num_output_heads):
                    loss_vec[i] = criterion(y_hat_list[i], y_true).sum(dim=1).mean()
            else:
                y_hat_list, _, projected_list = model(X, prediction_index=train_heads)
                y_true_cat = torch.cat([y_true, y_true], dim=0)   # labels of both views
                for i in range(num_output_heads):
                    proj = projected_list[i]                       # [2B, D]
                    bsz = proj.size(0) // 2
                    h1, h2 = torch.split(proj, [bsz, bsz], dim=0)
                    feats = torch.cat([h1.unsqueeze(1), h2.unsqueeze(1)], dim=1)   # [B, 2 views, D]

                    bce = criterion(y_hat_list[i], y_true_cat).sum(dim=1).mean()
                    scl = supcon_loss(features=feats, labels=y_true)
                    loss_vec[i] = bce + supcon_weight * scl

            loss = loss_vec.mean()
        else:
            y_hat, _, _ = model(X, prediction_index=-1)
            loss = criterion(y_hat, y_true).sum(dim=1).mean()

        running_loss += loss.item() * X.size(0)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer.step()

    return model, optimizer, running_loss / len(train_loader.dataset)


def training_loop_transformer(model, criterion, optimizer, scheduler, train_loader, valid_loader, epochs,
                              num_output_heads, device, supcon_loss=None, supcon_weight=0.1,
                              file_name="model.pth", print_every=1):
    """Train for ``epochs`` and save the checkpoint with the best validation accuracy to ``file_name``.

    Validation accuracy is measured after all patches (random order). With ``supcon_loss`` the training
    dataset is switched to return two augmented views per wafer.
    """
    best_acc = 0.0
    train_losses = []

    if supcon_loss is not None:
        train_loader.dataset.set_transform(TwoCropTransform(train_loader.dataset.transforms))

    for epoch in range(epochs):
        model, optimizer, train_loss = train_causal_transformer(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            num_output_heads=num_output_heads,
            device=device,
            supcon_loss=supcon_loss,
            supcon_weight=supcon_weight,
        )
        train_losses.append(train_loss)

        if scheduler is not None:
            scheduler.step()

        valid_acc, valid_f1 = accuracy_transformer(model, valid_loader, device=device)

        if valid_acc > best_acc:
            torch.save(model.state_dict(), file_name)
            best_acc = valid_acc

        if epoch % print_every == (print_every - 1):
            print(
                f"{datetime.now().time().replace(microsecond=0)} --- "
                f"Epoch: {epoch}\t"
                f"Train loss: {train_loss:.4f}\t"
                f"Valid accuracy: {100 * valid_acc:.2f}\t"
                f"Valid macro-F1: {valid_f1:.3f}"
            )

    return model, train_losses
