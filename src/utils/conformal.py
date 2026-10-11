"""Conformal-prediction stopping rule for UFLD.

Two-rule (per-label, Mondrian) split conformal calibration with the A/B recalibration described in the
paper (Algorithms 2-4), and evaluation with early stopping (Algorithm 1).

Non-conformity scores for label j with probability p_j:
    S0 = p_j        for negatives (y_j = 0)  -> threshold q0[j]
    S1 = 1 - p_j    for positives (y_j = 1)  -> threshold q1[j]

At every acquisition step a label is decided when exactly one of the two rules accepts it
(``p <= q0`` accepts 0, ``1 - p <= q1`` accepts 1). Acquisition stops when all labels are decided.

All functions work for a single model (``ensemble=False``: ``env.classifier`` is one classifier and
``agent.actor`` one actor) and for the inference-only ensemble (``ensemble=True``: lists of both).
"""
import math

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Subset

from src.utils.utils import img_to_patch


def make_AB_loaders(dataset, batch_size, split=0.5, shuffle=True, num_workers=4, pin_memory=True):
    """Split the calibration set into two disjoint parts A and B and return a DataLoader for each.

    ``split`` is the fraction of the data in A. The split uses NumPy's global random state.
    """
    n = len(dataset)
    idx = np.arange(n)
    if shuffle:
        np.random.shuffle(idx)

    n_A = int(split * n)
    loader_A = DataLoader(Subset(dataset, idx[:n_A]), batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, pin_memory=pin_memory)
    loader_B = DataLoader(Subset(dataset, idx[n_A:]), batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, pin_memory=pin_memory)
    return loader_A, loader_B


def conformal_quantile(scores, alpha):
    """Finite-sample conformal quantile: the k-th smallest score with k = ceil((n + 1)(1 - alpha)).

    If k > n (fewer than about (1 - alpha) / alpha scores) the largest score is returned, so the
    finite-sample guarantee does not hold for such small sets. An empty set returns 1.0.
    """
    scores = scores.flatten()
    n = scores.numel()
    if n == 0:
        return 1.0
    k = math.ceil((n + 1) * (1.0 - alpha))
    k = max(1, min(k, n))
    return float(torch.sort(scores).values[k - 1].item())


# --------------------------------------------------------------------------- helpers

def _prepare(agent, env, ensemble, max_steps=None):
    """Put the policy and classifier(s) in eval mode; return (T, C, patch_size)."""
    if ensemble:
        for actor, classifier in zip(agent.actors, env.classifier):
            actor.eval()
            classifier.eval()
        ref = env.classifier[0]
    else:
        agent.actor.eval()
        env.classifier.eval()
        ref = env.classifier
    return max_steps or ref.seq_len, ref.num_classes, ref.patch_size


def _step_probs(agent, env, obs, ensemble):
    """One acquisition step with the deterministic policy; returns (probabilities [B, C], next observation)."""
    action, _ = agent.choose_action(obs, env.action_mask.clone(), determenistic=True)
    outs, obs, *_ = env.step(action, mode="inference")
    probs = outs if ensemble else torch.sigmoid(outs)   # the ensemble env already returns probabilities
    return probs, obs


def _label_thresholds(P, Y, alpha, device):
    """Per-label thresholds q0, q1 [C] from probabilities P [N, C] and labels Y [N, C]."""
    C = Y.shape[1]
    q0 = torch.zeros(C, device=device)
    q1 = torch.zeros(C, device=device)
    for c in range(C):
        neg = Y[:, c] == 0
        pos = Y[:, c] == 1
        q0[c] = conformal_quantile(P[neg, c].to(device), alpha) if neg.any() else 1.0
        q1[c] = conformal_quantile((1.0 - P[pos, c]).to(device), alpha) if pos.any() else 1.0
    return q0, q1


def _rollout_with_stopping(agent, env, loader, q0, q1, device, m=4, min_steps=1, patience=1, max_steps=None,
                           break_ties_during_run=False, ensemble=False, cnn_model=None):
    """Run acquisition with the two-rule stopping criterion on every wafer in ``loader``.

    A wafer stops at the first step where all labels are decided, at most ``m`` labels are positive,
    at least ``min_steps`` patches were acquired and the decisions were unchanged for ``patience`` steps.
    Wafers that never stop use the probabilities after all ``T`` patches, or those of ``cnn_model``
    (hybrid setting) if given, thresholded at 0.5.

    Returns
    -------
    dict with numpy arrays Y_true [N, C], Y_pred [N, C] (bool), P_stop [N, C] (probabilities used at the
    stopping time), steps_used [N], and the number of unresolved wafers.
    """
    T, C, patch_size = _prepare(agent, env, ensemble, max_steps)
    thr0 = q0.to(device).unsqueeze(0)   # [1, C]
    thr1 = q1.to(device).unsqueeze(0)

    all_true, all_pred, all_probs, steps_used = [], [], [], []
    unresolved_count = 0

    for X, Y in loader:
        X = X.to(device)
        Y = Y.to(device).float()
        patches = img_to_patch(X, patch_size=patch_size)
        B = patches.size(0)

        _, obs = env.reset(patches, Y)

        preds_out = torch.zeros(B, C, dtype=torch.bool, device=device)
        probs_at_stop = torch.zeros(B, C, device=device)
        finished = torch.zeros(B, dtype=torch.bool, device=device)
        steps = torch.full((B,), T, device=device)
        stable = torch.zeros(B, dtype=torch.long, device=device)
        prev_decided, prev_preds = None, None
        last_probs = None

        for t in range(T):
            probs, obs = _step_probs(agent, env, obs, ensemble)
            last_probs = probs

            inc0 = probs <= thr0                # rule for label 0 accepts
            inc1 = (1.0 - probs) <= thr1        # rule for label 1 accepts
            tie = inc0 & inc1

            decide1 = inc1 & ~inc0
            decide0 = inc0 & ~inc1
            if break_ties_during_run:
                decide1 = decide1 | (tie & (probs >= 0.5))
                decide0 = decide0 | (tie & (probs < 0.5))

            decided = decide0 | decide1
            preds_t = decide1

            # Number of consecutive steps with unchanged decisions
            if prev_decided is None:
                stable += 1
            else:
                same_decided = (decided == prev_decided).all(dim=1)
                both = decided & prev_decided
                same_values = ~((preds_t != prev_preds) & both).any(dim=1)
                stable = torch.where(same_decided & same_values, stable + 1, torch.ones_like(stable))
            prev_decided, prev_preds = decided.clone(), preds_t.clone()

            stop = decided.all(dim=1) & (preds_t.sum(dim=1) <= m) & ((t + 1) >= min_steps) & (stable >= patience)

            newly_done = ~finished & stop
            if newly_done.any():
                finished[newly_done] = True
                steps[newly_done] = t + 1
                preds_out[newly_done] = preds_t[newly_done]
                probs_at_stop[newly_done] = probs[newly_done]

            if finished.all():
                break

        if (~finished).any():
            idx = torch.where(~finished)[0]
            unresolved_count += idx.numel()
            probs = last_probs[idx] if cnn_model is None else torch.sigmoid(cnn_model(X[idx]))
            preds_out[idx] = probs >= 0.5
            probs_at_stop[idx] = probs
            steps[idx] = T

        all_true.append(Y.cpu().numpy())
        all_pred.append(preds_out.cpu().numpy())
        all_probs.append(probs_at_stop.cpu().numpy())
        steps_used.extend(steps.cpu().tolist())

    return dict(
        Y_true=np.concatenate(all_true, axis=0),
        Y_pred=np.concatenate(all_pred, axis=0),
        P_stop=np.concatenate(all_probs, axis=0),
        steps_used=np.array(steps_used),
        unresolved_count=unresolved_count,
    )


# --------------------------------------------------------------------------- calibration

@torch.no_grad()
def calibrate_multilabel_CP_two_rule(agent, env, calib_loader, alpha=0.05, device="cuda", ensemble=False,
                                     verbose=False):
    """Stage A (Algorithm 3): thresholds from the scores after all T patches have been acquired."""
    T, C, patch_size = _prepare(agent, env, ensemble)

    P_list, Y_list = [], []
    for X, Y in calib_loader:
        X = X.to(device)
        Y = Y.to(device).float()
        _, obs = env.reset(img_to_patch(X, patch_size=patch_size), Y)

        for _ in range(T):
            probs, obs = _step_probs(agent, env, obs, ensemble)

        P_list.append(probs.cpu())
        Y_list.append(Y.cpu())

    q0, q1 = _label_thresholds(torch.cat(P_list), torch.cat(Y_list), alpha, device)
    if verbose:
        print(f"[CP stage A] mean q0 = {q0.mean().item():.4f}, mean q1 = {q1.mean().item():.4f}")
    return q0, q1


@torch.no_grad()
def recalibrate_multilabel_CP_two_rule_on_B(agent, env, calib_loader_B, q0_init, q1_init, alpha=0.05,
                                            device="cuda", m=4, min_steps=1, patience=1, max_steps=None,
                                            break_ties_during_run=False, ensemble=False, verbose=False):
    """Stage B (Algorithm 4): thresholds from the scores at the stopping times induced by q0_init, q1_init."""
    res = _rollout_with_stopping(agent, env, calib_loader_B, q0_init, q1_init, device, m=m, min_steps=min_steps,
                                 patience=patience, max_steps=max_steps,
                                 break_ties_during_run=break_ties_during_run, ensemble=ensemble)
    q0, q1 = _label_thresholds(torch.from_numpy(res["P_stop"]), torch.from_numpy(res["Y_true"]), alpha, device)
    if verbose:
        print(f"[CP stage B] mean q0 = {q0.mean().item():.4f}, mean q1 = {q1.mean().item():.4f}")
    return q0, q1


@torch.no_grad()
def calibrate_multilabel_CP_two_rule_AB(agent, env, calib_loader_A, calib_loader_B, alpha=0.05, device="cuda",
                                        m=4, min_steps=1, patience=1, max_steps=None,
                                        break_ties_during_run=False, ensemble=False, verbose=False):
    """Full A/B calibration (Algorithm 2). Returns q0_A, q1_A (preliminary) and q0_B, q1_B (final)."""
    q0_A, q1_A = calibrate_multilabel_CP_two_rule(agent, env, calib_loader_A, alpha=alpha, device=device,
                                                  ensemble=ensemble, verbose=verbose)
    q0_B, q1_B = recalibrate_multilabel_CP_two_rule_on_B(
        agent, env, calib_loader_B, q0_A, q1_A, alpha=alpha, device=device, m=m, min_steps=min_steps,
        patience=patience, max_steps=max_steps, break_ties_during_run=break_ties_during_run,
        ensemble=ensemble, verbose=verbose)
    return q0_A, q1_A, q0_B, q1_B


# --------------------------------------------------------------------------- evaluation

@torch.no_grad()
def evaluate_multilabel_CP_two_rule(agent, env, test_loader, q0, q1, cnn_model=None, device="cuda", m=4,
                                    min_steps=1, patience=1, max_steps=None, break_ties_during_run=False,
                                    ensemble=False, verbose=True):
    """Evaluate adaptive acquisition with the conformal stopping rule (Algorithm 1).

    With ``cnn_model`` (hybrid setting) wafers that never stop are classified by the CNN on the full map.
    Coverage is the micro-averaged fraction of label scores below their threshold at the stopping time.
    """
    if cnn_model is not None:
        cnn_model.eval()

    res = _rollout_with_stopping(agent, env, test_loader, q0, q1, device, m=m, min_steps=min_steps,
                                 patience=patience, max_steps=max_steps,
                                 break_ties_during_run=break_ties_during_run, ensemble=ensemble,
                                 cnn_model=cnn_model)
    Y_true, Y_pred, P_stop, steps_used = res["Y_true"], res["Y_pred"], res["P_stop"], res["steps_used"]

    subset_acc = 100 * (Y_true == Y_pred).all(axis=1).mean()
    micro_f1 = f1_score(Y_true.reshape(-1), Y_pred.reshape(-1), average="micro", zero_division=0)
    macro_f1 = f1_score(Y_true, Y_pred, average="macro", zero_division=0)
    avg_steps = steps_used.mean()
    avg_set_size = Y_pred.sum(axis=1).mean()

    q0_eval = q0.cpu().numpy()[None, :]
    q1_eval = q1.cpu().numpy()[None, :]
    neg_mask = Y_true == 0
    pos_mask = Y_true == 1
    cov_neg = ((P_stop <= q0_eval) & neg_mask).sum() / neg_mask.sum()
    cov_pos = (((1 - P_stop) <= q1_eval) & pos_mask).sum() / pos_mask.sum()
    coverage = (cov_neg * neg_mask.sum() + cov_pos * pos_mask.sum()) / Y_true.size

    if verbose:
        print(f"[CP eval] accuracy {subset_acc:.2f}%  micro-F1 {micro_f1:.3f}  macro-F1 {macro_f1:.3f}  "
              f"coverage {coverage:.3f}  avg steps {avg_steps:.1f}  unresolved {res['unresolved_count']}"
              + ("  (hybrid)" if cnn_model is not None else ""))

    return dict(
        subset_acc=subset_acc,
        micro_f1=micro_f1,
        macro_f1=macro_f1,
        coverage=coverage,
        coverage_neg=cov_neg,
        coverage_pos=cov_pos,
        avg_steps=avg_steps,
        avg_set_size=avg_set_size,
        unresolved_count=res["unresolved_count"],
        Y_true=Y_true,
        Y_pred=Y_pred,
        steps_used=steps_used,
    )
