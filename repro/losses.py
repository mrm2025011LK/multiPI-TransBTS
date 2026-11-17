"""Training loss.

`official_loss` reproduces trainer.py of the released code exactly:

    per head k in {WT, TC, ET}:  L_k = Dice(p_k, y_k) + BCE(p_k, y_k) / batch_size
    L = 0.3 * (L_WT + L_TC + L_ET)

(The paper's Eq. 24 has unit weights and no division of BCE by the batch size;
the released code is what produced the released numbers, so it is the baseline.)

The proposed variant changes one thing: the overlap term of the ET head becomes a
Tversky index with false-negative weight beta. With beta = 0.5 the Tversky index
equals the Dice coefficient, so beta = 0.5 is the baseline and beta > 0.5 trades
precision for sensitivity on the enhancing-tumour head only.
"""
import torch
import torch.nn.functional as F

EPS = 1e-7  # same constant as kits/losses.py::BinaryDiceLoss


def tversky_loss(prob, target, beta=0.5):
    """1 - mean_over_samples( TP / (TP + (1-beta) FP + beta FN) ).

    For beta = 0.5 this is algebraically kits.losses.BinaryDiceLoss:
        2 * sum(p*y) / (sum(p) + sum(y) + eps)
    """
    n = target.size(0)
    p = prob.reshape(n, -1)
    y = target.reshape(n, -1)
    tp = (p * y).sum(1)
    fp = (p * (1 - y)).sum(1)
    fn = ((1 - p) * y).sum(1)
    index = tp / (tp + (1 - beta) * fp + beta * fn + EPS / 2)
    return 1 - index.sum() / n


def official_loss(logit_wt, logit_tc, logit_et, target, et_beta=0.5, head_weight=0.3):
    """target: (B, 3, H, W) float with channels WT, TC, ET."""
    total = 0.0
    parts = {}
    for k, (name, logit) in enumerate((("WT", logit_wt), ("TC", logit_tc), ("ET", logit_et))):
        prob = torch.sigmoid(logit.float())
        y = target[:, k:k + 1]
        beta = et_beta if name == "ET" else 0.5
        overlap = tversky_loss(prob, y, beta)
        bce = F.binary_cross_entropy(prob, y, reduction="mean") / prob.size(0)
        parts[name] = (overlap + bce).detach()
        total = total + head_weight * (overlap + bce)
    return total, parts
