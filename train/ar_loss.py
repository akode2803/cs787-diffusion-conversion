"""Next-token objective for the AR control. Depends only on torch so it can be tested alone."""
import torch
import torch.nn.functional as F


def next_token_targets(labels, answer_mask, padding_mask):
    """Return (example_ids, positions, targets). The hidden state at `position` predicts `target`,
    which is the token at position + 1. Only answer tokens are targets, including the first EOS.
    EOS padding after it is not a target: an AR model stops at the first EOS and never reaches it."""
    is_target = (answer_mask & ~padding_mask)[:, 1:]
    example_ids, positions = is_target.nonzero(as_tuple=True)
    return example_ids, positions, labels[:, 1:][example_ids, positions]


def per_example_mean_ce(logits, targets, example_ids, batch_size):
    """Cross-entropy averaged within each example, then across examples.
    BYOD also normalises per example by answer length, so long answers do not dominate a batch."""
    ce = F.cross_entropy(logits, targets, reduction="none")
    sums = torch.zeros(batch_size, device=ce.device, dtype=ce.dtype).scatter_add(0, example_ids, ce)
    counts = torch.zeros(batch_size, device=ce.device, dtype=ce.dtype).scatter_add(0, example_ids, torch.ones_like(ce))
    valid = counts > 0
    return (sums[valid] / counts[valid]).mean(), int(valid.sum()), int(counts.sum())
