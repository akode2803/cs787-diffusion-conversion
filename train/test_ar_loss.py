"""Run with: python3 test_ar_loss.py"""
import math
import torch
from ar_loss import next_token_targets, per_example_mean_ce


def test_targets_are_shifted_answer_tokens():
    # prompt 10 11, answer 12 13, EOS 2, then EOS padding
    labels = torch.tensor([[10, 11, 12, 13, 2, 2, 2]])
    answer = torch.tensor([[0, 0, 1, 1, 1, 0, 0]], dtype=torch.bool)
    padding = torch.tensor([[0, 0, 0, 0, 0, 1, 1]], dtype=torch.bool)
    ex, pos, tgt = next_token_targets(labels, answer, padding)
    assert ex.tolist() == [0, 0, 0]
    assert pos.tolist() == [1, 2, 3]          # last prompt token predicts the first answer token
    assert tgt.tolist() == [12, 13, 2]        # the first EOS is learned, padding is not


def test_rows_with_different_lengths():
    labels = torch.tensor([[5, 6, 2, 2], [5, 6, 7, 2]])
    answer = torch.tensor([[0, 1, 1, 0], [0, 1, 1, 1]], dtype=torch.bool)
    padding = torch.tensor([[0, 0, 0, 1], [0, 0, 0, 0]], dtype=torch.bool)
    ex, pos, tgt = next_token_targets(labels, answer, padding)
    assert ex.tolist() == [0, 0, 1, 1, 1]
    assert pos.tolist() == [0, 1, 0, 1, 2]
    assert tgt.tolist() == [6, 2, 6, 7, 2]


def test_loss_averages_per_example_first():
    # Example 0 has one target with CE ln 4; example 1 has three targets with CE 0 (nearly).
    big = 50.0
    logits = torch.tensor([[0., 0., 0., 0.], [big, 0., 0., 0.], [big, 0., 0., 0.], [big, 0., 0., 0.]])
    targets = torch.tensor([0, 0, 0, 0])
    ex = torch.tensor([0, 1, 1, 1])
    loss, valid, tokens = per_example_mean_ce(logits, targets, ex, 3)
    assert valid == 2 and tokens == 4         # the third example has no targets and is skipped
    assert abs(loss.item() - math.log(4) / 2) < 1e-5


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
