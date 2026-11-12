import math

import torch

from dpo import collate, dpo_loss


def test_loss_is_log2_when_policy_equals_ref():
    x = torch.tensor([-5.0, -7.0])
    loss, _, _ = dpo_loss(x, x - 1, x, x - 1, beta=0.1)
    assert math.isclose(loss.item(), math.log(2), rel_tol=1e-6)


def test_gradient_raises_chosen_and_lowers_rejected():
    pc = torch.tensor([-5.0], requires_grad=True)
    pr = torch.tensor([-5.0], requires_grad=True)
    loss, _, _ = dpo_loss(pc, pr, torch.tensor([-5.0]), torch.tensor([-5.0]), beta=0.5)
    loss.backward()
    assert pc.grad.item() < 0 < pr.grad.item()  # descent pushes chosen up, rejected down


def test_collate_masks_only_response():
    ids, attn, resp = collate([[1, 2], [3]], [[4], [5, 6]], pad_id=0)
    assert ids.tolist() == [[1, 2, 4], [3, 5, 6]]
    assert resp.tolist() == [[False, False, True], [False, True, True]]
    assert attn.sum().item() == 6


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("ok")
