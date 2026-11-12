"""DPO core: the loss and per-sequence log-probs. Everything else is plumbing."""
import torch
import torch.nn.functional as F


def dpo_loss(pi_chosen, pi_rejected, ref_chosen, ref_rejected, beta):
    """Eq. 7 of Rafailov et al. 2023.

    All inputs are summed log-probs of the response tokens, shape (B,).
    Returns (loss, implicit reward of chosen, implicit reward of rejected).
    """
    r_chosen = beta * (pi_chosen - ref_chosen)
    r_rejected = beta * (pi_rejected - ref_rejected)
    loss = -F.logsigmoid(r_chosen - r_rejected).mean()
    return loss, r_chosen.detach(), r_rejected.detach()


def collate(prompts, responses, pad_id):
    """Right-pad prompt+response token lists. resp_mask marks response tokens only."""
    seqs = [p + r for p, r in zip(prompts, responses)]
    L = max(map(len, seqs))
    ids = torch.full((len(seqs), L), pad_id)
    attn = torch.zeros((len(seqs), L), dtype=torch.long)
    resp = torch.zeros((len(seqs), L), dtype=torch.bool)
    for i, (p, s) in enumerate(zip(prompts, seqs)):
        ids[i, : len(s)] = torch.tensor(s)
        attn[i, : len(s)] = 1
        resp[i, len(p) : len(s)] = True
    return ids, attn, resp


def seq_logps(model, ids, attn, resp):
    """Sum of log p(token | prefix) over response tokens, shape (B,)."""
    ids, attn, resp = (t.to(model.device) for t in (ids, attn, resp))
    logits = model(input_ids=ids, attention_mask=attn).logits[:, :-1]
    logp = torch.log_softmax(logits.float(), -1).gather(-1, ids[:, 1:, None]).squeeze(-1)
    return (logp * resp[:, 1:]).sum(-1)
