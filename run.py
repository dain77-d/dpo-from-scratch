"""IMDb controlled-sentiment DPO (the DPO paper's Sec. 6.1 setup, shrunk to run on a CPU).

1. SFT distilgpt2 on IMDb reviews            -> reference policy
2. Sample 2 completions per prompt, score with a sentiment classifier -> preference pairs
3. DPO for several beta, evaluating (reward, KL to ref) as training goes
4. Plot the reward-KL frontier, dump samples

Stages 1-2 are cached under work/. Usage: python run.py [--quick]
"""
import os

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")  # Xet downloads hung on Windows; plain HTTP is fine
import copy
import json
import random
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

from dpo import collate, dpo_loss, seq_logps

QUICK = "--quick" in sys.argv
GPU = torch.cuda.is_available() and "--cpu" not in sys.argv
DEV = "cuda" if GPU else "cpu"
CFG = dict(
    policy="distilbert/distilgpt2",
    reward="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
    sft_reviews=300 if QUICK else 2000, sft_len=64, sft_lr=5e-5,
    prompt_tokens=8, new_tokens=32,
    n_pairs=64 if QUICK else 1024, n_eval=32 if QUICK else 128,
    betas=[0.1] if QUICK else [0.1, 0.5], dpo_lr=1e-5,
    batch=16, epochs=1 if QUICK else 2, eval_every=2 if QUICK else 32, seed=0,
)
if GPU and not QUICK:  # Colab etc.: closer to the paper's scale (gpt2, more data, a beta sweep)
    CFG.update(policy="openai-community/gpt2", sft_reviews=10000, n_pairs=4096, n_eval=512,
               betas=[0.05, 0.1, 0.5, 1.0], eval_every=64)
WORK, OUT = Path("work-gpu" if GPU else "work"), Path("results/gpu" if GPU else "results")
WORK.mkdir(exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)
torch.manual_seed(CFG["seed"])
random.seed(CFG["seed"])

tok = AutoTokenizer.from_pretrained(CFG["policy"])
tok.pad_token = tok.eos_token
PAD = tok.eos_token_id
rm_tok = AutoTokenizer.from_pretrained(CFG["reward"])
rm = AutoModelForSequenceClassification.from_pretrained(CFG["reward"]).eval().to(DEV)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def imdb(split):
    path = hf_hub_download("stanfordnlp/imdb", f"plain_text/{split}-00000-of-00001.parquet", repo_type="dataset")
    texts = pd.read_parquet(path)["text"].str.replace("<br />", " ", regex=False).tolist()
    random.Random(0).shuffle(texts)
    return texts


@torch.no_grad()
def reward(prompts, responses):
    """Sentiment reward = logit(positive) - logit(negative) on the full text."""
    texts = [tok.decode(p + r) for p, r in zip(prompts, responses)]
    out = []
    for i in range(0, len(texts), 64):
        enc = rm_tok(texts[i : i + 64], return_tensors="pt", padding=True, truncation=True).to(DEV)
        logits = rm(**enc).logits
        out += (logits[:, 1] - logits[:, 0]).tolist()
    return out


@torch.no_grad()
def sample(model, prompts):
    """Pure ancestral sampling (top_k=0) so samples really come from the policy; needed for the KL estimate."""
    tok.padding_side = "left"
    out = []
    for i in range(0, len(prompts), 64):
        batch = prompts[i : i + 64]
        enc = tok.pad({"input_ids": batch}, return_tensors="pt").to(DEV)
        gen = model.generate(**enc, do_sample=True, top_k=0, top_p=1.0, max_new_tokens=CFG["new_tokens"], pad_token_id=PAD)
        for row in gen[:, enc["input_ids"].shape[1] :].tolist():
            out.append(row[: row.index(PAD)] if PAD in row else row)
    tok.padding_side = "right"
    return [r or [PAD] for r in out]  # never empty


@torch.no_grad()
def logps(model, prompts, responses):
    out = []
    for i in range(0, len(prompts), 32):
        out.append(seq_logps(model, *collate(prompts[i : i + 32], responses[i : i + 32], PAD)))
    return torch.cat(out)


def evaluate(policy, ref, prompts):
    resp = sample(policy, prompts)
    kl = (logps(policy, prompts, resp) - logps(ref, prompts, resp)).mean().item()
    return dict(reward=sum(reward(prompts, resp)) / len(resp), kl=kl), resp


# ---------- 1. SFT ----------
sft_dir = WORK / ("sft-quick" if QUICK else "sft")
if not (sft_dir / "config.json").exists():
    log("SFT on IMDb")
    model = AutoModelForCausalLM.from_pretrained(CFG["policy"]).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=CFG["sft_lr"])
    texts = imdb("unsupervised")[: CFG["sft_reviews"]]
    model.train()
    for step, i in enumerate(range(0, len(texts), CFG["batch"])):
        enc = tok(texts[i : i + CFG["batch"]], return_tensors="pt", padding=True, truncation=True, max_length=CFG["sft_len"]).to(DEV)
        labels = enc["input_ids"].masked_fill(enc["attention_mask"] == 0, -100)
        loss = model(**enc, labels=labels).loss
        loss.backward()
        opt.step()
        opt.zero_grad()
        if step % 25 == 0:
            log(f"sft step {step} loss {loss.item():.3f}")
    model.save_pretrained(sft_dir)
ref = AutoModelForCausalLM.from_pretrained(sft_dir).eval().to(DEV)

# ---------- 2. preference pairs ----------
test = imdb("test")
eval_prompts = [tok(t)["input_ids"][: CFG["prompt_tokens"]] for t in test[: CFG["n_eval"]]]
pairs_file = WORK / ("pairs-quick.pt" if QUICK else "pairs.pt")
if not pairs_file.exists():
    log("sampling preference pairs")
    prompts = [tok(t)["input_ids"][: CFG["prompt_tokens"]] for t in imdb("train")[: CFG["n_pairs"]]]
    a, b = sample(ref, prompts), sample(ref, prompts)
    ra, rb = reward(prompts, a), reward(prompts, b)
    chosen = [x if sx >= sy else y for x, y, sx, sy in zip(a, b, ra, rb)]
    rejected = [y if sx >= sy else x for x, y, sx, sy in zip(a, b, ra, rb)]
    torch.save(dict(prompts=prompts, chosen=chosen, rejected=rejected), pairs_file)
P = torch.load(pairs_file)
log(f"{len(P['prompts'])} pairs")
ref_c = logps(ref, P["prompts"], P["chosen"])  # reference log-probs never change: compute once
ref_r = logps(ref, P["prompts"], P["rejected"])

base, base_samples = evaluate(ref, ref, eval_prompts)
log(f"SFT: reward {base['reward']:.3f}")
results = dict(config=CFG, sft=base, dpo={})
samples = {"SFT": base_samples}

# ---------- 3. DPO ----------
n = len(P["prompts"])
for beta in CFG["betas"]:
    torch.manual_seed(CFG["seed"])
    policy = copy.deepcopy(ref).eval()  # dropout OFF: otherwise pi != ref at step 0 and the loss starts above log 2
    opt = torch.optim.AdamW(policy.parameters(), lr=CFG["dpo_lr"])
    curve, step = [dict(step=0, **base)], 0
    for epoch in range(CFG["epochs"]):
        order = torch.randperm(n).tolist()
        for i in range(0, n, CFG["batch"]):
            idx = order[i : i + CFG["batch"]]
            pr = [P["prompts"][j] for j in idx]
            pc = seq_logps(policy, *collate(pr, [P["chosen"][j] for j in idx], PAD))
            pj = seq_logps(policy, *collate(pr, [P["rejected"][j] for j in idx], PAD))
            loss, rc, rr = dpo_loss(pc, pj, ref_c[idx], ref_r[idx], beta)
            loss.backward()
            opt.step()
            opt.zero_grad()
            step += 1
            if step % CFG["eval_every"] == 0:
                m, resp = evaluate(policy, ref, eval_prompts)
                acc = (rc > rr).float().mean().item()
                curve.append(dict(step=step, loss=loss.item(), pref_acc=acc, **m))
                log(f"beta {beta} step {step} loss {loss.item():.3f} acc {acc:.2f} reward {m['reward']:.3f} kl {m['kl']:.2f}")
    results["dpo"][str(beta)] = curve
    samples[f"DPO beta={beta}"] = resp

(OUT / ("metrics-quick.json" if QUICK else "metrics.json")).write_text(json.dumps(results, indent=2))

# ---------- 4. report ----------
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(6, 4.2))
ax.scatter([0], [base["reward"]], c="k", marker="*", s=160, label="SFT (reference)", zorder=3)
for beta, curve in results["dpo"].items():
    ax.plot([c["kl"] for c in curve], [c["reward"] for c in curve], "o-", label=f"DPO β={beta}")
ax.set_xlabel("KL(π || π_ref)  [nats / sample]")
ax.set_ylabel("sentiment reward (logit pos − neg)")
ax.set_title("IMDb controlled sentiment: reward vs. KL")
ax.grid(alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(OUT / ("frontier-quick.png" if QUICK else "frontier.png"), dpi=150)

lines = ["| prompt | " + " | ".join(samples) + " |", "|---" * (len(samples) + 1) + "|"]
for k in range(8):
    cell = lambda s: tok.decode(s).replace("|", "\\|").replace("\n", " ").strip()
    lines.append(f"| {cell(eval_prompts[k])} | " + " | ".join(cell(v[k]) for v in samples.values()) + " |")
(OUT / ("samples-quick.md" if QUICK else "samples.md")).write_text("\n".join(lines) + "\n", encoding="utf-8")
log("done")
