#!/usr/bin/env python3
"""
Train a small Neural Collaborative Filtering (NCF) recommender and save:
  - full training checkpoint (.pth) with model + optimizer + meta
  - pure state_dict (.pth)
  - optional torchscript is NOT used (Java path reads ZIP pickle)

Usage:
  python3 scripts/train_ncf_recsys.py --out-dir /tmp/ncf_run --epochs 5
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class NCF(nn.Module):
    """Neural Collaborative Filtering: GMF + MLP towers, fused prediction."""

    def __init__(self, n_users: int, n_items: int, emb_dim: int = 32, mlp_dims=(64, 32, 16)):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.emb_dim = emb_dim

        # GMF embeddings
        self.user_gmf = nn.Embedding(n_users, emb_dim)
        self.item_gmf = nn.Embedding(n_items, emb_dim)
        # MLP embeddings
        self.user_mlp = nn.Embedding(n_users, emb_dim)
        self.item_mlp = nn.Embedding(n_items, emb_dim)

        layers = []
        in_dim = emb_dim * 2
        for d in mlp_dims:
            layers += [nn.Linear(in_dim, d), nn.ReLU(), nn.Dropout(0.1)]
            in_dim = d
        self.mlp = nn.Sequential(*layers)
        self.predict = nn.Linear(emb_dim + in_dim, 1)

        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.01)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, user: torch.Tensor, item: torch.Tensor) -> torch.Tensor:
        # GMF path
        ug = self.user_gmf(user)
        ig = self.item_gmf(item)
        gmf = ug * ig
        # MLP path
        um = self.user_mlp(user)
        im = self.item_mlp(item)
        mlp_in = torch.cat([um, im], dim=-1)
        mlp_out = self.mlp(mlp_in)
        fused = torch.cat([gmf, mlp_out], dim=-1)
        return self.predict(fused).squeeze(-1)

    def recommend(self, user_id: int, k: int = 10, exclude: set[int] | None = None) -> list[int]:
        self.eval()
        with torch.no_grad():
            u = torch.full((self.n_items,), user_id, dtype=torch.long)
            i = torch.arange(self.n_items, dtype=torch.long)
            scores = self.forward(u, i)
            if exclude:
                for it in exclude:
                    if 0 <= it < self.n_items:
                        scores[it] = -1e9
            topk = torch.topk(scores, k=min(k, self.n_items)).indices.tolist()
            return topk


def synth_interactions(n_users: int, n_items: int, n_pos: int, seed: int = 42):
    rng = random.Random(seed)
    # each user likes a latent cluster of items
    clusters = 8
    user_c = [rng.randrange(clusters) for _ in range(n_users)]
    item_c = [rng.randrange(clusters) for _ in range(n_items)]
    pos = set()
    while len(pos) < n_pos:
        u = rng.randrange(n_users)
        # prefer same-cluster items
        if rng.random() < 0.7:
            candidates = [i for i, c in enumerate(item_c) if c == user_c[u]]
            if not candidates:
                candidates = list(range(n_items))
            i = rng.choice(candidates)
        else:
            i = rng.randrange(n_items)
        pos.add((u, i))
    return list(pos)


def train(args):
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    n_users, n_items = args.users, args.items
    pos = synth_interactions(n_users, n_items, args.positives, args.seed)
    model = NCF(n_users, n_items, emb_dim=args.emb, mlp_dims=tuple(args.mlp))
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    print("=== NCF architecture ===")
    print(model)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"parameters: {n_params}")
    print("named_parameters:")
    for name, p in model.named_parameters():
        print(f"  {name:40s} {tuple(p.shape)} {p.dtype}")

    history = []
    model.train()
    rng = random.Random(args.seed + 7)
    for epoch in range(1, args.epochs + 1):
        total_loss = 0.0
        steps = args.steps
        for _ in range(steps):
            batch = [pos[rng.randrange(len(pos))] for _ in range(args.batch)]
            users = torch.tensor([u for u, _ in batch], dtype=torch.long)
            pos_items = torch.tensor([i for _, i in batch], dtype=torch.long)
            neg_items = torch.tensor([rng.randrange(n_items) for _ in batch], dtype=torch.long)

            pos_logits = model(users, pos_items)
            neg_logits = model(users, neg_items)
            # BPR-style pairwise loss
            loss = -F.logsigmoid(pos_logits - neg_logits).mean()

            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += float(loss.item())
        avg = total_loss / steps
        history.append({"epoch": epoch, "loss": avg})
        print(f"epoch {epoch}/{args.epochs}  loss={avg:.4f}")

    # recommendations sample
    recs = {str(u): model.recommend(u, k=5) for u in range(min(3, n_users))}
    print("sample recommendations:", recs)

    # save pure state_dict
    sd_path = out / "ncf_state_dict.pth"
    torch.save(model.state_dict(), sd_path)
    print("wrote", sd_path)

    # save full training checkpoint (tests stub tolerance for optimizer)
    ckpt_path = out / "ncf_checkpoint.pth"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": opt.state_dict(),
            "epoch": args.epochs,
            "loss": history[-1]["loss"] if history else None,
            "config": {
                "n_users": n_users,
                "n_items": n_items,
                "emb_dim": args.emb,
                "mlp_dims": list(args.mlp),
            },
            "history": history,
            "sample_recs": recs,
        },
        ckpt_path,
    )
    print("wrote", ckpt_path)

    meta = {
        "n_users": n_users,
        "n_items": n_items,
        "emb_dim": args.emb,
        "mlp_dims": list(args.mlp),
        "n_params": n_params,
        "history": history,
        "sample_recs": recs,
        "state_dict": str(sd_path),
        "checkpoint": str(ckpt_path),
    }
    (out / "ncf_meta.json").write_text(json.dumps(meta, indent=2))
    print("wrote", out / "ncf_meta.json")
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="/tmp/ncf_run")
    ap.add_argument("--users", type=int, default=200)
    ap.add_argument("--items", type=int, default=100)
    ap.add_argument("--positives", type=int, default=3000)
    ap.add_argument("--emb", type=int, default=32)
    ap.add_argument("--mlp", type=int, nargs="+", default=[64, 32, 16])
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    train(args)


if __name__ == "__main__":
    main()
