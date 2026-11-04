"""
Train the TerrainForecaster on synthetic stereo stair sequences.

Labels are produced by synthetic_terrain.TerrainSceneGenerator:

    TOI  (0..250 ms)  -> normalised /250
    COP  (0..263.2 mm) -> normalised /263.2

The model is trained with BCE-with-logits on the normalised targets, so its raw
outputs are logits. At inference we apply sigmoid and scale back to physical
units. That matches live_torque_visualizer.ForecasterEngine.

Usage:
    python train_terrain.py                       # generate 6000 seqs, 12 epochs
    python train_terrain.py --regen               # force dataset regeneration
    python train_terrain.py --sequences 2000 --epochs 5   # quick run
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from model import TerrainForecaster
from synthetic_terrain import (FOOT_LEN_MM, HEIGHT, TOI_MAX_MS, WIDTH,
                               TerrainSceneGenerator)

MID_FOOT_MM = FOOT_LEN_MM / 2.0
EMD_MS = 50.0


# ---------------------------------------------------------------- dataset I/O
def generate_dataset(path: str, sequences: int, seq_len: int, seed: int) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    gen = TerrainSceneGenerator(seed=seed, seq_len=seq_len)
    x = np.empty((sequences, seq_len, 2, HEIGHT, WIDTH), dtype=np.uint8)
    toi = np.empty((sequences, seq_len), dtype=np.float32)
    cop = np.empty((sequences, seq_len), dtype=np.float32)

    t0 = time.perf_counter()
    for i in range(sequences):
        _, _, xi, ti, ci = gen.sample_sequence(seq_len)
        x[i] = np.clip(xi * 255.0 + 0.5, 0, 255).astype(np.uint8)
        toi[i] = ti
        cop[i] = ci
        if (i + 1) % 500 == 0 or i == sequences - 1:
            dt = time.perf_counter() - t0
            print(f"  generated {i + 1}/{sequences} sequences ({dt:.1f}s)")

    np.savez(path, x=x, toi=toi, cop=cop, seq_len=seq_len)
    print(f"[INFO] wrote dataset to {path}")


def load_dataset(path: str):
    data = np.load(path)
    return data["x"], data["toi"], data["cop"]


# ------------------------------------------------------------------ training
def target_tensor(toi: np.ndarray, cop: np.ndarray, eps: float = 0.005) -> torch.Tensor:
    t = np.stack([toi / TOI_MAX_MS, cop / FOOT_LEN_MM], axis=-1).astype(np.float32)
    t = np.clip(t, eps, 1.0 - eps)
    return torch.from_numpy(t)


def evaluate(model, x, toi, cop, idx, batch_size, device, max_batches=None):
    model.eval()
    n = len(idx)
    mae_toi = mae_cop = 0.0
    correct = total = 0
    with torch.inference_mode():
        for b, start in enumerate(range(0, n, batch_size)):
            if max_batches is not None and b >= max_batches:
                break
            sel = idx[start:start + batch_size]
            frames = torch.from_numpy(x[sel]).float().div_(255.0).to(device)
            state = model.initial_state(frames.shape[0], device)
            preds = []
            for t in range(frames.shape[1]):
                logits, state = model.forward_frame(frames[:, t], state)
                preds.append(torch.sigmoid(logits))
            pred = torch.stack(preds, dim=1)  # (B,T,2)
            p_toi = pred[..., 0].cpu().numpy() * TOI_MAX_MS
            p_cop = pred[..., 1].cpu().numpy() * FOOT_LEN_MM
            t_toi, t_cop = toi[sel], cop[sel]
            mae_toi += np.abs(p_toi - t_toi).sum()
            mae_cop += np.abs(p_cop - t_cop).sum()

            p_cls = np.where(p_toi > EMD_MS, 0, np.where(p_cop <= MID_FOOT_MM, 1, 2))
            t_cls = np.where(t_toi > EMD_MS, 0, np.where(t_cop <= MID_FOOT_MM, 1, 2))
            correct += int((p_cls == t_cls).sum())
            total += int(p_cls.size)
    model.train()
    return mae_toi / total, mae_cop / total, correct / max(total, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequences", type=int, default=6000)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=15)
    parser.add_argument("--hidden1", type=int, default=32)
    parser.add_argument("--hidden2", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--data", default=os.path.join("data", "synthetic_terrain.npz"))
    parser.add_argument("--out", default=os.path.join("weights", "terrain_forecaster.pt"))
    parser.add_argument("--regen", action="store_true", help="regenerate the dataset")
    parser.add_argument("--data-only", action="store_true", help="generate data and exit")
    parser.add_argument("--max-batches", type=int, default=0, help="limit batches (smoke test)")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.regen or not os.path.exists(args.data):
        generate_dataset(args.data, args.sequences, args.seq_len, args.seed)
    if args.data_only:
        return

    x, toi, cop = load_dataset(args.data)
    n = len(x)
    if n == 0:
        raise RuntimeError("dataset is empty")
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(n)
    split = max(1, int(0.9 * n))
    train_idx, val_idx = order[:split], order[split:]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TerrainForecaster(hidden1=args.hidden1, hidden2=args.hidden2,
                              num_outputs=2).to(device)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    print(f"[INFO] device={device} train={len(train_idx)} val={len(val_idx)} "
          f"seq_len={x.shape[1]} frames={x.shape[2:]}")

    best_val = float("inf")
    t_start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        rng.shuffle(train_idx)
        total_loss = 0.0
        batches = 0
        for start in range(0, len(train_idx), args.batch_size):
            if args.max_batches and batches >= args.max_batches:
                break
            sel = train_idx[start:start + args.batch_size]
            frames = torch.from_numpy(x[sel]).float().div_(255.0).to(device)
            target = target_tensor(toi[sel], cop[sel]).to(device)

            state = model.initial_state(frames.shape[0], device)
            loss = 0.0
            for t in range(frames.shape[1]):
                logits, state = model.forward_frame(frames[:, t], state)
                loss = loss + F.binary_cross_entropy_with_logits(logits, target[:, t])
            loss = loss / frames.shape[1]

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total_loss += float(loss.detach())
            batches += 1

        sched.step()
        train_loss = total_loss / max(batches, 1)
        toi_mae, cop_mae, acc = evaluate(model, x, toi, cop, val_idx,
                                         args.batch_size, device,
                                         max_batches=args.max_batches or None)
        elapsed = time.perf_counter() - t_start
        print(f"epoch {epoch:3d}/{args.epochs} | loss {train_loss:.4f} | "
              f"val TOI MAE {toi_mae:6.1f} ms | COP MAE {cop_mae:6.1f} mm | "
              f"terrain acc {acc * 100:5.1f}% | {elapsed:6.1f}s")

        if toi_mae + cop_mae < best_val:
            best_val = toi_mae + cop_mae
            os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
            torch.save({
                "state_dict": model.state_dict(),
                "config": {
                    "hidden1": args.hidden1,
                    "hidden2": args.hidden2,
                    "num_outputs": 2,
                    "input_height": HEIGHT,
                    "input_width": WIDTH,
                    "seq_len": int(x.shape[1]),
                    "toi_max_ms": TOI_MAX_MS,
                    "cop_max_mm": FOOT_LEN_MM,
                    "emd_ms": EMD_MS,
                    "preprocess": "INTER_AREA resize to 50x25, /255.0",
                    "targets": ["toi_norm", "cop_norm"],
                },
                "metrics": {"toi_mae_ms": toi_mae, "cop_mae_mm": cop_mae,
                            "terrain_acc": acc},
            }, args.out)

    print(f"[INFO] best checkpoint saved to {args.out} (val {best_val:.2f})")


if __name__ == "__main__":
    main()
